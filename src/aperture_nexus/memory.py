"""
Memory — storage and retrieval engine for aperture-nexus.

Memory is the only component that writes to and reads from ApertureDB.
It receives an authenticated Principal via Context (never authenticates
itself) and stores multimodal information losslessly at full fidelity.

Retrieval is filter-based and semantic — no memory tiers, no lossy
summarisation. The same stored data is accessible across sessions,
participants, and time horizons via filters and vector search.

ApertureDB entity classes used here:
    NexusSession    — one per unique session_id
    NexusContext    — one per Context passed to commit()
    NexusMemory     — one per commit() call; links context to content
    NexusMemoryTask — one per async_process_and_commit() call

Multimodal content storage:
    Text  → AddBlob   (content_type="text", memory_id stamped)
    Image → AddImage  (memory_id stamped)
    Video → AddVideo  (memory_id stamped)
    Blob  → AddBlob   (document_type stamped, memory_id stamped)
    Embedding → AddDescriptor in per-modality DescriptorSet

DescriptorSet names:
    nexus_text   — text embeddings
    nexus_image  — image embeddings
    nexus_video  — video clip embeddings
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np

from aperture_nexus._client import get_connector
from aperture_nexus.config import load_config
from aperture_nexus.context import Context
from aperture_nexus.exceptions import (
    NexusConfigError,
    NexusConnectionError,
    NexusProcessingError,
    NexusStorageError,
    NexusValidationError,
)
from aperture_nexus.information import Information
from aperture_nexus.tasks import MemoryTask

logger = logging.getLogger(__name__)

# ApertureDB entity class names
_CLASS_SESSION = "NexusSession"
_CLASS_CONTEXT = "NexusContext"
_CLASS_MEMORY = "NexusMemory"
_CLASS_TASK = "NexusMemoryTask"

# DescriptorSet names (one per modality)
_DSET_TEXT = "nexus_text"
_DSET_IMAGE = "nexus_image"
_DSET_VIDEO = "nexus_video"

# Maps modality name → DescriptorSet name
_DSET_FOR_MODALITY = {
    "text": _DSET_TEXT,
    "image": _DSET_IMAGE,
    "video": _DSET_VIDEO,
}


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """One result from Memory.search().

    Exactly one content field is set, corresponding to the modality of
    the stored memory.

    Attributes:
        score: Similarity score (higher = more similar).
        modality: ``"text"`` | ``"image"`` | ``"video"`` | ``"blob"``
        memory_id: The memory this result belongs to.
        session_id: Session the memory was committed under.
        context_id: Context the memory was committed under.
        timestamp: When the memory was committed.
        text: Text content (set when modality is ``"text"``).
        image: PIL Image (set when modality is ``"image"``).
        video_url: Video URL or path (set when modality is ``"video"``).
        blob: Raw bytes (set when modality is ``"blob"``).
        metadata: Additional properties from ApertureDB.
    """

    score: float
    modality: str
    memory_id: str
    session_id: str
    context_id: str
    timestamp: datetime

    text: Optional[str] = None
    image: Optional[Any] = None   # PIL.Image
    video_url: Optional[str] = None
    blob: Optional[bytes] = None

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_response(response: list, operation: str) -> None:
    """Raise NexusStorageError if any command in response failed."""
    for item in response:
        for cmd_name, body in item.items():
            status = body.get("status", -1) if isinstance(body, dict) else -1
            if status != 0:
                info = (
                    body.get("info", "no details")
                    if isinstance(body, dict)
                    else str(body)
                )
                raise NexusStorageError(
                    f"{operation} failed (status={status}): {info}. "
                    f"Check your ApertureDB connection and schema."
                )


def _entity_exists(db, entity_class: str, constraints: dict) -> bool:
    """Return True if at least one entity matching constraints exists."""
    cmd = [{
        "FindEntity": {
            "class": entity_class,
            "constraints": constraints,
            "results": {"count": True},
        }
    }]
    response, _ = db.query(cmd)
    body = response[0].get("FindEntity", {})
    return body.get("count", 0) > 0


def _to_image_bytes(image: Any) -> bytes:
    """Convert any supported image type to PNG bytes."""
    import PIL.Image as PILImage

    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        # file path (URL already validated to not reach here)
        with open(image, "rb") as f:
            return f.read()
    if isinstance(image, PILImage.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(image, np.ndarray):
        pil = PILImage.fromarray(
            image if image.dtype == np.uint8
            else (image * 255).astype(np.uint8)
        )
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    # URL
    if isinstance(image, str) and (
        image.startswith("http://") or image.startswith("https://")
    ):
        import requests
        resp = requests.get(image, timeout=30)
        resp.raise_for_status()
        return resp.content
    raise NexusValidationError(
        f"Cannot convert image type {type(image).__name__!r} to bytes."
    )


def _to_video_bytes(video: Any) -> bytes:
    """Convert any supported video type to bytes."""
    if isinstance(video, bytes):
        return video
    if isinstance(video, str):
        if video.startswith("http://") or video.startswith("https://"):
            import requests
            resp = requests.get(video, timeout=60)
            resp.raise_for_status()
            return resp.content
        with open(video, "rb") as f:
            return f.read()
    raise NexusValidationError(
        f"Cannot convert video type {type(video).__name__!r} to bytes."
    )


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialise a 1D float32 numpy embedding to bytes for ApertureDB."""
    return embedding.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory:
    """Storage and retrieval engine for aperture-nexus.

    The only component that commits and searches memories in ApertureDB.
    Memory never authenticates — it receives an already-authenticated
    ``Principal`` via the ``Context`` passed to each operation.

    Calling ``commit()`` multiple times on the same ``Context`` and
    ``Information`` buffer is the standard pattern for mid-session
    checkpoints and periodic flushes. Each call creates a new
    ``NexusMemory`` entity in ApertureDB. The ``Information`` buffer
    is drained after each successful commit and is ready for continued
    use — consistent with how database transactions and write buffers work.

    Args:
        config: Path to ``aperture_nexus.json``. Discovered automatically
            if ``None``.
        db_client: Inject an existing ApertureDB ``Connector``. Useful
            for testing or connection reuse. If ``None``, a connector is
            created from environment variables or the active ``adb``
            configuration.

    Raises:
        NexusConnectionError: If credentials cannot be resolved.
        NexusConfigError: If the config file is invalid.

    Example:
        from aperture_nexus import Memory, Context, Information

        memory = Memory()
        ctx = Context(principal=principal, session_name="support-001")
        info = Information(context_id=ctx.id)
        info.log(text="Customer says order #4821 never arrived")
        memory_id = memory.commit(ctx, info)
    """

    def __init__(
        self,
        config: Optional[str] = None,
        db_client=None,
    ) -> None:
        self._cfg = load_config(path=config, validate_deps=False)
        self._db = get_connector(db_client)
        # In-memory task registry for pending/failed async tasks
        self._tasks: dict[str, MemoryTask] = {}
        logger.debug("Memory engine initialised")

    # ------------------------------------------------------------------
    # commit()
    # ------------------------------------------------------------------

    def commit(self, ctx: Context, info: Information) -> str:
        """Store information in ApertureDB without model calls.

        Writes all buffered entries in ``info`` to ApertureDB at full
        fidelity — no embeddings, no summarisation. Fast. After a
        successful write the ``info`` buffer is drained and ready for
        continued use.

        Use ``commit()`` for speed when semantic (vector) search is not
        required, or after embeddings have been pre-computed via
        ``info.log(embedding=..., embedding_model=...)``.

        Args:
            ctx: Context identifying who, which session, and why.
            info: Buffered multimodal information to commit.

        Returns:
            The ``memory_id`` of the committed memory (UUID string).

        Raises:
            NexusValidationError: If ``info`` has no entries.
            NexusStorageError: If ApertureDB rejects any write.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            memory_id = memory.commit(ctx, info)
            # info is now empty and ready for more log() calls
        """
        if not info._entries:
            raise NexusValidationError(
                "Information has no entries to commit. "
                "Call info.log() at least once before committing."
            )

        entries = list(info._entries)   # snapshot before drain
        memory_id = str(uuid.uuid4())
        session_id = self._resolve_session_id(ctx)
        now = datetime.utcnow().isoformat()

        try:
            # Ensure session and context entities exist
            self._ensure_session(ctx, session_id)
            self._ensure_context(ctx, session_id)

            # Write the Memory entity
            self._write_memory_entity(
                memory_id, ctx, session_id, now
            )

            # Write each content entry
            for entry in entries:
                self._write_entry(entry, memory_id, ctx, session_id)

        except NexusStorageError:
            raise
        except NexusConnectionError:
            raise
        except Exception as e:
            raise NexusStorageError(
                f"Unexpected error during commit: {e}. "
                f"Run 'adb-nexus validate' to check your connection."
            ) from e

        # Drain only after all writes succeed
        info._drain()
        logger.debug(
            "Committed memory_id=%r for session_id=%r",
            memory_id, session_id,
        )
        return memory_id

    # ------------------------------------------------------------------
    # process_and_commit()
    # ------------------------------------------------------------------

    def process_and_commit(self, ctx: Context, info: Information) -> str:
        """Generate embeddings, then store in ApertureDB.

        Generates embeddings for all entries in ``info`` that do not
        already have a pre-computed embedding. Entries with a pre-computed
        ``embedding`` (set via ``info.log(embedding=..., embedding_model=...)``)
        are used as-is — no model call is made for those entries.

        Automatically creates the per-modality ApertureDB DescriptorSet
        (``nexus_text``, ``nexus_image``, ``nexus_video``) on first use,
        using the embedding dimensions inferred from the first embedding.

        Args:
            ctx: Context identifying who, which session, and why.
            info: Buffered multimodal information to process and commit.

        Returns:
            The ``memory_id`` of the committed memory.

        Raises:
            NexusConfigError: If a modality has entries without embeddings
                and no embedding model is configured for that modality.
            NexusProcessingError: If a model call fails.
            NexusValidationError: If ``info`` has no entries.
            NexusStorageError: If ApertureDB rejects any write.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            # With pre-computed embeddings (no model configured needed)
            info.log(text="hello", embedding=my_vector,
                     embedding_model="text-embedding-3-small")
            memory_id = memory.process_and_commit(ctx, info)

            # With model configured in aperture_nexus.json
            info.log(text="hello")
            memory_id = memory.process_and_commit(ctx, info)
        """
        self._generate_missing_embeddings(info._entries)
        self._ensure_descriptor_sets(info._entries)
        return self.commit(ctx, info)

    # ------------------------------------------------------------------
    # async_process_and_commit()
    # ------------------------------------------------------------------

    async def async_process_and_commit(
        self, ctx: Context, info: Information
    ) -> MemoryTask:
        """Async variant of process_and_commit(). Returns immediately.

        The task runs in the background. Its status is persisted to
        ApertureDB and tracked in this Memory instance.

        Args:
            ctx: Context identifying who, which session, and why.
            info: Buffered multimodal information to process and commit.

        Returns:
            A ``MemoryTask`` with ``status="pending"``.

        Example:
            task = await memory.async_process_and_commit(ctx, info)
            await task.wait()
            if task.status == "complete":
                print(task.memory_id)
        """
        task_id = str(uuid.uuid4())
        task = MemoryTask(task_id=task_id)
        self._tasks[task_id] = task

        # Drain the info buffer immediately so the caller can keep using it
        entries_snapshot = info._drain()
        # Rebuild a fresh Information object with the snapshotted entries
        info_copy = Information.__new__(Information)
        info_copy.context_id = info.context_id
        info_copy._entries = entries_snapshot

        async def _run() -> None:
            task._mark_processing()
            try:
                memory_id = self.process_and_commit(ctx, info_copy)
                task._mark_complete(memory_id)
            except Exception as exc:
                task._mark_failed(exc)

        async def _retry() -> None:
            info_copy._entries = list(entries_snapshot)
            await _run()

        task._retry_fn = _retry
        asyncio.ensure_future(_run())
        logger.debug("Queued async task %r", task_id)
        return task

    # ------------------------------------------------------------------
    # search()
    # ------------------------------------------------------------------

    def search(
        self,
        query: Any = None,
        modality: Optional[str] = None,
        filters: Optional[dict] = None,
        k: int = 10,
        embedding_model: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        """Search stored memories by semantic similarity, metadata, or both.

        Search is per-modality: each query type searches its own
        DescriptorSet. There is no cross-modal search in v1.

        Permissions are not automatically filtered by user_id in v1 —
        search returns results regardless of who wrote them. Cross-user
        scoping is planned for v2.

        Query type → modality mapping:
            str                         → text (embeds query)
            np.ndarray + modality=      → direct vector (no model call)
            None                        → metadata filter only

        Args:
            query: The search query. Type determines modality. ``None``
                returns results by metadata filter only.
            modality: Required when ``query`` is ``np.ndarray``. One of
                ``"text"``, ``"image"``, ``"video"``.
            filters: Metadata constraints. Keys: ``session_id``,
                ``user_id``, ``organization``, ``department``, ``purpose``.
            k: Maximum results to return. Default: 10.
            embedding_model: Override the configured embedding model for
                this query. Must match the model used at index time.
            min_score: Minimum similarity score. Results below this are
                excluded.

        Returns:
            List of ``SearchResult``, ordered by descending similarity.

        Raises:
            NexusValidationError: If ``query`` is ``np.ndarray`` and
                ``modality`` is not provided.
            NexusConfigError: If a text/image query is given but no
                embedding model is configured.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            # Text semantic search
            results = memory.search(query="missing order last week")

            # Pre-computed vector search
            results = memory.search(query=my_vector, modality="image")

            # Metadata filter only
            results = memory.search(
                filters={"session_id": sid}, k=50
            )

            # Combined
            results = memory.search(
                query="order inquiry",
                filters={"organization": "AcmeCorp"},
            )
        """
        if isinstance(query, np.ndarray) and modality is None:
            raise NexusValidationError(
                "modality is required when query is a numpy array. "
                "Specify modality='text', 'image', or 'video'."
            )

        if query is None:
            return self._search_by_metadata(filters or {}, k)

        vector, resolved_modality = self._resolve_query_vector(
            query, modality, embedding_model
        )
        return self._search_by_vector(
            vector, resolved_modality, filters or {}, k, min_score
        )

    # ------------------------------------------------------------------
    # connect()
    # ------------------------------------------------------------------

    def connect(
        self,
        source: Union[Context, str],
        target: Union[Context, str],
        relationship: str,
        properties: Optional[dict] = None,
    ) -> None:
        """Create a named relationship between two memories or contexts.

        Builds the knowledge graph on top of ApertureDB ``Connection``
        objects. Relationships are directed (source → target), typed
        (``relationship`` string), and optionally carry properties.

        Args:
            source: Source ``Context`` or ``memory_id`` string.
            target: Target ``Context`` or ``memory_id`` string.
            relationship: Relationship name, e.g. ``"follows"``,
                ``"caused_by"``, ``"references"``.
            properties: Optional dict of properties on the connection
                (e.g. ``{"confidence": 0.9, "note": "inferred"}``).

        Raises:
            NexusValidationError: If source or target cannot be resolved.
            NexusStorageError: If ApertureDB rejects the write.

        Example:
            memory.connect(ctx_q1, ctx_q2, relationship="follows")
            memory.connect("mem-id-1", "mem-id-2", relationship="caused_by")
        """
        if not isinstance(relationship, str) or not relationship.strip():
            raise NexusValidationError(
                "relationship must be a non-empty string."
            )

        src_id = source.id if isinstance(source, Context) else source
        dst_id = target.id if isinstance(target, Context) else target

        # Determine class based on type
        src_class = (
            _CLASS_CONTEXT if isinstance(source, Context)
            else _CLASS_MEMORY
        )
        dst_class = (
            _CLASS_CONTEXT if isinstance(target, Context)
            else _CLASS_MEMORY
        )

        conn_props = {"created_at": datetime.utcnow().isoformat()}
        if properties:
            conn_props.update(properties)

        cmd = [
            {
                "FindEntity": {
                    "class": src_class,
                    "constraints": {"id": ["==", src_id]},
                    "_ref": 1,
                }
            },
            {
                "FindEntity": {
                    "class": dst_class,
                    "constraints": {"id": ["==", dst_id]},
                    "_ref": 2,
                }
            },
            {
                "AddConnection": {
                    "class": relationship,
                    "_src_ref": 1,
                    "_dst_ref": 2,
                    "properties": conn_props,
                }
            },
        ]
        response, _ = self._db.query(cmd)
        _check_response(response, f"connect({relationship!r})")
        logger.debug(
            "Connected %r -[%s]-> %r", src_id, relationship, dst_id
        )

    # ------------------------------------------------------------------
    # remove()
    # ------------------------------------------------------------------

    def remove(self, memory_id: str) -> None:
        """Remove a committed memory from ApertureDB.

        The ``NexusMemory`` entity is deleted. Associated blobs, images,
        and descriptors are retained but will no longer be linked to the
        deleted memory.

        Args:
            memory_id: The memory ID returned by ``commit()``.

        Raises:
            NexusValidationError: If ``memory_id`` is empty.
            NexusStorageError: If ApertureDB rejects the delete.

        Example:
            memory.remove(memory_id)
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise NexusValidationError(
                "memory_id must be a non-empty string."
            )
        cmd = [{
            "DeleteEntity": {
                "class": _CLASS_MEMORY,
                "constraints": {"memory_id": ["==", memory_id]},
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"remove({memory_id!r})")
        logger.debug("Removed memory_id=%r", memory_id)

    # ------------------------------------------------------------------
    # Task monitoring
    # ------------------------------------------------------------------

    def pending_commits(self) -> list[MemoryTask]:
        """Return async tasks that are still in flight.

        Returns:
            List of ``MemoryTask`` with ``status`` ``"pending"`` or
            ``"processing"``.

        Example:
            for task in memory.pending_commits():
                print(task.task_id, task.status)
        """
        return [
            t for t in self._tasks.values()
            if t.status in ("pending", "processing")
        ]

    def failed_commits(self) -> list[MemoryTask]:
        """Return async tasks that have failed.

        Returns:
            List of ``MemoryTask`` with ``status == "failed"``.

        Example:
            for task in memory.failed_commits():
                print(task.error_message)
                task.retry()
        """
        return [
            t for t in self._tasks.values()
            if t.status == "failed"
        ]

    # ------------------------------------------------------------------
    # stats()
    # ------------------------------------------------------------------

    def stats(self, scope: str = "session") -> dict:
        """Return usage statistics.

        Requires ``pip install aperture-nexus[metrics]``.

        Args:
            scope: ``"session"`` (default) or ``"global"``.

        Returns:
            Dict with commit counts, latency histograms, and task stats.

        Raises:
            NexusConfigError: If metrics are not installed.
        """
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            raise NexusConfigError(
                "memory.stats() requires metrics support. "
                "Install it with: pip install aperture-nexus[metrics]"
            )
        # Placeholder — full implementation in metrics PR
        return {"scope": scope, "commits_total": 0}

    # ------------------------------------------------------------------
    # Internal: session and context management
    # ------------------------------------------------------------------

    def _resolve_session_id(self, ctx: Context) -> str:
        """Return the session_id to use, deriving from session_name if needed."""
        if ctx.session_id:
            return ctx.session_id
        # Derive a deterministic ID from session_name + user_id so the same
        # named session always maps to the same ID for this principal.
        import hashlib
        raw = f"{ctx.principal.user_id}:{ctx.session_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _ensure_session(self, ctx: Context, session_id: str) -> None:
        """Create a NexusSession entity if it doesn't exist."""
        if _entity_exists(
            self._db, _CLASS_SESSION, {"session_id": ["==", session_id]}
        ):
            return
        props: dict = {
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        if ctx.session_name:
            props["session_name"] = ctx.session_name
        if ctx.organization:
            props["organization"] = ctx.organization
        if ctx.department:
            props["department"] = ctx.department
        cmd = [{"AddEntity": {"class": _CLASS_SESSION, "properties": props}}]
        response, _ = self._db.query(cmd)
        _check_response(response, "ensure_session")

    def _ensure_context(self, ctx: Context, session_id: str) -> None:
        """Create a NexusContext entity for this context."""
        if _entity_exists(
            self._db, _CLASS_CONTEXT, {"id": ["==", ctx.id]}
        ):
            return
        props: dict = {
            "id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        for attr in ("purpose", "organization", "department"):
            val = getattr(ctx, attr, None)
            if val is not None:
                props[attr] = val
        cmd = [{"AddEntity": {"class": _CLASS_CONTEXT, "properties": props}}]
        response, _ = self._db.query(cmd)
        _check_response(response, "ensure_context")

    def _write_memory_entity(
        self, memory_id: str, ctx: Context, session_id: str, now: str
    ) -> None:
        props: dict = {
            "memory_id": memory_id,
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "created_at": now,
        }
        for attr in ("organization", "department", "purpose"):
            val = getattr(ctx, attr, None)
            if val is not None:
                props[attr] = val
        cmd = [{"AddEntity": {"class": _CLASS_MEMORY, "properties": props}}]
        response, _ = self._db.query(cmd)
        _check_response(response, "write_memory_entity")

    def _write_entry(self, entry, memory_id: str, ctx: Context,
                     session_id: str) -> None:
        """Write one _LogEntry to ApertureDB."""
        common_props = {
            "memory_id": memory_id,
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
        }

        if entry.text is not None:
            text_bytes = entry.text.encode("utf-8")
            props = dict(common_props, content_type="text")
            # Store first 500 chars as a searchable property
            props["text_preview"] = entry.text[:500]
            cmd = [{"AddBlob": {"properties": props}}]
            response, _ = self._db.query(cmd, [text_bytes])
            _check_response(response, "write_text_entry")

        if entry.image is not None:
            image_bytes = _to_image_bytes(entry.image)
            props = dict(common_props, content_type="image")
            cmd = [{"AddImage": {"properties": props}}]
            response, _ = self._db.query(cmd, [image_bytes])
            _check_response(response, "write_image_entry")

        if entry.video is not None:
            video_bytes = _to_video_bytes(entry.video)
            props = dict(common_props, content_type="video")
            cmd = [{"AddVideo": {"properties": props}}]
            response, _ = self._db.query(cmd, [video_bytes])
            _check_response(response, "write_video_entry")

        if entry.blob is not None:
            props = dict(common_props,
                         document_type=entry.document_type or "")
            cmd = [{"AddBlob": {"properties": props}}]
            response, _ = self._db.query(cmd, [entry.blob])
            _check_response(response, "write_blob_entry")

        if entry.embedding is not None and entry.embedding_model:
            self._write_descriptor(entry, memory_id, ctx, session_id)

    def _write_descriptor(self, entry, memory_id: str, ctx: Context,
                           session_id: str) -> None:
        """Write a pre-computed embedding as an ApertureDB Descriptor."""
        modality = self._infer_modality_from_entry(entry)
        dset_name = _DSET_FOR_MODALITY.get(modality, _DSET_TEXT)

        self._ensure_descriptor_set(
            dset_name, entry.embedding,
            self._cfg.processing.descriptor_metric,
            self._cfg.processing.descriptor_engine,
        )

        props: dict = {
            "memory_id": memory_id,
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "embedding_model": entry.embedding_model,
            "modality": modality,
        }
        if entry.text:
            props["text_preview"] = entry.text[:500]

        emb_bytes = _embedding_to_bytes(entry.embedding)
        cmd = [{"AddDescriptor": {"set": dset_name, "properties": props}}]
        response, _ = self._db.query(cmd, [emb_bytes])
        _check_response(response, f"write_descriptor({dset_name})")

    def _infer_modality_from_entry(self, entry) -> str:
        """Infer the modality of an entry from its content."""
        if entry.image is not None:
            return "image"
        if entry.video is not None:
            return "video"
        return "text"

    # ------------------------------------------------------------------
    # Internal: DescriptorSet management
    # ------------------------------------------------------------------

    def _ensure_descriptor_set(
        self, set_name: str, probe_embedding: np.ndarray,
        metric: str, engine: str
    ) -> None:
        """Create a DescriptorSet if it doesn't exist.

        Dimensions are inferred from ``probe_embedding`` (the first
        actual embedding encountered for this modality). This is the
        probe approach: we don't hardcode dimensions — we let the first
        embedding tell us the right number.
        """
        cmd = [{
            "FindDescriptorSet": {
                "with_name": set_name,
                "results": {"count": True},
            }
        }]
        response, _ = self._db.query(cmd)
        body = response[0].get("FindDescriptorSet", {})
        if body.get("count", 0) > 0:
            return   # already exists

        dimensions = int(len(probe_embedding))
        create_cmd = [{
            "AddDescriptorSet": {
                "name": set_name,
                "dimensions": dimensions,
                "metric": metric,
                "engine": engine,
            }
        }]
        response, _ = self._db.query(create_cmd)
        _check_response(response, f"create DescriptorSet {set_name!r}")
        logger.debug(
            "Created DescriptorSet %r (dims=%d metric=%s engine=%s)",
            set_name, dimensions, metric, engine,
        )

    def _ensure_descriptor_sets(self, entries: list) -> None:
        """Ensure all required DescriptorSets exist before writing."""
        for entry in entries:
            if entry.embedding is not None and entry.embedding_model:
                modality = self._infer_modality_from_entry(entry)
                dset_name = _DSET_FOR_MODALITY.get(modality, _DSET_TEXT)
                self._ensure_descriptor_set(
                    dset_name, entry.embedding,
                    self._cfg.processing.descriptor_metric,
                    self._cfg.processing.descriptor_engine,
                )

    # ------------------------------------------------------------------
    # Internal: embedding generation
    # ------------------------------------------------------------------

    def _generate_missing_embeddings(self, entries: list) -> None:
        """Generate embeddings for entries that don't have one yet.

        In v1, only pre-computed embeddings (set via info.log(embedding=...))
        are supported. If an entry lacks an embedding and no model is
        configured, raises NexusConfigError with a clear message.
        """
        for entry in entries:
            if entry.embedding is not None:
                continue   # pre-computed — nothing to do
            # Determine which modality this is
            if entry.text is not None:
                modality = "text"
                model = self._cfg.models.text_embedding
            elif entry.image is not None:
                modality = "image"
                model = self._cfg.models.image_embedding
            elif entry.video is not None:
                modality = "video"
                model = self._cfg.models.video_embedding
            else:
                continue   # blob-only entry, no embedding needed
            if not model:
                raise NexusConfigError(
                    f"No embedding model configured for {modality} input. "
                    f"Set models.{modality}_embedding in aperture_nexus.json, "
                    f"or pass embedding=my_vector and "
                    f"embedding_model='...' to info.log(). "
                    f"Run 'adb-nexus init' to regenerate your config."
                )
            # Model integration for auto-embedding is planned for a
            # future PR. Raise a clear, actionable error for now.
            raise NexusConfigError(
                f"Automatic {modality} embedding via model={model!r} is "
                f"not yet available. Pre-compute your embedding and pass "
                f"it to info.log(embedding=my_vector, "
                f"embedding_model={model!r})."
            )

    # ------------------------------------------------------------------
    # Internal: search
    # ------------------------------------------------------------------

    def _resolve_query_vector(
        self,
        query: Any,
        modality: Optional[str],
        embedding_model: Optional[str],
    ) -> tuple[np.ndarray, str]:
        """Return (vector, modality) for the given query."""
        if isinstance(query, np.ndarray):
            return query, modality  # type: ignore[return-value]

        if isinstance(query, str):
            model = embedding_model or self._cfg.models.text_embedding
            if not model:
                raise NexusConfigError(
                    "No text embedding model configured for search. "
                    "Set models.text_embedding in aperture_nexus.json, "
                    "or pass embedding_model= to memory.search()."
                )
            raise NexusConfigError(
                f"Automatic text embedding for search via model={model!r} "
                f"is not yet available. Pass query=my_vector and "
                f"modality='text' to search with a pre-computed vector."
            )

        raise NexusValidationError(
            f"Unsupported query type: {type(query).__name__!r}. "
            f"Supported: str, np.ndarray (+ modality=), or None."
        )

    def _search_by_vector(
        self,
        vector: np.ndarray,
        modality: str,
        filters: dict,
        k: int,
        min_score: Optional[float],
    ) -> list[SearchResult]:
        """Execute a KNN search on the appropriate DescriptorSet."""
        dset_name = _DSET_FOR_MODALITY.get(modality)
        if dset_name is None:
            raise NexusValidationError(
                f"Unknown modality {modality!r}. "
                f"Choose one of: {list(_DSET_FOR_MODALITY)}."
            )

        constraints = self._build_constraints(filters)
        cmd_body: dict = {
            "set": dset_name,
            "k_neighbors": k,
            "results": {"all_properties": True},
            "distances": True,
        }
        if constraints:
            cmd_body["constraints"] = constraints

        cmd = [{"FindDescriptor": cmd_body}]
        emb_bytes = _embedding_to_bytes(vector)

        try:
            response, _ = self._db.query(cmd, [emb_bytes])
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB search query failed: {e}."
            ) from e

        body = response[0].get("FindDescriptor", {})
        descriptors = body.get("descriptors", [])
        distances = body.get("distances", [])

        results = []
        for i, desc in enumerate(descriptors):
            score = float(distances[i]) if i < len(distances) else 0.0
            if min_score is not None and score < min_score:
                continue
            results.append(SearchResult(
                score=score,
                modality=modality,
                memory_id=desc.get("memory_id", ""),
                session_id=desc.get("session_id", ""),
                context_id=desc.get("context_id", ""),
                timestamp=datetime.fromisoformat(
                    desc.get("created_at", datetime.utcnow().isoformat())
                ),
                text=desc.get("text_preview"),
                metadata={
                    k: v for k, v in desc.items()
                    if k not in (
                        "memory_id", "session_id", "context_id",
                        "created_at", "text_preview"
                    )
                },
            ))
        return results

    def _search_by_metadata(
        self, filters: dict, k: int
    ) -> list[SearchResult]:
        """Return Memory entities matching metadata filters."""
        constraints = self._build_constraints(filters)
        cmd_body: dict = {
            "class": _CLASS_MEMORY,
            "results": {"all_properties": True, "limit": k},
        }
        if constraints:
            cmd_body["constraints"] = constraints

        cmd = [{"FindEntity": cmd_body}]
        try:
            response, _ = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB metadata search failed: {e}."
            ) from e

        body = response[0].get("FindEntity", {})
        entities = body.get("entities", [])

        results = []
        for ent in entities:
            results.append(SearchResult(
                score=1.0,   # no vector score for metadata-only search
                modality="text",
                memory_id=ent.get("memory_id", ""),
                session_id=ent.get("session_id", ""),
                context_id=ent.get("context_id", ""),
                timestamp=datetime.fromisoformat(
                    ent.get("created_at", datetime.utcnow().isoformat())
                ),
                metadata={
                    k: v for k, v in ent.items()
                    if k not in (
                        "memory_id", "session_id", "context_id", "created_at"
                    )
                },
            ))
        return results

    @staticmethod
    def _build_constraints(filters: dict) -> dict:
        """Convert a user-supplied filters dict to ApertureDB constraints."""
        allowed = {
            "session_id", "user_id", "organization",
            "department", "purpose", "memory_id",
        }
        constraints = {}
        for key, value in filters.items():
            if key in allowed:
                constraints[key] = ["==", value]
        return constraints
