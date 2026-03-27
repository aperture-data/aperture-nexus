"""
Memory — storage and retrieval engine for aperture-nexus.

Memory is the only component that writes to and reads from ApertureDB.
It receives an authenticated Principal via Context (never authenticates
itself) and stores multimodal information losslessly at full fidelity.

Retrieval is filter-based and semantic — no memory tiers, no lossy
summarisation. The same stored data is accessible across sessions,
participants, and time horizons via filters and vector search.

ApertureDB entity classes used here:
    NexusSession — one per unique session_id
    NexusContext — one per Context passed to commit()

ApertureDB connection classes used here:
    nexus_session_context — Session → Context
    nexus_entry           — Context → Blob / Image / Video
    nexus_link            — Context → Context (user-defined relationship)

Multimodal content storage:
    Text  → AddBlob   (document_type="text", context_id/session_id stamped)
    Image → AddImage  (context_id/session_id stamped)
    Video → AddVideo  (context_id/session_id stamped)
    Blob  → AddBlob   (document_type stamped, context_id/session_id stamped)
    Embedding → AddDescriptor in per-modality+model DescriptorSet

DescriptorSet names: nexus_{modality}__{model_name}  (model name used verbatim)
    e.g. nexus_text__text-embedding-3-small
         nexus_image__ViT-B/16
         nexus_video__ViT-B/16
"""

from __future__ import annotations

import asyncio
import hashlib
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
from aperture_nexus.auth import Principal, validate_credentials
from aperture_nexus.exceptions import (
    NexusConfigError,
    NexusConnectionError,
    NexusPermissionError,
    NexusProcessingError,
    NexusStorageError,
    NexusValidationError,
)
from aperture_nexus.information import Information
from aperture_nexus.tasks import MemoryTask

# ApertureDB entity class for user principals (shared with admin.py)
_CLASS_USER = "NexusUser"

logger = logging.getLogger(__name__)

# ApertureDB entity class names
_CLASS_SESSION = "NexusSession"
_CLASS_CONTEXT = "NexusContext"

# ApertureDB connection class names (snake_case, nexus-prefixed)
_CONN_SESSION_CONTEXT = "nexus_session_context"  # Session → Context
_CONN_ENTRY = "nexus_entry"                       # Context → Blob/Image/Video
_CONN_LINK = "nexus_link"                         # Context → Context


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """One result from Memory.search().

    Attributes:
        score: Similarity score (higher = more similar).
        modality: ``"text"`` | ``"image"`` | ``"video"`` | ``"blob"``
        session_id: Session the entry was committed under.
        context_id: Context the entry was committed under.
        user_id: Principal who committed this entry.
        created_at: When the entry was committed.
        text: Short text content stored inline on the Descriptor
            (set when modality is ``"text"`` and text is short).
        start_frame: First frame of the matched video clip.
        stop_frame: Last frame of the matched video clip.
        metadata: Additional properties from ApertureDB.
    """

    score: float
    modality: str
    session_id: str
    context_id: str
    user_id: Optional[str]
    created_at: datetime

    text: Optional[str] = None
    start_frame: Optional[int] = None
    stop_frame: Optional[int] = None

    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_response(response, operation: str) -> None:
    """Raise NexusStorageError if any command in the response failed.

    Handles both the normal list-of-dicts format and the bare-dict format
    that ApertureDB returns for schema/parameter errors.
    """
    # ApertureDB returns a bare dict (not wrapped in a list) for invalid queries
    if isinstance(response, dict):
        status = response.get("status", -1)
        if status != 0:
            raise NexusStorageError(
                f"{operation} failed (status={status}): "
                f"{response.get('info', 'no details')}. "
                f"Check your ApertureDB connection and schema."
            )
        return
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
            "with_class": entity_class,
            "constraints": constraints,
            "results": {"count": True},
        }
    }]
    response, _ = db.query(cmd)
    if isinstance(response, dict):
        return False  # error response — entity does not exist
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


def _dset_name(modality: str, model: str) -> str:
    """Return the DescriptorSet name for a given modality and model.

    The model name is used verbatim — no sanitization. Invalid characters
    (double-quotes, backslashes, control chars) are rejected at log() time
    by _validate_embedding() so they never reach here.
    """
    return f"nexus_{modality}__{model}"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory:
    """Storage and retrieval engine for aperture-nexus.

    The primary runtime interface. Handles authentication, commit,
    search, and connection operations. Requires regular (non-admin)
    ApertureDB credentials — admin credentials are never needed at
    session time.

    Calling ``commit()`` multiple times on the same ``Context`` and
    ``Information`` buffer is the standard pattern for mid-session
    checkpoints and periodic flushes. The ``Information`` buffer
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
        import os
        from aperture_nexus import Memory, Context, Information

        memory = Memory()
        principal = memory.authenticate(
            user_id="alice",
            api_key=os.environ["NEXUS_API_KEY"],
        )
        ctx = Context(principal=principal, session_name="support-001")
        info = Information(context_id=ctx.id)
        info.log(text="Customer says order #4821 never arrived")
        context_id = memory.commit(ctx, info)
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
        self._schema_ensured = False
        # Track which contexts have had their nexus_session_context connection written.
        # Using an in-memory set means first-call-per-context writes the connection;
        # subsequent calls within the same Memory instance are no-ops.
        self._ensured_contexts: set[str] = set()
        logger.debug("Memory engine initialised")

    def ensure_schema(self) -> None:
        """Create property indexes for fast constraint lookups — idempotent.

        Called automatically on the first ``commit()``. Safe to call
        explicitly at startup to pre-warm the schema before first use.

        ApertureDB returns status=2 when an index already exists — treated
        as success so this is safe to call on every deployment.
        """
        if self._schema_ensured:
            return
        indexes = [
            (_CLASS_SESSION, "session_id"),
            (_CLASS_CONTEXT, "id"),
            (_CLASS_CONTEXT, "session_id"),
            (_CLASS_CONTEXT, "user_id"),
        ]
        for cls, prop in indexes:
            cmd = [{"CreateIndex": {
                "index_type": "entity",
                "class": cls,
                "property_key": prop,
            }}]
            response, _ = self._db.query(cmd)
            for item in response:
                for _, body in item.items():
                    status = body.get("status", -1) if isinstance(body, dict) else -1
                    if status not in (0, 2):
                        info = body.get("info", "no details") if isinstance(body, dict) else str(body)
                        logger.warning(
                            "CreateIndex %r.%r returned status=%d: %s",
                            cls, prop, status, info,
                        )
            logger.debug("Ensured index on %s.%s", cls, prop)
        self._schema_ensured = True

    # ------------------------------------------------------------------
    # authenticate()
    # ------------------------------------------------------------------

    def authenticate(self, user_id: str, api_key: str) -> Principal:
        """Validate credentials and return an authenticated Principal.

        Looks up the ``NexusUser`` entity for ``user_id`` and compares
        the SHA-256 hash of ``api_key`` against the stored hash. Requires
        only regular (non-admin) ApertureDB credentials.

        In normal use, ``api_key`` comes from the ``NEXUS_API_KEY``
        environment variable written to ``.env`` by ``adb-nexus init``.

        Args:
            user_id: The user's unique identifier.
            api_key: The API key issued at setup time via
                ``adb-nexus init`` or ``NexusAdmin.create_principal()``.

        Returns:
            An authenticated ``Principal`` ready to be attached to a
            ``Context``.

        Raises:
            NexusPermissionError: If credentials are invalid or the
                user does not exist.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            import os
            principal = memory.authenticate(
                user_id="alice",
                api_key=os.environ["NEXUS_API_KEY"],
            )
            ctx = Context(principal=principal, session_name="s-001")
        """
        user_id = validate_credentials(user_id, api_key)

        cmd = [{
            "FindEntity": {
                "with_class": _CLASS_USER,
                "constraints": {"user_id": ["==", user_id]},
                "results": {"all_properties": True},
            }
        }]
        try:
            response, _ = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB query failed during authentication: {e}. "
                f"Run 'adb-nexus validate' to check your connection."
            ) from e

        if isinstance(response, dict):
            raise NexusConnectionError(
                f"ApertureDB returned an unexpected response during authentication. "
                f"Run 'adb-nexus validate' to check your connection."
            )
        body = response[0].get("FindEntity", {})
        entities = body.get("entities", [])

        if not entities:
            raise NexusPermissionError(
                f"Authentication failed for user_id={user_id!r}. "
                f"User does not exist or credentials are invalid."
            )

        record = entities[0]
        stored_hash = record.get("api_key_hash", "")
        if stored_hash != hashlib.sha256(api_key.encode()).hexdigest():
            raise NexusPermissionError(
                f"Authentication failed for user_id={user_id!r}. "
                f"Invalid credentials."
            )

        principal = Principal(
            user_id=record["user_id"],
            user_name=record.get("user_name"),
            department=record.get("department"),
            organization=record.get("organization"),
        )
        logger.debug("Authenticated principal %r", user_id)
        return principal

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
            The ``context_id`` of the committed context.

        Raises:
            NexusValidationError: If ``info`` has no entries.
            NexusStorageError: If ApertureDB rejects any write.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            context_id = memory.commit(ctx, info)
            # info is now empty and ready for more log() calls
        """
        if not info._entries:
            raise NexusValidationError(
                "Information has no entries to commit. "
                "Call info.log() at least once before committing."
            )

        self.ensure_schema()

        entries = list(info._entries)          # snapshot before drain
        pending_conns = list(info._pending_connections)
        session_id = self._resolve_session_id(ctx)

        try:
            # Ensure session and context entities exist
            self._ensure_session(ctx, session_id)
            self._ensure_context(ctx, session_id)

            # Write each content entry
            for entry in entries:
                self._write_entry(entry, ctx, session_id)

            # Write any connections buffered via info.connect()
            for pc in pending_conns:
                self.connect(ctx, pc.target_id, pc.relationship, pc.properties or {})

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
            "Committed context_id=%r for session_id=%r",
            ctx.id, session_id,
        )
        return ctx.id

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
            The ``context_id`` of the committed context.

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
            context_id = memory.process_and_commit(ctx, info)

            # With model configured in aperture_nexus.json
            info.log(text="hello")
            context_id = memory.process_and_commit(ctx, info)
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
                print(task.context_id)
        """
        task_id = str(uuid.uuid4())
        task = MemoryTask(task_id=task_id)
        self._tasks[task_id] = task

        # Drain the info buffer immediately so the caller can keep using it
        entries_snapshot = list(info._entries)
        conns_snapshot = list(info._pending_connections)
        info._drain()
        # Rebuild a fresh Information object with the snapshotted entries
        info_copy = Information.__new__(Information)
        info_copy.context_id = info.context_id
        info_copy._entries = entries_snapshot
        info_copy._pending_connections = conns_snapshot

        async def _run() -> None:
            task._mark_processing()
            try:
                self.process_and_commit(ctx, info_copy)
                task._mark_complete(ctx.id)
            except Exception as exc:
                task._mark_failed(exc)

        async def _retry() -> None:
            info_copy._entries = list(entries_snapshot)
            info_copy._pending_connections = list(conns_snapshot)
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
            vector, resolved_modality, filters or {}, k, min_score,
            embedding_model=embedding_model,
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
        """Create a named relationship between two contexts.

        Writes a ``nexus_link`` connection from source to target with
        a ``type`` property. Both source and target must be ``NexusContext``
        entities — pass a ``Context`` object or its ``context_id`` string.

        Args:
            source: Source ``Context`` object or ``context_id`` string.
            target: Target ``Context`` object or ``context_id`` string.
            relationship: Relationship type, e.g. ``"follows"``,
                ``"caused_by"``, ``"references"``.
            properties: Optional extra properties on the connection.

        Raises:
            NexusValidationError: If ``relationship`` is empty.
            NexusStorageError: If ApertureDB rejects the write.

        Example:
            memory.connect(ctx_q1, ctx_q2, relationship="follows")
            memory.connect("ctx-id-1", "ctx-id-2", relationship="caused_by")
        """
        if not isinstance(relationship, str) or not relationship.strip():
            raise NexusValidationError(
                "relationship must be a non-empty string."
            )

        src_id = source.id if isinstance(source, Context) else source
        dst_id = target.id if isinstance(target, Context) else target

        conn_props: dict = {
            "type": relationship.strip(),
            "created_at": datetime.utcnow().isoformat(),
        }
        if properties:
            conn_props.update(properties)

        cmd = [
            {
                "FindEntity": {
                    "with_class": _CLASS_CONTEXT,
                    "constraints": {"id": ["==", src_id]},
                    "_ref": 1,
                    "results": {"count": True},
                }
            },
            {
                "FindEntity": {
                    "with_class": _CLASS_CONTEXT,
                    "constraints": {"id": ["==", dst_id]},
                    "_ref": 2,
                    "results": {"count": True},
                }
            },
            {
                "AddConnection": {
                    "class": _CONN_LINK,
                    "src": 1,
                    "dst": 2,
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

    def remove(self, context_id: str) -> None:
        """Remove a committed context from ApertureDB.

        Deletes the ``NexusContext`` entity. Associated blobs, images,
        videos, and descriptors retain their ``context_id`` property and
        can be cleaned up separately by filtering on ``context_id``.

        Args:
            context_id: The context ID returned by ``commit()``.

        Raises:
            NexusValidationError: If ``context_id`` is empty.
            NexusStorageError: If ApertureDB rejects the delete.

        Example:
            memory.remove(ctx.id)
        """
        if not isinstance(context_id, str) or not context_id.strip():
            raise NexusValidationError(
                "context_id must be a non-empty string."
            )
        cmd = [{
            "DeleteEntity": {
                "with_class": _CLASS_CONTEXT,
                "constraints": {"id": ["==", context_id]},
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"remove({context_id!r})")
        logger.debug("Removed context_id=%r", context_id)

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
        """Create a NexusSession entity if one doesn't exist yet."""
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
        cmd = [{"AddEntity": {
            "class": _CLASS_SESSION,
            "properties": props,
            "if_not_found": {"session_id": ["==", session_id]},
        }}]
        response, _ = self._db.query(cmd)
        _check_response(response, "ensure_session")

    def _ensure_context(self, ctx: Context, session_id: str) -> None:
        """Create a NexusContext entity and connect it to its session."""
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
        cmd = [{"AddEntity": {
            "class": _CLASS_CONTEXT,
            "properties": props,
            "if_not_found": {"id": ["==", ctx.id]},
        }}]
        response, _ = self._db.query(cmd)
        _check_response(response, "ensure_context")

        if ctx.id not in self._ensured_contexts:
            conn_cmd = [
                {
                    "FindEntity": {
                        "with_class": _CLASS_SESSION,
                        "constraints": {"session_id": ["==", session_id]},
                        "_ref": 1,
                        "results": {"count": True},
                    }
                },
                {
                    "FindEntity": {
                        "with_class": _CLASS_CONTEXT,
                        "constraints": {"id": ["==", ctx.id]},
                        "_ref": 2,
                        "results": {"count": True},
                    }
                },
                {
                    "AddConnection": {
                        "class": _CONN_SESSION_CONTEXT,
                        "src": 1,
                        "dst": 2,
                        "properties": {"created_at": datetime.utcnow().isoformat()},
                    }
                },
            ]
            response, _ = self._db.query(conn_cmd)
            _check_response(response, "nexus_session_context")
            self._ensured_contexts.add(ctx.id)

    def _write_entry(self, entry, ctx: Context, session_id: str) -> None:
        """Write one _LogEntry to ApertureDB."""
        common_props = {
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        if entry.metadata:
            common_props.update(entry.metadata)

        if entry.text is not None:
            text_bytes = entry.text.encode("utf-8")
            props = dict(common_props, document_type="text")
            if len(entry.text) <= 2000:
                # Short text: also store inline on Descriptor (done in _write_descriptor)
                props["text_preview"] = entry.text[:500]
            cmd = [{"AddBlob": {"properties": props}}]
            response, _ = self._db.query(cmd, [text_bytes])
            _check_response(response, "write_text_entry")

        if entry.image is not None:
            image_bytes = _to_image_bytes(entry.image)
            props = dict(common_props)
            cmd = [{"AddImage": {"properties": props}}]
            response, _ = self._db.query(cmd, [image_bytes])
            _check_response(response, "write_image_entry")

        if entry.video is not None:
            video_bytes = _to_video_bytes(entry.video)
            props = dict(common_props)
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
            self._write_descriptor(entry, ctx, session_id)

        if entry.video_clip_embeddings is not None and entry.embedding_model:
            self._write_video_clip_descriptors(entry, ctx, session_id)

    def _write_descriptor(self, entry, ctx: Context, session_id: str) -> None:
        """Write a pre-computed embedding as an ApertureDB Descriptor."""
        modality = self._infer_modality_from_entry(entry)
        dset = _dset_name(modality, entry.embedding_model)

        self._ensure_descriptor_set(
            dset, entry.embedding, modality, entry.embedding_model,
            self._cfg.processing.descriptor_metric,
            self._cfg.processing.descriptor_engine,
        )

        props: dict = {
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "embedding_model": entry.embedding_model,
            "modality": modality,
            "created_at": datetime.utcnow().isoformat(),
        }
        if entry.metadata:
            props.update(entry.metadata)
        if entry.text and len(entry.text) <= 2000:
            props["text"] = entry.text   # inline for short text

        emb_bytes = _embedding_to_bytes(entry.embedding)
        cmd = [{"AddDescriptor": {"set": dset, "properties": props}}]
        response, _ = self._db.query(cmd, [emb_bytes])
        _check_response(response, f"write_descriptor({dset})")

    def _write_video_clip_descriptors(
        self, entry, ctx: Context, session_id: str
    ) -> None:
        """Write one Descriptor per clip segment for a video entry."""
        dset = _dset_name("video", entry.embedding_model)
        first_emb = entry.video_clip_embeddings[0][0]
        self._ensure_descriptor_set(
            dset, first_emb, "video", entry.embedding_model,
            self._cfg.processing.descriptor_metric,
            self._cfg.processing.descriptor_engine,
        )

        for clip_emb, clip_meta in entry.video_clip_embeddings:
            props: dict = {
                "context_id": ctx.id,
                "session_id": session_id,
                "user_id": ctx.principal.user_id,
                "embedding_model": entry.embedding_model,
                "modality": "video",
                "start_frame": clip_meta["start_frame"],
                "stop_frame": clip_meta["stop_frame"],
                "created_at": datetime.utcnow().isoformat(),
            }
            if entry.metadata:
                props.update(entry.metadata)
            emb_bytes = _embedding_to_bytes(clip_emb)
            cmd = [{"AddDescriptor": {"set": dset, "properties": props}}]
            response, _ = self._db.query(cmd, [emb_bytes])
            _check_response(response, f"write_video_clip_descriptor(frames {clip_meta['start_frame']}-{clip_meta['stop_frame']})")

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
        modality: str, embedding_model: str,
        metric: str, engine: str,
    ) -> None:
        """Create a DescriptorSet if it doesn't exist."""
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
                "properties": {
                    "embeddings_model": embedding_model,
                    "modality": modality,
                },
            }
        }]
        response, _ = self._db.query(create_cmd)
        _check_response(response, f"create DescriptorSet {set_name!r}")
        logger.debug(
            "Created DescriptorSet %r (dims=%d metric=%s engine=%s model=%s)",
            set_name, dimensions, metric, engine, embedding_model,
        )

    def _ensure_descriptor_sets(self, entries: list) -> None:
        """Ensure all required DescriptorSets exist before writing."""
        for entry in entries:
            if entry.embedding is not None and entry.embedding_model:
                modality = self._infer_modality_from_entry(entry)
                dset = _dset_name(modality, entry.embedding_model)
                self._ensure_descriptor_set(
                    dset, entry.embedding, modality, entry.embedding_model,
                    self._cfg.processing.descriptor_metric,
                    self._cfg.processing.descriptor_engine,
                )
            if entry.video_clip_embeddings and entry.embedding_model:
                first_emb = entry.video_clip_embeddings[0][0]
                dset = _dset_name("video", entry.embedding_model)
                self._ensure_descriptor_set(
                    dset, first_emb, "video", entry.embedding_model,
                    self._cfg.processing.descriptor_metric,
                    self._cfg.processing.descriptor_engine,
                )

    # ------------------------------------------------------------------
    # Internal: embedding generation
    # ------------------------------------------------------------------

    def _generate_missing_embeddings(self, entries: list) -> None:
        """Generate embeddings for entries that don't have one yet.

        Calls the CLIP embedder for entries that lack a pre-computed
        embedding. If the configured model is not a CLIP model, raises
        NexusConfigError with a clear message.
        """
        from aperture_nexus._embeddings import is_clip_model, get_clip_embedder

        for entry in entries:
            if entry.embedding is not None or entry.video_clip_embeddings is not None:
                continue   # pre-computed — nothing to do

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
                continue   # blob-only — no embedding needed

            if not model:
                raise NexusConfigError(
                    f"No embedding model configured for {modality} input. "
                    f"Set models.{modality}_embedding in aperture_nexus.json "
                    f"(e.g. \"ViT-B/16\" for CLIP), or pass "
                    f"embedding=my_vector and embedding_model='...' to info.log(). "
                    f"Install CLIP with: pip install aperture-nexus[clip]. "
                    f"Run 'adb-nexus init' to regenerate your config."
                )

            if not is_clip_model(model):
                raise NexusConfigError(
                    f"Automatic embedding via model={model!r} is not yet available. "
                    f"Use a CLIP model (e.g. 'ViT-B/16') or pre-compute your "
                    f"embedding and pass it to info.log(embedding=my_vector, "
                    f"embedding_model={model!r})."
                )

            embedder = get_clip_embedder(model)
            try:
                if modality == "text":
                    entry.embedding = embedder.embed_text([entry.text])[0]
                elif modality == "image":
                    entry.embedding = embedder.embed_image(entry.image)
                else:  # video — per-clip embeddings, not a single mean-pool
                    entry.video_clip_embeddings = embedder.embed_video(
                        entry.video,
                        frame_interval=self._cfg.processing.video_frame_interval,
                        frames_per_clip=self._cfg.processing.video_frames_per_clip,
                    )
                    # entry.embedding stays None — written via _write_video_clip_descriptors
                entry.embedding_model = model
            except (NexusConfigError, NexusValidationError):
                raise
            except Exception as e:
                raise NexusProcessingError(
                    f"CLIP embedding failed for {modality} entry: {e}. "
                    f"Check that the model {model!r} is available."
                ) from e

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
                    "Set models.text_embedding in aperture_nexus.json "
                    "(e.g. \"ViT-B/16\"), or pass embedding_model= to "
                    "memory.search(). Install CLIP with: "
                    "pip install aperture-nexus[clip]."
                )
            from aperture_nexus._embeddings import is_clip_model, get_clip_embedder
            if not is_clip_model(model):
                raise NexusConfigError(
                    f"Automatic text embedding for search via model={model!r} "
                    f"is not yet available. Use a CLIP model (e.g. 'ViT-B/16') "
                    f"or pass query=my_vector and modality='text'."
                )
            embedder = get_clip_embedder(model)
            try:
                vector = embedder.embed_text([query])[0]
            except (NexusConfigError, NexusValidationError):
                raise
            except Exception as e:
                raise NexusProcessingError(
                    f"CLIP text embedding for search failed: {e}."
                ) from e
            return vector, "text"

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
        embedding_model: Optional[str] = None,
    ) -> list[SearchResult]:
        """Execute a KNN search on the appropriate DescriptorSet."""
        model = embedding_model or getattr(self._cfg.models, f"{modality}_embedding", None)
        if modality not in ("text", "image", "video"):
            raise NexusValidationError(
                f"Unknown modality {modality!r}. "
                f"Choose one of: 'text', 'image', 'video'."
            )
        dset = _dset_name(modality, model) if model else f"nexus_{modality}"

        constraints = self._build_constraints(filters)
        cmd_body: dict = {
            "set": dset,
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
                session_id=desc.get("session_id", ""),
                context_id=desc.get("context_id", ""),
                user_id=desc.get("user_id"),
                created_at=datetime.fromisoformat(
                    desc.get("created_at", datetime.utcnow().isoformat())
                ),
                text=desc.get("text"),
                start_frame=desc.get("start_frame"),
                stop_frame=desc.get("stop_frame"),
                metadata={
                    k: v for k, v in desc.items()
                    if k not in (
                        "session_id", "context_id", "user_id",
                        "created_at", "text", "start_frame", "stop_frame",
                        "modality", "embedding_model",
                    )
                },
            ))
        return results

    def _search_by_metadata(
        self, filters: dict, k: int
    ) -> list[SearchResult]:
        """Return Context entities matching metadata filters."""
        constraints = self._build_constraints(filters)
        cmd_body: dict = {
            "with_class": _CLASS_CONTEXT,
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

        if isinstance(response, dict):
            return []
        body = response[0].get("FindEntity", {})
        entities = body.get("entities", [])

        results = []
        for ent in entities:
            results.append(SearchResult(
                score=1.0,
                modality="text",
                session_id=ent.get("session_id", ""),
                context_id=ent.get("id", ""),
                user_id=ent.get("user_id"),
                created_at=datetime.fromisoformat(
                    ent.get("created_at", datetime.utcnow().isoformat())
                ),
                metadata={
                    k: v for k, v in ent.items()
                    if k not in (
                        "id", "session_id", "user_id", "created_at"
                    )
                },
            ))
        return results

    @staticmethod
    def _build_constraints(filters: dict) -> dict:
        """Convert a user-supplied filters dict to ApertureDB constraints."""
        allowed = {
            "session_id", "user_id", "organization",
            "department", "purpose",
        }
        constraints = {}
        for key, value in filters.items():
            if key in allowed:
                constraints[key] = ["==", value]
        return constraints
