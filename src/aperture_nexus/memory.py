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
_CONN_DESCRIPTOR = "nexus_descriptor"             # Descriptor → Blob/Image/Video


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """One content item returned by Memory.retrieve().

    Attributes:
        modality: ``"text"``, ``"image"``, ``"video"``, or ``"blob"``.
        context_id: Context the entry was committed under.
        session_id: Session the entry was committed under.
        created_at: When the entry was committed.
        text: Decoded text content (modality ``"text"`` only).
        data: Raw bytes (modality ``"image"``, ``"video"``,
            or ``"blob"`` only).
        document_type: File-type hint for blobs (e.g. ``"pdf"``,
            ``"mp3"``). ``None`` for non-blob modalities.
        properties: All other properties stored on the entry.
    """

    modality: str
    context_id: str
    session_id: str
    created_at: datetime

    text: Optional[str] = None
    data: Optional[bytes] = None
    document_type: Optional[str] = None

    properties: dict = field(default_factory=dict)


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

    # Internal — set by search(), consumed by memory.remove(results=...)
    _entry_id: Optional[str] = field(default=None, repr=False, compare=False)


@dataclass
class ContextResult:
    """One result from Memory.search_contexts().

    Returned by ``search_contexts()`` which searches context graph
    nodes by semantic similarity of their ``purpose`` field.

    Attributes:
        score: Similarity score (higher = more similar).
        context_id: The context ID.
        session_id: Session the context belongs to.
        user_id: Principal who created this context.
        purpose: The stated purpose of this context.
        created_at: When the context was committed.
        organization: Organization of the principal, if set.
        department: Department of the principal, if set.
    """

    score: float
    context_id: str
    session_id: str
    user_id: Optional[str]
    purpose: Optional[str]
    created_at: datetime

    organization: Optional[str] = None
    department: Optional[str] = None


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
            # status=0: success; status=2: "object already exists" (expected
            # from AddEntity/AddDescriptorSet with if_not_found — not an error)
            if status not in (0, 2):
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
        # URL must be checked before the file-path branch so that http/https
        # strings are downloaded rather than treated as local paths.
        if image.startswith("http://") or image.startswith("https://"):
            import requests
            resp = requests.get(image, timeout=30)
            resp.raise_for_status()
            return resp.content
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
    raise NexusValidationError(
        f"Cannot convert image type {type(image).__name__!r} to bytes. "
        "Pass a file path, URL, bytes, PIL Image, or numpy array."
    )


def _to_blob_bytes(blob: Any) -> bytes:
    """Convert any supported blob type to bytes."""
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, str):
        if blob.startswith("http://") or blob.startswith("https://"):
            import requests
            resp = requests.get(blob, timeout=60)
            resp.raise_for_status()
            return resp.content
        with open(blob, "rb") as f:
            return f.read()
    raise NexusValidationError(
        f"Cannot convert blob type {type(blob).__name__!r} to bytes. "
        "Pass a file path, URL, or bytes."
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
        f"Cannot convert video type {type(video).__name__!r} to bytes. "
        "Pass a file path, URL, or bytes."
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
            (_CLASS_CONTEXT, "nexus_ctx_id"),
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
            A ``commit_id`` string identifying this specific commit. Pass it to ``memory.remove(commit_id=...)`` to remove everything written in this call.

        Raises:
            NexusValidationError: If ``info`` has no entries.
            NexusStorageError: If ApertureDB rejects any write.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            commit_id = memory.commit(ctx, info)
            # info is now empty and ready for more log() calls
        """
        if not info._entries:
            raise NexusValidationError(
                "Information has no entries to commit. "
                "Call info.log() at least once before committing."
            )

        self.ensure_schema()
        commit_id = str(uuid.uuid4())

        entries = list(info._entries)          # snapshot before drain
        pending_conns = list(info._pending_connections)
        session_id = self._resolve_session_id(ctx)

        try:
            # Ensure session and context entities exist
            self._ensure_session(ctx, session_id)
            self._ensure_context(ctx, session_id)

            # Ensure all required DescriptorSets exist before writing.
            # Must happen after embeddings are generated (process_and_commit)
            # and before any AddDescriptor in _write_entry.
            self._ensure_descriptor_sets(entries)

            # Write each content entry
            for entry in entries:
                self._write_entry(entry, ctx, session_id, commit_id)

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
        return commit_id

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
            A ``commit_id`` string. See ``commit()``.

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
            commit_id = memory.process_and_commit(ctx, info)

            # With model configured in aperture_nexus.json
            info.log(text="hello")
            commit_id = memory.process_and_commit(ctx, info)
        """
        self._generate_missing_embeddings(info._entries)
        context_id = self.commit(ctx, info)
        session_id = self._resolve_session_id(ctx)
        self._write_context_embedding(ctx, session_id)
        return context_id

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
                print(task.commit_id)
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
                commit_id = self.process_and_commit(ctx, info_copy)
                task._mark_complete(commit_id)
            except Exception as exc:
                task._mark_failed(exc)

        async def _retry() -> None:
            info_copy._entries = list(entries_snapshot)
            info_copy._pending_connections = list(conns_snapshot)
            await _run()

        task._retry_fn = _retry
        asyncio.ensure_future(_run())
        logger.warning(
            "async_process_and_commit() has not been validated "
            "against a live ApertureDB instance. Use with caution."
        )
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
            filters: Metadata constraints. Supported keys:
                ``session_id``, ``session_name``, ``context_id``,
                ``user_id``, ``organization``, ``department``,
                ``purpose``. Unknown keys raise
                ``NexusValidationError``.
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
                "relationship must be a non-empty string. "
                "Examples: 'follows', 'caused_by', 'references'."
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
                    "constraints": {"nexus_ctx_id": ["==", src_id]},
                    "_ref": 1,
                    "results": {"count": True},
                }
            },
            {
                "FindEntity": {
                    "with_class": _CLASS_CONTEXT,
                    "constraints": {"nexus_ctx_id": ["==", dst_id]},
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
    # search_contexts()
    # ------------------------------------------------------------------

    def search_contexts(
        self,
        query: str,
        filters: Optional[dict] = None,
        k: int = 10,
        embedding_model: Optional[str] = None,
    ) -> list[ContextResult]:
        """Search committed contexts by semantic similarity of their purpose.

        Searches the ``nexus_context`` DescriptorSet populated by
        ``process_and_commit()`` when a context has a ``purpose`` set.
        Only contexts committed via ``process_and_commit()`` with a
        purpose appear here — ``commit()`` does not embed context nodes.

        Args:
            query: Text query to match against context purposes.
            filters: Metadata constraints applied to results. Supported
                keys: ``session_id``, ``user_id``, ``organization``,
                ``department``.
            k: Maximum results to return. Default: 10.
            embedding_model: Override the configured text embedding
                model. Must match the model used at index time.

        Returns:
            List of ``ContextResult``, ordered by descending similarity.

        Raises:
            NexusConfigError: If no text embedding model is configured
                and ``embedding_model`` is not provided.
            NexusProcessingError: If the embedding model call fails.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            results = memory.search_contexts("customer order inquiry")
            for r in results:
                print(r.purpose, r.session_id, r.score)
        """
        model = embedding_model or self._cfg.models.text_embedding
        if not model:
            raise NexusConfigError(
                "No text embedding model configured for search_contexts(). "
                "Set models.text_embedding in aperture_nexus.json "
                "(e.g. \"ViT-B/16\"), or pass embedding_model= to "
                "search_contexts(). Install CLIP with: "
                "pip install aperture-nexus[clip]."
            )

        from aperture_nexus._embeddings import is_clip_model, get_clip_embedder
        if not is_clip_model(model):
            raise NexusConfigError(
                f"Automatic embedding for search_contexts() via "
                f"model={model!r} is not yet available. Use a CLIP "
                f"model (e.g. 'ViT-B/16') or pass a pre-embedded "
                f"query vector via embedding_model= and query=np.ndarray."
            )

        embedder = get_clip_embedder(model)
        try:
            vector = embedder.embed_text([query])[0]
        except (NexusConfigError, NexusValidationError):
            raise
        except Exception as e:
            raise NexusProcessingError(
                f"CLIP text embedding for search_contexts failed: {e}."
            ) from e

        dset = _dset_name("context", model)
        constraints = self._build_constraints(filters or {})
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
                f"ApertureDB search_contexts query failed: {e}."
            ) from e

        if not isinstance(response, list) or not response:
            return []
        body = response[0].get("FindDescriptor", {})
        if body.get("status", 0) != 0:
            return []

        results = []
        for desc in body.get("entities", []):
            results.append(ContextResult(
                score=float(desc.get("_distance", 0.0)),
                context_id=desc.get("context_id", ""),
                session_id=desc.get("session_id", ""),
                user_id=desc.get("user_id"),
                purpose=desc.get("purpose"),
                organization=desc.get("organization"),
                department=desc.get("department"),
                created_at=datetime.fromisoformat(
                    desc.get("created_at", datetime.utcnow().isoformat())
                ),
            ))
        return results

    # ------------------------------------------------------------------
    # remove()
    # ------------------------------------------------------------------

    def remove(
        self,
        *,
        commit_id: Optional[str] = None,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None,
        before: Optional[datetime] = None,
        since: Optional[datetime] = None,
        results: Optional[list] = None,
    ) -> None:
        """Remove committed content from ApertureDB.

        Accepts one or more filters that determine what to remove. Filters
        AND together — e.g. ``before=ts, session_id=sid`` removes only old
        entries within that session. ``results=`` is exclusive and cannot
        be combined with other filters.

        Args:
            commit_id: Remove all content written in one ``commit()`` call.
                Pass the value returned by ``memory.commit()``.
            context_id: Remove all content committed under this context.
                Also removes the ``NexusContext`` entity when used alone.
            session_id: Remove all content committed under this session.
            before: Remove entries whose ``created_at`` is strictly before
                this UTC-aware datetime. Keeps recent entries.
            since: Remove entries whose ``created_at`` is at or after this
                UTC-aware datetime. Rollback pattern — undo recent commits.
            results: Remove the specific entries returned by a prior
                ``memory.search()`` call. Cannot be combined with other
                filters. Uses entry-level granularity — only the matched
                entries are removed, not the whole commit or context.

        Raises:
            NexusValidationError: If no filter is provided, if ``results``
                is combined with other filters, if ``before`` and ``since``
                are both set, or if a timestamp is not timezone-aware.
            NexusStorageError: If ApertureDB rejects any delete.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            # Remove one specific commit
            commit_id = memory.commit(ctx, info)
            memory.remove(commit_id=commit_id)

            # Remove everything from a context
            memory.remove(context_id=ctx.id)

            # Remove everything from a session
            memory.remove(session_id=ctx.session_id)

            # Remove stale entries (keep last 24 hours)
            from datetime import datetime, timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            memory.remove(before=cutoff, session_id=sid)

            # Search then remove matching entries
            results = memory.search(query="old pricing model")
            memory.remove(results=results)
        """
        if results is not None:
            if any(x is not None for x in (commit_id, context_id, session_id, before, since)):
                raise NexusValidationError(
                    "results= cannot be combined with other filters. "
                    "Pass either results= or one or more of: "
                    "commit_id, context_id, session_id, before, since."
                )
            if not results:
                return
            self._remove_by_results(results)
            return

        if not any(x is not None for x in (commit_id, context_id, session_id, before, since)):
            raise NexusValidationError(
                "At least one filter is required. Pass commit_id=, "
                "context_id=, session_id=, before=, since=, or results=."
            )

        if before is not None and since is not None:
            raise NexusValidationError(
                "before= and since= cannot be combined. "
                "Use before= to discard old entries (keep recent), "
                "or since= to discard recent entries (rollback pattern)."
            )

        if before is not None and not (
            before.tzinfo is not None and before.tzinfo.utcoffset(before) is not None
        ):
            raise NexusValidationError(
                "before= must be a timezone-aware datetime. "
                "Use datetime.now(timezone.utc) or attach a tzinfo."
            )

        if since is not None and not (
            since.tzinfo is not None and since.tzinfo.utcoffset(since) is not None
        ):
            raise NexusValidationError(
                "since= must be a timezone-aware datetime. "
                "Use datetime.now(timezone.utc) or attach a tzinfo."
            )

        if commit_id is not None and (not isinstance(commit_id, str) or not commit_id.strip()):
            raise NexusValidationError(
                "commit_id must be a non-empty string. "
                "Pass the value returned by memory.commit()."
            )
        if context_id is not None and (not isinstance(context_id, str) or not context_id.strip()):
            raise NexusValidationError(
                "context_id must be a non-empty string. "
                "Pass ctx.id or the context_id from a SearchResult."
            )
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise NexusValidationError(
                "session_id must be a non-empty string. "
                "Pass ctx.session_id or the session_id from a SearchResult."
            )

        constraints: dict = {}
        if commit_id:
            constraints["commit_id"] = ["==", commit_id]
        if context_id:
            constraints["context_id"] = ["==", context_id]
        if session_id:
            constraints["session_id"] = ["==", session_id]
        if before is not None:
            constraints["created_at"] = ["<", before.isoformat()]
        if since is not None:
            constraints["created_at"] = [">=", since.isoformat()]

        self._delete_content_by_constraints(constraints)

        # Remove the NexusContext entity only when removing by context_id alone
        # (other filter combinations do partial removes — leave the entity intact)
        if context_id and not any(x is not None for x in (commit_id, session_id, before, since)):
            cmd = [{"DeleteEntity": {
                "with_class": _CLASS_CONTEXT,
                "constraints": {"nexus_ctx_id": ["==", context_id]},
            }}]
            response, _ = self._db.query(cmd)
            _check_response(response, f"remove_context_entity({context_id!r})")

        logger.debug("remove() completed with constraints=%r", constraints)

    def _list_nexus_descriptor_sets(self) -> list[str]:
        """Return names of all nexus DescriptorSets in ApertureDB."""
        cmd = [{"FindDescriptorSet": {
            "results": {"list": ["_name"]},
        }}]
        response, _ = self._db.query(cmd)
        if isinstance(response, dict):
            return []
        body = response[0].get("FindDescriptorSet", {})
        return [
            e["_name"]
            for e in body.get("entities", [])
            if isinstance(e.get("_name"), str)
            and e["_name"].startswith("nexus_")
        ]

    def _delete_content_by_constraints(self, constraints: dict) -> None:
        """Delete all content entities matching the given ApertureDB constraints."""
        for cmd_name in ("DeleteBlob", "DeleteImage", "DeleteVideo"):
            cmd = [{cmd_name: {"constraints": constraints}}]
            response, _ = self._db.query(cmd)
            _check_response(response, f"remove/{cmd_name}")
        for set_name in self._list_nexus_descriptor_sets():
            cmd = [{"DeleteDescriptor": {
                "set": set_name,
                "constraints": constraints,
            }}]
            response, _ = self._db.query(cmd)
            _check_response(response, f"remove/DeleteDescriptor[{set_name}]")

    def _remove_by_results(self, results: list) -> None:
        """Remove specific content entries identified by SearchResult._entry_id."""
        seen: set = set()
        for result in results:
            entry_id = getattr(result, "_entry_id", None)
            if not entry_id or entry_id in seen:
                continue
            seen.add(entry_id)
            self._delete_content_by_constraints({"entry_id": ["==", entry_id]})
        if not seen:
            logger.warning(
                "remove(results=...) found no _entry_id on any result — "
                "nothing removed. Results may have been obtained from an "
                "older index that predates entry_id stamping."
            )

    # ------------------------------------------------------------------
    # retrieve()
    # ------------------------------------------------------------------

    def retrieve(self, context_id: str) -> list[MemoryEntry]:
        """Retrieve all content stored under a context_id.

        Fetches every text, image, video, and blob entry that was
        committed under the given context_id, returning the actual
        content bytes alongside metadata.

        Text entries are decoded to ``str`` in the ``text`` field.
        Image, video, and blob entries are returned as raw ``bytes``
        in the ``data`` field.

        Args:
            context_id: The context ID returned by ``commit()``.

        Returns:
            List of ``MemoryEntry``, one per stored content item.
            Order within each modality matches insertion order.

        Raises:
            NexusValidationError: If ``context_id`` is empty.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            context_id = memory.commit(ctx, info)
            entries = memory.retrieve(context_id)
            for entry in entries:
                if entry.modality == "text":
                    print(entry.text)
                elif entry.modality == "image":
                    with open("out.png", "wb") as f:
                        f.write(entry.data)
        """
        if not isinstance(context_id, str) or not context_id.strip():
            raise NexusValidationError(
                "context_id must be a non-empty string. "
                "Pass the context_id returned by memory.commit()."
            )
        constraints = {"context_id": ["==", context_id]}
        entries: list[MemoryEntry] = []
        entries.extend(self._retrieve_blobs(constraints, context_id))
        entries.extend(self._retrieve_images(constraints, context_id))
        entries.extend(self._retrieve_videos(constraints, context_id))
        logger.debug(
            "Retrieved %d entries for context_id=%r",
            len(entries), context_id,
        )
        return entries

    def _retrieve_blobs(
        self, constraints: dict, context_id: str
    ) -> list[MemoryEntry]:
        """Fetch Blob entries (text and raw blobs) for a context."""
        cmd = [{"FindBlob": {
            "constraints": constraints,
            "blobs": True,
            "results": {"all_properties": True},
        }}]
        try:
            response, blobs = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB blob retrieval failed: {e}. "
                "Run 'adb-nexus validate' to check your connection."
            ) from e
        if isinstance(response, dict):
            return []
        body = response[0].get("FindBlob", {})
        blob_start = body.get("blobs_start", 0)
        entries = []
        for ent in body.get("entities", []):
            idx = blob_start + ent.get("_blob_index", 0)
            raw = blobs[idx] if idx < len(blobs) else None
            doc_type = ent.get("document_type", "")
            created_at = datetime.fromisoformat(
                ent.get("created_at", datetime.utcnow().isoformat())
            )
            props = {
                k: v for k, v in ent.items()
                if not k.startswith("_")
            }
            if doc_type == "text":
                entries.append(MemoryEntry(
                    modality="text",
                    context_id=context_id,
                    session_id=ent.get("session_id", ""),
                    created_at=created_at,
                    text=raw.decode("utf-8") if raw else None,
                    properties=props,
                ))
            else:
                entries.append(MemoryEntry(
                    modality="blob",
                    context_id=context_id,
                    session_id=ent.get("session_id", ""),
                    created_at=created_at,
                    data=raw,
                    document_type=doc_type or None,
                    properties=props,
                ))
        return entries

    def _retrieve_images(
        self, constraints: dict, context_id: str
    ) -> list[MemoryEntry]:
        """Fetch Image entries for a context."""
        cmd = [{"FindImage": {
            "constraints": constraints,
            "blobs": True,
            "results": {"all_properties": True},
        }}]
        try:
            response, blobs = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB image retrieval failed: {e}. "
                "Run 'adb-nexus validate' to check your connection."
            ) from e
        if isinstance(response, dict):
            return []
        body = response[0].get("FindImage", {})
        blob_start = body.get("blobs_start", 0)
        entries = []
        for ent in body.get("entities", []):
            idx = blob_start + ent.get("_blob_index", 0)
            raw = blobs[idx] if idx < len(blobs) else None
            entries.append(MemoryEntry(
                modality="image",
                context_id=context_id,
                session_id=ent.get("session_id", ""),
                created_at=datetime.fromisoformat(
                    ent.get("created_at", datetime.utcnow().isoformat())
                ),
                data=raw,
                properties={
                    k: v for k, v in ent.items()
                    if not k.startswith("_")
                },
            ))
        return entries

    def _retrieve_videos(
        self, constraints: dict, context_id: str
    ) -> list[MemoryEntry]:
        """Fetch Video entries for a context."""
        cmd = [{"FindVideo": {
            "constraints": constraints,
            "blobs": True,
            "results": {"all_properties": True},
        }}]
        try:
            response, blobs = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB video retrieval failed: {e}. "
                "Run 'adb-nexus validate' to check your connection."
            ) from e
        if isinstance(response, dict):
            return []
        body = response[0].get("FindVideo", {})
        blob_start = body.get("blobs_start", 0)
        entries = []
        for ent in body.get("entities", []):
            idx = blob_start + ent.get("_blob_index", 0)
            raw = blobs[idx] if idx < len(blobs) else None
            entries.append(MemoryEntry(
                modality="video",
                context_id=context_id,
                session_id=ent.get("session_id", ""),
                created_at=datetime.fromisoformat(
                    ent.get("created_at", datetime.utcnow().isoformat())
                ),
                data=raw,
                properties={
                    k: v for k, v in ent.items()
                    if not k.startswith("_")
                },
            ))
        return entries

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
        raise NexusConfigError(
            "memory.stats() is not yet implemented. "
            "Full metrics support is planned for a future release. "
            "Install: pip install aperture-nexus[metrics]"
        )

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
            "nexus_ctx_id": ctx.id,
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
            "if_not_found": {"nexus_ctx_id": ["==", ctx.id]},
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
                        "constraints": {"nexus_ctx_id": ["==", ctx.id]},
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

    def _write_context_embedding(
        self, ctx: Context, session_id: str
    ) -> None:
        """Embed the context's purpose into the nexus_context DescriptorSet.

        Called by ``process_and_commit()`` after a successful ``commit()``.
        Silently skips if ``ctx.purpose`` is not set or no text embedding
        model is configured — so callers without a purpose or model never
        see an error from this path.

        Args:
            ctx: The context whose purpose should be embedded.
            session_id: Resolved session ID (from ``_resolve_session_id``).
        """
        if not ctx.purpose:
            return
        model = self._cfg.models.text_embedding
        if not model:
            return

        from aperture_nexus._embeddings import is_clip_model, get_clip_embedder
        if not is_clip_model(model):
            raise NexusConfigError(
                f"Automatic context embedding via model={model!r} is not "
                f"yet available. Use a CLIP model (e.g. 'ViT-B/16') or "
                f"set models.text_embedding in aperture_nexus.json."
            )

        embedder = get_clip_embedder(model)
        try:
            vector = embedder.embed_text([ctx.purpose])[0]
        except (NexusConfigError, NexusValidationError):
            raise
        except Exception as e:
            raise NexusProcessingError(
                f"CLIP embedding of context purpose failed: {e}."
            ) from e

        dset = _dset_name("context", model)
        self._ensure_descriptor_set(
            dset, vector, "context", model,
            self._cfg.processing.descriptor_metric,
            self._cfg.processing.descriptor_engine,
        )

        props: dict = {
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "purpose": ctx.purpose,
            "embedding_model": model,
            "created_at": datetime.utcnow().isoformat(),
        }
        for attr in ("organization", "department"):
            val = getattr(ctx, attr, None) or getattr(ctx.principal, attr, None)
            if val is not None:
                props[attr] = val

        emb_bytes = _embedding_to_bytes(vector)
        cmd = [{"AddDescriptor": {"set": dset, "properties": props}}]
        response, _ = self._db.query(cmd, [emb_bytes])
        _check_response(response, "write_context_embedding")
        logger.debug(
            "Wrote context embedding for context_id=%r purpose=%r",
            ctx.id, ctx.purpose,
        )

    def _descriptor_props(
        self, entry, ctx: Context, session_id: str, modality: str,
        commit_id: str, entry_id: str,
    ) -> dict:
        """Build the properties dict for an AddDescriptor command."""
        props: dict = {
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "embedding_model": entry.embedding_model,
            "modality": modality,
            "created_at": datetime.utcnow().isoformat(),
            "commit_id": commit_id,
            "entry_id": entry_id,
        }
        if entry.metadata:
            props.update(entry.metadata)
        # Store short text inline on the Descriptor for search result hydration
        if entry.text and len(entry.text) <= 2000:
            props["text"] = entry.text
        return props

    def _write_entry(self, entry, ctx: Context, session_id: str, commit_id: str) -> None:
        """Write one InformationEntry to ApertureDB.

        Each content object (Blob, Image, Video) and its Descriptor are
        written in a single atomic transaction using ``_ref`` /
        ``AddConnection``. This ensures the Descriptor is always
        connected to its source entity — without the connection the
        Descriptor is an orphan and cannot be traced back to the original
        content.
        """
        entry_id = str(uuid.uuid4())
        common_props = {
            "context_id": ctx.id,
            "session_id": session_id,
            "user_id": ctx.principal.user_id,
            "created_at": datetime.utcnow().isoformat(),
            "commit_id": commit_id,
            "entry_id": entry_id,
        }
        # Include session_name so entries can be filtered by it (session_name
        # is also stored on NexusSession, but the filter path queries content
        # entities directly, so the property must live here too).
        if ctx.session_name:
            common_props["session_name"] = ctx.session_name
        # Include principal org/dept so content entities can be filtered by them
        if ctx.principal.organization:
            common_props["organization"] = ctx.principal.organization
        if ctx.principal.department:
            common_props["department"] = ctx.principal.department
        if entry.metadata:
            common_props.update(entry.metadata)

        has_emb = entry.embedding is not None and entry.embedding_model

        # ---- Text blob -------------------------------------------------------
        if entry.text is not None:
            text_bytes = entry.text.encode("utf-8")
            props = dict(common_props, document_type="text")
            if len(entry.text) <= 2000:
                props["text_preview"] = entry.text[:500]

            if has_emb:
                dset = _dset_name("text", entry.embedding_model)
                desc_props = self._descriptor_props(entry, ctx, session_id, "text", commit_id, entry_id)
                emb_bytes = _embedding_to_bytes(entry.embedding)
                cmd = [
                    {"AddBlob": {"properties": props, "_ref": 1}},
                    {"AddDescriptor": {"set": dset, "properties": desc_props, "_ref": 2}},
                    {"AddConnection": {"class": _CONN_DESCRIPTOR, "src": 2, "dst": 1}},
                ]
                response, _ = self._db.query(cmd, [text_bytes, emb_bytes])
                _check_response(response, "write_text_entry+descriptor")
            else:
                cmd = [{"AddBlob": {"properties": props}}]
                response, _ = self._db.query(cmd, [text_bytes])
                _check_response(response, "write_text_entry")

        # ---- Image -----------------------------------------------------------
        if entry.image is not None:
            image_bytes = _to_image_bytes(entry.image)
            props = dict(common_props)

            if has_emb:
                dset = _dset_name("image", entry.embedding_model)
                desc_props = self._descriptor_props(entry, ctx, session_id, "image", commit_id, entry_id)
                emb_bytes = _embedding_to_bytes(entry.embedding)
                cmd = [
                    {"AddImage": {"properties": props, "_ref": 1}},
                    {"AddDescriptor": {"set": dset, "properties": desc_props, "_ref": 2}},
                    {"AddConnection": {"class": _CONN_DESCRIPTOR, "src": 2, "dst": 1}},
                ]
                response, _ = self._db.query(cmd, [image_bytes, emb_bytes])
                _check_response(response, "write_image_entry+descriptor")
            else:
                cmd = [{"AddImage": {"properties": props}}]
                response, _ = self._db.query(cmd, [image_bytes])
                _check_response(response, "write_image_entry")

        # ---- Video -----------------------------------------------------------
        if entry.video is not None:
            video_bytes = _to_video_bytes(entry.video)
            props = dict(common_props)
            # Video clip embeddings (one descriptor per segment) are handled
            # below; a single-vector embedding on a video entry is unusual
            # but supported.
            if has_emb and entry.video_clip_embeddings is None:
                dset = _dset_name("video", entry.embedding_model)
                desc_props = self._descriptor_props(entry, ctx, session_id, "video", commit_id, entry_id)
                emb_bytes = _embedding_to_bytes(entry.embedding)
                cmd = [
                    {"AddVideo": {"properties": props, "_ref": 1}},
                    {"AddDescriptor": {"set": dset, "properties": desc_props, "_ref": 2}},
                    {"AddConnection": {"class": _CONN_DESCRIPTOR, "src": 2, "dst": 1}},
                ]
                response, _ = self._db.query(cmd, [video_bytes, emb_bytes])
                _check_response(response, "write_video_entry+descriptor")
            else:
                cmd = [{"AddVideo": {"properties": props}}]
                response, _ = self._db.query(cmd, [video_bytes])
                _check_response(response, "write_video_entry")

        # ---- Raw blob --------------------------------------------------------
        if entry.blob is not None:
            blob_bytes = _to_blob_bytes(entry.blob)
            props = dict(common_props, document_type=entry.document_type or "")

            if has_emb:
                dset = _dset_name("text", entry.embedding_model)
                desc_props = self._descriptor_props(entry, ctx, session_id, "text", commit_id, entry_id)
                emb_bytes = _embedding_to_bytes(entry.embedding)
                cmd = [
                    {"AddBlob": {"properties": props, "_ref": 1}},
                    {"AddDescriptor": {"set": dset, "properties": desc_props, "_ref": 2}},
                    {"AddConnection": {"class": _CONN_DESCRIPTOR, "src": 2, "dst": 1}},
                ]
                response, _ = self._db.query(cmd, [blob_bytes, emb_bytes])
                _check_response(response, "write_blob_entry+descriptor")
            else:
                cmd = [{"AddBlob": {"properties": props}}]
                response, _ = self._db.query(cmd, [blob_bytes])
                _check_response(response, "write_blob_entry")

        # ---- Per-clip video descriptors --------------------------------------
        if entry.video_clip_embeddings is not None and entry.embedding_model:
            self._write_video_clip_descriptors(entry, ctx, session_id, commit_id, entry_id)

    def _write_video_clip_descriptors(
        self, entry, ctx: Context, session_id: str, commit_id: str, entry_id: str,
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
                "commit_id": commit_id,
                "entry_id": entry_id,
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
        _check_response(response, f"find DescriptorSet {set_name!r}")
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

            # Image takes priority over text so that combined text+image
            # entries are indexed in the image DescriptorSet rather than
            # silently receiving only a text descriptor.
            if entry.image is not None:
                modality = "image"
                model = self._cfg.models.image_embedding
            elif entry.video is not None:
                modality = "video"
                model = self._cfg.models.video_embedding
            elif entry.text is not None:
                modality = "text"
                model = self._cfg.models.text_embedding
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
                    f"CLIP text embedding for search failed: {e}. "
                    "Check that the model is available and run "
                    "'adb-nexus validate' to check your connection."
                ) from e
            # Use the caller's modality if provided — enables cross-modal
            # search (text query → image or video DescriptorSet).
            return vector, modality or "text"

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
                f"ApertureDB search query failed: {e}. "
                "Run 'adb-nexus validate' to check your connection."
            ) from e

        # ApertureDB may return a bare dict (not a list) when the descriptor
        # set does not exist or the query is malformed.  Treat it as empty.
        if not isinstance(response, list) or not response:
            return []
        body = response[0].get("FindDescriptor", {})
        if body.get("status", 0) != 0:
            return []
        # ApertureDB returns descriptors under "entities"; distance is
        # embedded as "_distance" on each entity (not a separate list).
        descriptors = body.get("entities", [])

        results = []
        for desc in descriptors:
            score = float(desc.get("_distance", 0.0))
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
                        "entry_id", "commit_id",
                    )
                },
                _entry_id=desc.get("entry_id"),
            ))
        return results

    def _search_by_metadata(
        self, filters: dict, k: int
    ) -> list[SearchResult]:
        """Return content entities (Blob, Image, Video) matching metadata filters.

        Searches all three content entity types in a single multi-command
        query and combines the results. Blob results are returned with
        modality ``"text"`` (for text blobs) or ``"blob"`` (for raw blobs).
        """
        constraints = self._build_constraints(filters)
        results_limit = {"all_properties": True, "limit": k}

        def _body():
            b: dict = {"results": results_limit}
            if constraints:
                b["constraints"] = constraints
            return b

        cmd = [
            {"FindBlob": _body()},
            {"FindImage": _body()},
            {"FindVideo": _body()},
        ]
        try:
            response, _ = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB metadata search failed: {e}. "
                "Run 'adb-nexus validate' to check your connection."
            ) from e

        if isinstance(response, dict):
            return []

        results = []
        entity_types = [
            ("FindBlob", None),    # modality derived from document_type
            ("FindImage", "image"),
            ("FindVideo", "video"),
        ]
        for resp_item, (cmd_name, fixed_modality) in zip(response, entity_types):
            body = resp_item.get(cmd_name, {})
            for ent in body.get("entities", []):
                if fixed_modality is None:
                    doc_type = ent.get("document_type", "")
                    modality = "text" if doc_type == "text" else "blob"
                else:
                    modality = fixed_modality
                results.append(SearchResult(
                    score=1.0,
                    modality=modality,
                    session_id=ent.get("session_id", ""),
                    context_id=ent.get("context_id", ""),
                    user_id=ent.get("user_id"),
                    created_at=datetime.fromisoformat(
                        ent.get("created_at", datetime.utcnow().isoformat())
                    ),
                    text=ent.get("text_preview") or ent.get("text"),
                    metadata={
                        k: v for k, v in ent.items()
                        if k not in (
                            "context_id", "session_id", "user_id", "created_at",
                            "text_preview", "text", "document_type",
                            "entry_id", "commit_id",
                        )
                    },
                    _entry_id=ent.get("entry_id"),
                ))
        return results[:k]

    @staticmethod
    def _build_constraints(filters: dict) -> dict:
        """Convert a user-supplied filters dict to ApertureDB constraints.

        Raises NexusValidationError for any key not in the allowed set so
        that typos and unsupported keys produce a loud error rather than
        silent empty results.
        """
        allowed = {
            "session_id", "session_name", "context_id", "commit_id",
            "user_id", "organization", "department", "purpose",
        }
        unknown = set(filters) - allowed
        if unknown:
            raise NexusValidationError(
                f"Unknown filter key(s): {sorted(unknown)}. "
                f"Supported keys: {sorted(allowed)}."
            )
        return {key: ["==", value] for key, value in filters.items()}
