---
title: API Reference
description: Full reference for NexusAdmin, Memory, Context, Information, MemoryTask, and the exception hierarchy
sidebar_position: 2
---

# API Reference

Full reference for all public classes and methods. For the conceptual overview, see [Concepts](concepts.md).

---

## `generate_session_id()`

```python
from aperture_nexus import generate_session_id

generate_session_id(prefix: str | None = None) -> str
```

Generate a unique session ID. Use this when coordinating a shared session across multiple participants before any of them construct a `Context`.

```python
sid = generate_session_id()
# or with a prefix for readability
sid = generate_session_id(prefix="support")  # e.g. "support-a3f7c..."

ctx_customer = Context(principal=customer_principal, session_id=sid, ...)
ctx_agent    = Context(principal=agent_principal,    session_id=sid, ...)
```

---

## `NexusAdmin`

Identity authority. Creates and manages app-level Principals. Requires admin
ApertureDB credentials.

```python
from aperture_nexus import NexusAdmin
```

### Construction

```python
NexusAdmin(
    config: str | None = None,
    db_client: Connector | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str \| None` | Path to `aperture_nexus.json`. Discovered automatically if `None`. |
| `db_client` | `Connector \| None` | Inject an existing admin `Connector`. If `None`, a connector is created from environment variables or the active `adb` configuration. |

If `organization` or `department` is not passed to `admin.create_principal()`,
the defaults (`nexus_default_org` and `nexus_default_dept`) are used. These names
are configurable via `admin.default_organization` and `admin.default_department`
in `aperture_nexus.json`.

**Raises:** `NexusConnectionError` if admin credentials cannot be resolved;
`NexusConfigError` if the config file is invalid.

```python
admin = NexusAdmin()
admin = NexusAdmin(config="/path/to/aperture_nexus.json")
admin = NexusAdmin(db_client=existing_admin_connector)
```

---

### `admin.create_principal()`

```python
admin.create_principal(
    user_id: str,
    user_name: str | None = None,
    department: str | None = None,
    organization: str | None = None,
) -> str
```

Register a new app-level Principal in ApertureDB. Returns a generated API key
that must be delivered to the user out-of-band. The key is stored hashed —
aperture-nexus cannot recover it after this call.

**Returns:** `str` — the plaintext API key (show once, store securely)

```python
api_key = admin.create_principal(
    user_id="alice",
    user_name="Alice Chen",
    department="support",
    organization="AcmeCorp",
)
```

---

### `admin.delete_principal()`

```python
admin.delete_principal(user_id: str) -> None
```

Remove a Principal from ApertureDB. Existing memories written by this user are
retained; the user can no longer authenticate.

---

## `Memory`

Storage and retrieval engine. The only component that commits and searches
memories in ApertureDB. Operates on behalf of an authenticated `Principal`
via the `Context` passed to each operation.

```python
from aperture_nexus import Memory
```

### Construction

```python
Memory(
    config: str | None = None,
    db_client: Connector | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str \| None` | Path to `aperture_nexus.json`. If `None`, config is discovered automatically (see [Configuration](configuration.md#config-discovery)). |
| `db_client` | `Connector \| None` | Inject an existing ApertureDB `Connector`. Useful for testing or connection reuse. If `None`, a connector is created from environment variables or the active `adb` configuration. |

The ApertureDB connection is not established at construction time — it is
created on the first operation.

**Raises:** `NexusConnectionError` if credentials cannot be resolved;
`NexusConfigError` if the config file is invalid.

```python
# Default — credentials from environment or adb config
memory = Memory()

# Explicit config file
memory = Memory(config="/path/to/aperture_nexus.json")

# Inject connector (testing, connection reuse)
memory = Memory(db_client=existing_connector)
```

---

### `memory.authenticate()`

```python
memory.authenticate(user_id: str, api_key: str) -> Principal
```

Validate credentials and return an authenticated `Principal`. Looks up the `NexusUser` entity for `user_id` and compares the SHA-256 hash of `api_key` against the stored hash. Requires regular (non-admin) ApertureDB credentials.

In normal use, `api_key` comes from the `NEXUS_API_KEY` environment variable written to `.env` by `adb-nexus init`.

**Returns:** `Principal` — pass to `Context` to identify who is acting.

**Raises:** `NexusPermissionError` if credentials are invalid or the user does not exist; `NexusConnectionError` if ApertureDB is unreachable.

```python
import os
principal = memory.authenticate(
    user_id="alice",
    api_key=os.environ["NEXUS_API_KEY"],
)
ctx = Context(principal=principal, session_name="support-001")
```

---

### `memory.commit()`

```python
memory.commit(ctx: Context, info: Information) -> str
```

Store `Information` in ApertureDB as-is — no model calls, no embeddings. Fast. Returns a `commit_id` — a new UUID identifying this specific commit call.

Use `commit()` when you need speed and plan to search by metadata (session, organization, time range) rather than semantic similarity.

**Returns:** `str` — the `commit_id`. Pass it to `memory.remove(commit_id=...)` to remove everything written in this call.
**Raises:** `NexusConnectionError`, `NexusStorageError`, `NexusPermissionError`

```python
commit_id = memory.commit(ctx, info)
```

---

### `memory.process_and_commit()`

```python
memory.process_and_commit(ctx: Context, info: Information) -> str
```

Generate embeddings and summaries for all items in `info`, then store everything in ApertureDB. Slower than `commit()` but enables semantic (vector) search.

Requires at least one model configured in `aperture_nexus.json` or supplied via `info.log(embedding_model=...)`.

**Blob-only entries** (e.g. `info.log(blob=pdf_bytes, document_type="pdf")` with no `text=` or `embedding=`) raise `NexusConfigError` because Nexus does not extract text from documents in v1. Options: extract upstream and pass with `text=`, provide a pre-computed embedding, or use `memory.commit()` for opaque storage.

**Returns:** `str` — the `commit_id`. See `commit()`.
**Raises:** `NexusConfigError` if no model is configured or if the entry is blob-only; `NexusProcessingError` if model calls fail; `NexusConnectionError`, `NexusStorageError`, `NexusPermissionError`

```python
commit_id = memory.process_and_commit(ctx, info)
```

---

### `memory.async_process_and_commit()`

```python
await memory.async_process_and_commit(ctx: Context, info: Information) -> MemoryTask
```

Async variant of `process_and_commit()`. Returns a `MemoryTask` immediately without blocking. The task runs in the background and updates its status in ApertureDB.

**Returns:** `MemoryTask` with `status="pending"`

```python
task = await memory.async_process_and_commit(ctx, info)
print(task.status)   # "pending"

await task.wait()

if task.status == "complete":
    print(f"commit_id: {task.commit_id}")
else:
    print(f"Failed: {task.error_message}")
    await task.retry()
```

---

### `memory.search()`

```python
memory.search(
    query: str | Path | PIL.Image | np.ndarray | bytes | None = None,
    modality: str | None = None,
    filters: dict | None = None,
    k: int = 10,
    embedding_model: str | None = None,
    min_score: float | None = None,
) -> list[SearchResult]
```

Search stored memories by semantic similarity, metadata, or both.
Permissions are enforced automatically from the Memory's principal.

Search is **per-modality**: each query type searches its own descriptor
set. There is no cross-modal search (e.g. image query finding text
results) in v1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | see below | The search query. Type determines modality. `None` returns results by metadata only. |
| `modality` | `str \| None` | Required when `query` is `np.ndarray` (ambiguous). One of `"text"`, `"image"`, `"video"`. |
| `filters` | `dict \| None` | Metadata filters. Supported keys: `session_id`, `session_name`, `context_id`, `user_id`, `organization`, `department`, `purpose`. Unknown keys raise `NexusValidationError`. |
| `k` | `int` | Maximum results. Default: `10`. |
| `embedding_model` | `str \| None` | Override the configured embedding model for this query. Must match the model used at index time. |
| `min_score` | `float \| None` | Minimum similarity score threshold. Results below this are excluded. |

**Query type → modality mapping:**

| `query` type | Modality | Embedding call |
|---|---|---|
| `str` | text | yes — uses `models.text_embedding` |
| `Path` / URL `str` / `PIL.Image` | image | yes — uses `models.image_embedding` |
| video `Path` / `bytes` | video | yes — clip embedded, uses `models.video_embedding` |
| `np.ndarray` | requires `modality=` | no — used directly |
| `None` | — | no — metadata filter only |

**Returns:** `list[SearchResult]`
**Raises:** `NexusPermissionError`, `NexusConnectionError`,
`NexusConfigError` (missing model, mismatched embedding space)

```python
# Text query → searches text descriptor set
results = memory.search(query="missing order last week")

# Image query → searches image descriptor set
results = memory.search(query="photo.jpg")
results = memory.search(query=pil_image)

# Pre-computed embedding → modality required
results = memory.search(query=my_vector, modality="image")

# Metadata filter only — no embedding needed
results = memory.search(filters={"user_id": "alice"}, k=50)

# Combined: semantic + metadata filter
results = memory.search(
    query="order inquiry",
    filters={"session_id": sid, "organization": "AcmeCorp"},
)

for r in results:
    print(r.score, r.modality, r.text or r.image)
```

---

### `SearchResult`

Returned by `memory.search()`.

```python
@dataclass
class SearchResult:
    score: float        # similarity (higher = more similar)
    modality: str       # "text" | "image" | "video" | "blob"
    session_id: str
    context_id: str
    user_id: str | None
    created_at: datetime

    # Text content — set when modality is "text" and text is short
    text: str | None = None

    # Video clip boundaries (modality "video" only)
    start_frame: int | None = None
    stop_frame: int | None = None

    metadata: dict = field(default_factory=dict)
```

To retrieve the raw bytes for image, video, or blob results, fetch
the entry via the `context_id` or `session_id` from ApertureDB
directly, or use `memory.search()` with `modality=` and use the
`metadata` field to identify the stored object reference.

---

### `memory.search_contexts()`

```python
memory.search_contexts(
    query: str,
    filters: dict | None = None,
    k: int = 10,
    embedding_model: str | None = None,
) -> list[ContextResult]
```

Search committed **contexts** by semantic similarity of their `purpose` field. Complements `memory.search()`, which searches stored content (blobs, images, etc.) — `search_contexts()` searches the context graph nodes themselves.

Only contexts committed via `process_and_commit()` with a `purpose` set are indexed. `commit()` does not embed context nodes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Text query matched against context purposes. |
| `filters` | `dict \| None` | Metadata constraints. Supported keys: `session_id`, `user_id`, `organization`, `department`. |
| `k` | `int` | Maximum results. Default: `10`. |
| `embedding_model` | `str \| None` | Override the configured text model. Must match the model used at index time. |

**Returns:** `list[ContextResult]`, ordered by descending similarity score.

**Raises:** `NexusConfigError` (no text model configured), `NexusProcessingError` (embedding call failed), `NexusConnectionError`.

```python
results = memory.search_contexts("customer order inquiry")
for r in results:
    print(r.purpose, r.session_id, r.score)

# Narrow to a specific session
results = memory.search_contexts(
    "warehouse inventory shortage",
    filters={"session_id": sid},
    embedding_model="ViT-B/16",
)
```

---

### `ContextResult`

Returned by `memory.search_contexts()`.

```python
@dataclass
class ContextResult:
    score: float            # similarity (higher = more similar)
    context_id: str
    session_id: str
    user_id: str | None
    purpose: str | None     # the purpose set on the Context
    created_at: datetime

    organization: str | None = None
    department: str | None = None
```

---

### `memory.connect()`

```python
memory.connect(
    source: Context | str,
    target: Context | str,
    relationship: str,
) -> None
```

Create a named relationship between two contexts. Both `source` and `target` must be `NexusContext` entities (pass a `Context` or its `context_id` string). Writes a `nexus_link` connection with `type=relationship`. Memory-to-memory (commit-to-commit) linking is not part of v1; if you need to associate specific memories, link the contexts that authored them.

```python
memory.connect(source=ctx_q1, target=ctx_q2, relationship="follows")
memory.connect(source=ctx_id_1, target=ctx_id_2, relationship="related_to")
```

---

### `memory.remove()`

```python
memory.remove(
    *,
    commit_id: str | None = None,
    context_id: str | None = None,
    session_id: str | None = None,
    before: datetime | None = None,
    since: datetime | None = None,
    results: list[SearchResult] | None = None,
) -> None
```

Remove committed content from ApertureDB. At least one filter is required. Filters AND together — e.g. `before=ts, session_id=sid` removes only old entries within that session. `results=` is exclusive and cannot be combined with other filters.

| Filter | Removes |
|--------|---------|
| `commit_id=` | All content written in one `commit()` call, plus the `NexusCommit` entity for that commit |
| `context_id=` | All content from a context; also cascades to `NexusCommit` and `NexusContext` entities when used alone |
| `session_id=` | All content from a session; also cascades to `NexusCommit`, `NexusContext`, and `NexusSession` entities when used alone |
| `before=` | Entries whose `created_at` is strictly before this UTC-aware datetime (keep recent, discard old) |
| `since=` | Entries whose `created_at` is at or after this datetime (rollback pattern) |
| `results=` | Specific entries returned by a prior `memory.search()` call (entry-level granularity) |

**Raises:** `NexusValidationError` if no filter is provided, if `results=` is combined with other filters, if `before=` and `since=` are both set, or if a timestamp is not timezone-aware. `NexusStorageError` if ApertureDB rejects any delete.

```python
from datetime import datetime, timedelta, timezone

# Remove one specific commit
commit_id = memory.commit(ctx, info)
memory.remove(commit_id=commit_id)

# Remove everything from a context
memory.remove(context_id=ctx.id)

# Remove everything from a session
memory.remove(session_id=ctx.session_id)

# Prune stale entries (keep last 24 hours)
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
memory.remove(before=cutoff, session_id=sid)

# Search then remove matching entries
results = memory.search(query="old pricing model")
memory.remove(results=results)
```

---

### `memory.pending_commits()` / `memory.failed_commits()`

```python
memory.pending_commits() -> list[MemoryTask]
memory.failed_commits() -> list[MemoryTask]
```

List async tasks that are still in flight or have failed.

```python
for task in memory.failed_commits():
    print(task.error_message)
    await task.retry()
```

---

### `memory.stats()`

```python
memory.stats(scope: str = "session") -> dict
```

Return usage statistics. Requires `pip install aperture-nexus[metrics]`.

| `scope` | Description |
|---------|-------------|
| `"session"` | Stats for the current session |
| `"global"` | Stats across all sessions |

---

## `Context`

Captures who is doing what, in which session, and why. Data object only — does not write to ApertureDB.

```python
from aperture_nexus import Context
```

### Construction

```python
Context(
    principal: Principal,
    session_id: str | None = None,
    session_name: str | None = None,
    purpose: str | None = None,
    organization: str | None = None,
    department: str | None = None,
    priority: int = 0,
    restrictions: dict | None = None,
)
```

Either `session_id` or `session_name` is required. If `session_name` is provided and no session with that name exists, a new session is created on first `memory.commit()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `principal` | `Principal` | The authenticated user or agent. Required. |
| `session_id` | `str \| None` | ID of an existing session. Use `generate_session_id()` (from `aperture_nexus`) for multi-participant sessions. |
| `session_name` | `str \| None` | Human-readable session name. Must be unique within a principal's scope. |
| `purpose` | `str \| None` | The task or intent behind this interaction — a short phrase describing what is being done (e.g. `"debug failing export"`, `"Q3 budget review"`, `"customer support ticket #4821"`). Stored as metadata; filterable at search time via `filters={"purpose": "..."}`. |
| `organization` | `str \| None` | Group scope for permission and search filtering. |
| `department` | `str \| None` | Department this context belongs to. Inherited from `principal.department` if not set explicitly. |
| `priority` | `int` | Relative priority hint. Higher values are processed first in batch operations. |
| `restrictions` | `dict \| None` | `{"local": [...], "global": [...]}` — access constraints applied during search. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `ctx.id` | `str` | Auto-generated context ID. |
| `ctx.session_id` | `str` | The session this context belongs to. |
| `ctx.principal` | `Principal` | The authenticated principal. |

```python
# Single participant
ctx = Context(
    principal=principal,
    session_name="support-2024-001",
    purpose="Customer reporting missing order",
    organization="AcmeCorp",
)

# One of many participants in a shared session
from aperture_nexus import generate_session_id
sid = generate_session_id()
ctx_customer = Context(principal=customer_principal, session_id=sid, purpose="Order inquiry")
ctx_ai       = Context(principal=ai_principal,       session_id=sid, purpose="First response")
```

---

## `Information`

Local buffer for multimodal inputs. Nothing is written to ApertureDB until `memory.commit()` is called.

```python
from aperture_nexus import Information
```

### Construction

```python
Information(context_id: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `context_id` | `str` | ID of the `Context` this information belongs to. Use `ctx.id`. |

### `info.log()`

```python
info.log(
    text: str | None = None,
    image: str | bytes | PIL.Image | np.ndarray | None = None,
    video: str | bytes | None = None,
    blob: str | bytes | None = None,
    document_type: str | None = None,
    embedding: np.ndarray | None = None,
    embedding_model: str | None = None,
    metadata: dict | None = None,
    tag: str | None = None,
) -> InformationEntry
```

Add one entry to the buffer and return it. Multiple modalities can be combined in a single call (e.g. text + blob for "see attached PDF").

Validation happens eagerly at `log()` time — bad inputs raise `NexusValidationError` immediately. For file path inputs, file existence and read permissions are checked at `log()` time; content validity (format, decoding) is checked at `commit()` time.

The returned `InformationEntry` can be passed to `remove()` to discard the entry before it is committed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str \| None` | Plain text. Long text is chunked automatically at commit time. |
| `image` | `str \| bytes \| PIL.Image \| np.ndarray \| None` | File path, URL, raw bytes, PIL Image, or numpy array (HWC uint8/float32). File existence and read permission checked at `log()` time; content decoded at `commit()` time. |
| `video` | `str \| bytes \| None` | File path, URL, or raw bytes. File existence and read permission checked at `log()` time; content stored at `commit()` time. |
| `blob` | `str \| bytes \| None` | File path, URL, or raw bytes for any binary format. Requires `document_type`. File existence and read permission checked at `log()` time; content read at `commit()` time. |
| `document_type` | `str \| None` | File extension hint for blobs: `"pdf"`, `"mp3"`, `"docx"`, `"csv"`, etc. Required when `blob` is provided. |
| `embedding` | `np.ndarray \| None` | Pre-computed embedding vector. Skips model call at commit time. Requires `embedding_model`. |
| `embedding_model` | `str \| None` | Name of the model that produced the embedding. Required when `embedding` is provided. |
| `metadata` | `dict \| None` | Arbitrary key-value properties stored alongside the entry. Keys must be `str`; values must be `str`, `int`, `float`, or `bool`. Reserved keys (`context_id`, `session_id`, etc.) are rejected. |
| `tag` | `str \| None` | Optional label. Pass the same tag to several `log()` calls and then call `remove_tagged(tag)` to discard them all at once. |

**Returns:** `InformationEntry` — holds a reference to the buffered entry. Pass it to `remove()` to discard it.

**Storage semantics:**

- File paths and URLs are stored as references in the local buffer — content is read from disk or network only when `memory.commit()` is called, not at `log()` time.
- Raw `bytes` are held in memory until commit.

**Raises:** `NexusValidationError` if input is invalid (missing file, permission denied, wrong numpy shape, missing `document_type` for blob, reserved metadata key, etc.)

```python
info = Information(context_id=ctx.id)

# Text
info.log(text="Customer says order #4821 never arrived")

# Images — path, URL, PIL Image, or numpy array
info.log(image="screenshot.png")
info.log(image="https://example.com/photo.jpg")
info.log(image=pil_image)
info.log(image=numpy_array)

# Video — file path, URL, or bytes
info.log(video="recording.mp4")
info.log(video="https://example.com/clip.mp4")

# Blobs — file path, URL, or bytes; document_type is required
info.log(blob="contract.pdf", document_type="pdf")
info.log(blob="https://example.com/report.pdf", document_type="pdf")
info.log(blob=audio_bytes, document_type="mp3")

# Metadata alongside any entry
info.log(
    text="Order #4821 missing",
    metadata={"ticket_id": "T-99", "priority": 1},
)

# Pre-computed embedding — skips model call at commit time
info.log(
    image=img,
    embedding=my_vector,
    embedding_model="clip-vit-base-patch32",
)

# Combined — one log entry with multiple modalities
info.log(text="See attached invoice", blob="invoice.pdf", document_type="pdf")
```

### `info.remove()`

```python
info.remove(entry: InformationEntry) -> bool
```

Remove a specific pending entry from the buffer by identity. Pass back the `InformationEntry` returned by `log()`. Uses `is` comparison — not equality — so two entries with identical content are treated as distinct.

| Parameter | Type | Description |
|-----------|------|-------------|
| `entry` | `InformationEntry` | The object returned by `log()`. |

**Returns:** `True` if the entry was found and removed; `False` if it was not in the buffer (already committed or already removed).

**Raises:** `NexusValidationError` if `entry` is not an `InformationEntry` instance.

```python
entry = info.log(text="preliminary draft")
info.log(text="final version")
info.remove(entry)          # discard the draft; only "final version" commits
memory.commit(ctx, info)
```

### `info.remove_tagged()`

```python
info.remove_tagged(tag: str) -> int
```

Remove all pending entries with the given tag. Useful for cancelling a logical group of entries (e.g. all entries for a cancelled order) atomically before commit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tag` | `str` | The tag value to match. Must be non-empty. |

**Returns:** Number of entries removed (0 if none matched).

**Raises:** `NexusValidationError` if `tag` is not a non-empty string.

```python
info.log(text="Order #4412 placed", tag="order-4412")
info.log(blob=receipt, document_type="pdf", tag="order-4412")
if order_cancelled:
    info.remove_tagged("order-4412")   # both entries removed atomically
```

### `info.remove_before()`

```python
info.remove_before(timestamp: datetime) -> int
```

Remove all pending entries logged before `timestamp`. Discards older staged entries while keeping more recent ones. For the rollback pattern (discard entries since a checkpoint), use `remove_since()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `timestamp` | `datetime` | UTC-aware `datetime`. Entries logged strictly before this value are removed. |

**Returns:** Number of entries removed (0 if none matched).

**Raises:** `NexusValidationError` if `timestamp` is not a timezone-aware `datetime`.

```python
from datetime import datetime, timedelta, timezone

cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
info.remove_before(cutoff)   # discard stale entries, keep recent ones
```

### `info.remove_since()`

```python
info.remove_since(checkpoint: datetime) -> int
```

Remove all pending entries logged at or after `checkpoint`. Use this as a rollback: capture a checkpoint before a block of `log()` calls, then call `remove_since(checkpoint)` to undo all logging since that point.

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint` | `datetime` | UTC-aware `datetime`. Entries logged at or after this value are removed. |

**Returns:** Number of entries removed (0 if none matched).

**Raises:** `NexusValidationError` if `checkpoint` is not a timezone-aware `datetime`.

```python
from datetime import datetime, timezone

checkpoint = datetime.now(timezone.utc)
info.log(text="attempt A")
info.log(image="draft.png")
# … something went wrong …
info.remove_since(checkpoint)   # undo everything since the checkpoint
```

### `info.remove_all()`

```python
info.remove_all() -> None
```

Remove all pending entries and connections from the buffer. Nothing is written to or deleted from ApertureDB — only the local buffer is affected.

Use this to abandon a work-in-progress batch before starting over, for example after an upstream error.

```python
info.log(text="wrong context — discard all of this")
info.remove_all()
info.log(text="fresh start")
memory.commit(ctx, info)   # only "fresh start" is stored
```

---

## `MemoryTask`

Returned by `memory.async_process_and_commit()`. Tracks the status of an
in-flight background commit in memory on the `Memory` instance.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `task.status` | `str` | `"pending"` \| `"processing"` \| `"complete"` \| `"failed"` |
| `task.commit_id` | `str \| None` | The commit ID returned by the completed `commit()` call. Available when `status == "complete"` |
| `task.completed_at` | `datetime \| None` | Available when `status == "complete"` |
| `task.error` | `Exception \| None` | Available when `status == "failed"` |
| `task.error_message` | `str \| None` | Human-readable failure reason. Available when `status == "failed"` |
| `task.failed_at` | `datetime \| None` | Available when `status == "failed"` |

### Methods

```python
task.is_ready() -> bool
await task.wait() -> None    # async block until complete or failed
await task.retry() -> None   # resubmit a failed task
```

```python
task = await memory.async_process_and_commit(ctx, info)
await task.wait()

if task.status == "complete":
    print(f"commit_id: {task.commit_id}")
elif task.status == "failed":
    print(f"failed: {task.error_message}")
    await task.retry()
```

---

## `Principal`

Returned by `memory.authenticate()`. Passed to `Context` to identify who is
performing an action. Do not construct directly.

```python
principal = memory.authenticate(user_id="alice", api_key="...")

# Properties
principal.user_id      # "alice"
principal.user_name    # display name, if set at create_principal() time
principal.department   # department this principal belongs to
principal.organization # organization this principal belongs to
```

---

## Exceptions

All exceptions are subclasses of `NexusError`. Import individually or catch the base:

```python
from aperture_nexus.exceptions import (
    NexusError,            # base — catch all aperture-nexus errors
    NexusConfigError,      # misconfiguration, missing models, missing optional deps
    NexusValidationError,  # bad input at log() time
    NexusConnectionError,  # ApertureDB unreachable or credentials rejected
    NexusPermissionError,  # principal lacks permission
    NexusProcessingError,  # model call failed
    NexusStorageError,     # ApertureDB rejected the write
)
```

| Exception | Raised when |
|-----------|-------------|
| `NexusConfigError` | Config file is invalid; required model not configured; optional dep missing; UI exposed without `api_key` |
| `NexusValidationError` | Bad input passed to `info.log()`: missing file, wrong numpy shape, missing `document_type`, unreachable URL |
| `NexusConnectionError` | ApertureDB unreachable; credentials invalid or expired; network error |
| `NexusPermissionError` | Principal lacks permission for the operation; context restrictions violated |
| `NexusProcessingError` | LLM/VLM model call failed or timed out; unexpected model output |
| `NexusStorageError` | ApertureDB returned an error status for a write; schema constraint violated |

Every exception chains the original cause (`raise NexusXxxError("...") from e`), so the full traceback is always available.

```python
try:
    memory.process_and_commit(ctx, info)
except NexusConfigError as e:
    # No model configured — run 'adb-nexus init' to add one
    print(e)
except NexusConnectionError:
    # ApertureDB unreachable — run 'adb-nexus validate'
    pass
except NexusError as e:
    # Catch-all for any aperture-nexus error
    print(f"aperture-nexus error: {e}")
```
