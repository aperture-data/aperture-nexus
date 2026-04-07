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

Identity authority. Creates and manages department-level ApertureDB users and
app-level Principals. Requires admin ApertureDB credentials.

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

On first use, `NexusAdmin` creates the default organization (`nexus_default_org`)
and department (`nexus_default_dept`) entities in ApertureDB if they do not exist.
These names are configurable via `admin.default_organization` and
`admin.default_department` in `aperture_nexus.json`.

**Raises:** `NexusConnectionError` if admin credentials cannot be resolved;
`NexusConfigError` if the config file is invalid.

```python
admin = NexusAdmin()
admin = NexusAdmin(config="/path/to/aperture_nexus.json")
admin = NexusAdmin(db_client=existing_admin_connector)
```

---

### `admin.authenticate()`

```python
admin.authenticate(
    user_id: str,
    api_key: str,
) -> Principal
```

Validate credentials and return a `Principal` for use in a `Context`.
The `user_id` must have been created via `admin.create_principal()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | Unique identifier for this principal. |
| `api_key` | `str` | API key issued at `create_principal()` time. Never log or store this value. |

**Returns:** `Principal`
**Raises:** `NexusPermissionError` if credentials are invalid or the user does not exist.

```python
principal = admin.authenticate(user_id="alice", api_key="...")
```

---

### `admin.create_department()`

```python
admin.create_department(
    department: str,
    organization: str | None = None,
) -> None
```

Create a department-level ApertureDB user with entity, connection, and index
read/write permissions. Principals in this department use these credentials
when `Memory` connects to ApertureDB.

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

### `memory.commit()`

```python
memory.commit(ctx: Context, info: Information) -> str
```

Store `Information` in ApertureDB as-is — no model calls, no embeddings. Fast. Returns the `memory_id` of the committed memory.

Use `commit()` when you need speed and plan to search by metadata (session, organization, time range) rather than semantic similarity.

**Returns:** `str` — the `memory_id`
**Raises:** `NexusConnectionError`, `NexusStorageError`, `NexusPermissionError`

```python
memory_id = memory.commit(ctx, info)
```

---

### `memory.process_and_commit()`

```python
memory.process_and_commit(ctx: Context, info: Information) -> str
```

Generate embeddings and summaries for all items in `info`, then store everything in ApertureDB. Slower than `commit()` but enables semantic (vector) search.

Requires at least one model configured in `aperture_nexus.json` or supplied via `info.log(embedding_model=...)`.

**Returns:** `str` — the `memory_id`
**Raises:** `NexusConfigError` if no model is configured; `NexusProcessingError` if model calls fail; `NexusConnectionError`, `NexusStorageError`, `NexusPermissionError`

```python
memory_id = memory.process_and_commit(ctx, info)
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
    print(f"Stored as memory: {task.memory_id}")
else:
    print(f"Failed: {task.error_message}")
    task.retry()
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
| `filters` | `dict \| None` | Metadata filters. Keys: `session_id`, `user_id`, `organization`, `purpose`. |
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

Returned by `memory.search()`. Exactly one content field is set,
corresponding to the modality of the stored memory.

```python
@dataclass
class SearchResult:
    score: float              # similarity (higher = more similar)
    modality: str             # "text" | "image" | "video" | "blob"
    memory_id: str
    session_id: str
    context_id: str
    timestamp: datetime

    # Content — exactly one is set:
    text: str | None = None
    image: PIL.Image | None = None
    video_url: str | None = None
    blob: bytes | None = None

    metadata: dict = field(default_factory=dict)
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

Create a named relationship between two contexts or memories. Builds the knowledge graph on top of ApertureDB `Connection` objects.

```python
memory.connect(source=ctx_q1, target=ctx_q2, relationship="follows")
memory.connect(source=memory_id_1, target=memory_id_2, relationship="related_to")
```

---

### `memory.remove()`

```python
memory.remove(memory_id: str) -> None
```

Remove a committed memory from ApertureDB.

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
    task.retry()
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
    priority: int = 0,
    restrictions: dict | None = None,
)
```

Either `session_id` or `session_name` is required. If `session_name` is provided and no session with that name exists, a new session is created on first `memory.commit()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `principal` | `Principal` | The authenticated user or agent. Required. |
| `session_id` | `str \| None` | ID of an existing session. Use `memory.generate_session_id()` for multi-participant sessions. |
| `session_name` | `str \| None` | Human-readable session name. Must be unique within a principal's scope. |
| `purpose` | `str \| None` | Why this interaction is happening. Stored as metadata; searchable. |
| `organization` | `str \| None` | Group scope for permission and search filtering. |
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
sid = memory.generate_session_id()
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
) -> None
```

Add one entry to the buffer. Multiple modalities can be combined in a single call (e.g. text + blob for "see attached PDF").

Validation happens eagerly at `log()` time — bad inputs raise `NexusValidationError` immediately.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str \| None` | Plain text. Long text is chunked automatically at commit time. |
| `image` | `str \| bytes \| PIL.Image \| np.ndarray \| None` | Image in any common form. File path, URL, bytes, PIL Image, or numpy array (HWC uint8 or float32). |
| `video` | `str \| bytes \| None` | Video file path, URL, or raw bytes. |
| `blob` | `str \| bytes \| None` | File path, URL, or raw bytes for any binary format. Requires `document_type`. Paths and URLs are resolved at commit time. |
| `document_type` | `str \| None` | File extension for blobs: `"pdf"`, `"mp3"`, `"docx"`, `"csv"`, etc. |
| `embedding` | `np.ndarray \| None` | Pre-computed embedding vector. Skips model call at commit time. Requires `embedding_model`. |
| `embedding_model` | `str \| None` | Name of the model that produced the embedding. Required when `embedding` is provided. |
| `metadata` | `dict \| None` | Arbitrary key-value properties stored alongside the entry. Keys must be strings; values must be `str`, `int`, `float`, or `bool`. Reserved keys (`context_id`, `session_id`, etc.) are rejected. |

**Raises:** `NexusValidationError` if input is invalid (missing file, wrong shape, missing `document_type` for blob, reserved metadata key, etc.)

> **Storage semantics for paths and URLs:** File paths and URLs passed to `image`, `video`, or `blob` are stored as references in the local buffer. Content is read from disk or network only when `memory.commit()` is called — not at `log()` time. Raw `bytes` are held in memory until commit.

```python
info = Information(context_id=ctx.id)

# Text
info.log(text="Customer says order #4821 never arrived")

# Images — any of these forms
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

# Metadata — custom properties stored alongside the entry
info.log(text="Order #4821 missing", metadata={"ticket_id": "T-99", "priority": 1})

# Pre-computed embedding — skips model call at commit time
info.log(image=img, embedding=my_vector, embedding_model="clip-vit-base-patch32")

# Combined — one log entry with multiple modalities
info.log(text="See attached invoice", blob="invoice.pdf", document_type="pdf")
```

### `info.remove()`

```python
info.remove(index: int) -> None
```

Remove a single pending entry from the buffer by 0-based position. Consistent with `memory.remove()` which removes a committed memory by ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `int` | 0-based position of the entry to remove. Negative indices (Python-style) are supported. |

**Raises:** `IndexError` if `index` is out of range. `NexusValidationError` if `index` is not an integer.

```python
info.log(text="preliminary draft")
info.log(text="final version")
info.remove(0)          # discard the draft; only "final version" commits
memory.commit(ctx, info)
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

Returned by `memory.async_process_and_commit()`. State is persisted in ApertureDB.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `task.status` | `str` | `"pending"` \| `"processing"` \| `"complete"` \| `"failed"` |
| `task.memory_id` | `str \| None` | Available when `status == "complete"` |
| `task.completed_at` | `datetime \| None` | Available when `status == "complete"` |
| `task.error` | `Exception \| None` | Available when `status == "failed"` |
| `task.error_message` | `str \| None` | Human-readable failure reason. Available when `status == "failed"` |
| `task.failed_at` | `datetime \| None` | Available when `status == "failed"` |

### Methods

```python
task.is_ready() -> bool
await task.wait() -> None    # async block until complete or failed
task.retry() -> None         # resubmit a failed task
```

```python
task = await memory.async_process_and_commit(ctx, info)
await task.wait()

if task.status == "complete":
    print(f"memory_id: {task.memory_id}")
elif task.status == "failed":
    print(f"failed: {task.error_message}")
    task.retry()
```

---

## `Principal`

Returned by `admin.authenticate()`. Passed to `Context` to identify who is
performing an action. Do not construct directly.

```python
principal = admin.authenticate(user_id="alice", api_key="...")

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
