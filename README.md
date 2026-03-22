# aperture-nexus

**The Unified KMC (Knowledge, Memory, Context) Engine for Agentic State.**

aperture-nexus gives AI agents and applications a persistent, searchable memory layer that understands text, images, video, and more — powered by [ApertureDB](https://aperturedata.io)'s vector search, knowledge graph, and multimodal data capabilities.

---

## What It Does

- **Context** — capture who is doing what, in which session, and why
- **Information** — buffer multimodal inputs (text, images, video, blobs) during a session
- **Memory** — commit, process, connect, and search across everything stored

```python
from aperture_nexus import Memory, Context, Information

memory = Memory()
principal = memory.authenticate(user_id="alice", api_key="...")

# Describe the session
ctx = Context(
    principal=principal,
    session_name="support-2024-001",
    purpose="Customer reporting missing order",
    organization="AcmeCorp",
)

# Collect multimodal inputs during the session
info = Information(context_id=ctx.id)
info.log(text="Customer says order #4821 never arrived")
info.log(image="screenshot.png")   # file path, URL, PIL Image, or numpy array

# Store raw (fast) or process with models (embeddings, summarization)
memory.commit(ctx, info)
# or
memory.process_and_commit(ctx, info)

# Search later — permissions applied automatically
results = memory.search(
    query="missing orders last week",
    filters={"organization": "AcmeCorp"}
)
```

---

## Installation

### Core (no optional features)

```bash
pip install aperture-nexus
```

### With optional features

```bash
pip install aperture-nexus[video]     # video scene detection
pip install aperture-nexus[tokens]    # token-based text chunking
pip install aperture-nexus[metrics]   # Prometheus metrics export
pip install aperture-nexus[ui]        # web UI and REST API
pip install aperture-nexus[all]       # everything
```

### ApertureDB

aperture-nexus requires a running ApertureDB instance.

**Quickest option — Docker Compose (recommended):**

```bash
docker compose up -d
```

This starts ApertureDB with persistent storage. Data survives restarts.
To stop: `docker compose down`. To wipe data: `docker compose down -v`.

**Mac users:** If port 55555 conflicts with another service, use a different host port without editing any files:

```bash
APERTUREDB_PORT=15555 docker compose up -d
```

Then set the same port in your `.env`:
```
APERTUREDB_HOST=localhost
APERTUREDB_PORT=15555
```

**Bare Docker (no persistence):**
```bash
docker run -p 55555:55555 aperturedata/aperturedb-community
```

Or follow the [ApertureDB setup guide](https://docs.aperturedata.dev/Setup/server/Local).

---

## Setup

Run the setup wizard after installation:

```bash
adb-nexus init
```

This walks you through:
1. ApertureDB connection (encoded key, or host/port/user/password)
2. AI models (optional — only needed for `process_and_commit()`)
3. Processing config (defaults recommended for most users)
4. UI and metrics (disabled by default)

Generated files:
- `aperture_nexus.json` — your config (see [Config Reference](aperture_nexus_config.md))
- `aperture_nexus.schema.json` — IDE autocomplete and validation
- `aperture_nexus_config.md` — full field reference
- `.env` — credentials (automatically added to `.gitignore`)

Verify your setup at any time:

```bash
adb-nexus validate
```

---

## Deployment

### Local Development

```bash
pip install aperture-nexus
adb-nexus init
python my_script.py
```

### Virtual Machine

```bash
pip install aperture-nexus
adb-nexus init
export APERTUREDB_KEY="adbp_xxxx..."
python my_script.py
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install aperture-nexus

COPY aperture_nexus.json .
COPY my_app/ ./my_app/

# Credentials via environment — never bake into the image
ENV APERTUREDB_KEY=""
ENV APERTURE_NEXUS_LOG_LEVEL="ERROR"

CMD ["python", "my_app/main.py"]
```

```bash
docker build -t my-app .
docker run -e APERTUREDB_KEY="adbp_xxxx..." my-app
```

### Docker Compose (app + ApertureDB together)

```yaml
version: "3.9"
services:
  aperturedb:
    image: aperturedata/aperturedb-community
    ports:
      - "55555:55555"
    volumes:
      - aperturedb_data:/aperturedb/db

  app:
    build: .
    environment:
      APERTUREDB_HOST: aperturedb
      APERTUREDB_PORT: 55555
      APERTUREDB_USER: admin
      APERTUREDB_PASSWORD: ${APERTUREDB_PASSWORD}
      APERTURE_NEXUS_LOG_LEVEL: ERROR
    depends_on:
      - aperturedb
    volumes:
      - ./aperture_nexus.json:/app/aperture_nexus.json:ro

volumes:
  aperturedb_data:
```

---

## Multi-User Sessions

One session can have multiple participants. Each participant gets their own `Context` into the shared session:

```python
# Customer support: customer + AI agent + human support agent
sid = memory.generate_session_id()

ctx_customer = Context(principal=customer_principal, session_id=sid, purpose="Order inquiry")
ctx_ai       = Context(principal=ai_principal,       session_id=sid, purpose="First response")
ctx_human    = Context(principal=human_principal,    session_id=sid, purpose="Escalation")

# Each participant logs independently — attribution is implicit
info_customer = Information(context_id=ctx_customer.id)
info_customer.log(text="My order #4821 never arrived")

info_ai = Information(context_id=ctx_ai.id)
info_ai.log(text="I can see your order was shipped on Monday")

# Search across the full session — all participants' information returned
results = memory.search(query="order 4821", filters={"session_id": sid})
```

One user can also participate in multiple sessions — the relationship is many-to-many.

---

## Async Processing

For production workloads where model inference cannot block:

```python
import asyncio
from aperture_nexus import Memory, Context, Information

async def main():
    memory = Memory()
    principal = memory.authenticate(user_id="alice", api_key="...")

    ctx = Context(principal=principal, session_name="batch-2024-001")
    info = Information(context_id=ctx.id)
    info.log(text="Quarterly earnings report")
    info.log(blob=open("report.pdf", "rb").read(), document_type="pdf")

    # Returns a MemoryTask immediately — does not block
    task = await memory.async_process_and_commit(ctx, info)
    print(task.status)   # "pending"

    await task.wait()

    if task.status == "complete":
        print(f"Stored as memory: {task.memory_id}")
    else:
        print(f"Failed: {task.error_message}")
        task.retry()

asyncio.run(main())
```

Monitor pending and failed tasks:

```python
for task in memory.failed_commits():
    print(task.error_message)
    task.retry()
```

---

## Multimodal Inputs

`Information.log()` accepts the most natural form for each modality — conversion happens internally:

| Input | Accepted forms |
|-------|---------------|
| Text | `str` |
| Image | file path, URL, `bytes`, PIL `Image`, `numpy.ndarray` |
| Video | file path, URL, `bytes` |
| Blob | `bytes` + `document_type` (e.g. `"pdf"`, `"mp3"`, `"docx"`) |
| Embedding | `numpy.ndarray` + `embedding_model` |

```python
info.log(text="Meeting notes from Q1 review")
info.log(image="photo.jpg")
info.log(image="https://example.com/photo.jpg")
info.log(image=pil_image)
info.log(image=numpy_array)
info.log(video="recording.mp4")
info.log(blob=open("contract.pdf", "rb").read(), document_type="pdf")
info.log(blob=open("call.mp3", "rb").read(), document_type="mp3")

# Pre-computed embedding — skips model call
info.log(image=img, embedding=my_embedding, embedding_model="clip-vit-base-patch32")

# Mixed in one entry
info.log(text="See attached invoice", blob=pdf_bytes, document_type="pdf")
```

---

## Web UIs

aperture-nexus involves two separate web interfaces — one for ApertureDB, one for aperture-nexus itself.

### ApertureDB Web UI

ApertureDB ships with its own web UI for exploring and visualizing the underlying database objects (entities, connections, descriptors, images, videos). It is not part of aperture-nexus.

Access it after running `docker compose up -d` at:
```
http://localhost:8087
```

Use this to inspect raw ApertureDB objects, debug schema issues, or verify that aperture-nexus is storing data correctly.

### aperture-nexus UI

The aperture-nexus UI is a higher-level interface focused on the KMC model — sessions, contexts, memories, and search. It sits on top of ApertureDB and shows concepts in aperture-nexus terms, not raw database objects.

```bash
pip install aperture-nexus[ui]
adb-nexus ui
# Opens at http://127.0.0.1:8000
```

Includes session browser, multimodal search, MemoryTask monitor, and REST API at `/api/v1/` with interactive docs at `/docs`.

For hosted deployments:

```bash
export APERTURE_NEXUS_UI_API_KEY="your-secret-key"
adb-nexus ui --host 0.0.0.0 --port 8080
```

---

## Error Handling

```python
from aperture_nexus.exceptions import (
    NexusError,           # base — catch all aperture-nexus errors
    NexusConfigError,     # misconfiguration
    NexusValidationError, # bad input
    NexusConnectionError, # ApertureDB unreachable
    NexusPermissionError, # insufficient permissions
    NexusProcessingError, # model call failed
    NexusStorageError,    # ApertureDB rejected the write
)

try:
    memory.commit(ctx, info)
except NexusConnectionError:
    # ApertureDB unreachable — check connection
    # Run: adb-nexus validate
except NexusPermissionError:
    # Principal lacks permission for this operation
except NexusStorageError:
    # ApertureDB rejected the write
except NexusError as e:
    # Catch-all for any aperture-nexus error
    print(e)
```

---

## CLI Reference

```bash
adb-nexus init                             # interactive setup wizard
adb-nexus init --defaults                  # non-interactive, all defaults
adb-nexus validate                         # test connection and config
adb-nexus validate --config path/to/config.json
adb-nexus stats                            # session stats summary
adb-nexus stats --detailed                 # full breakdown
adb-nexus ui                               # launch web UI (local only)
adb-nexus ui --host 0.0.0.0 --port 8080   # launch for network access
```

---

## Security

- **No root required** — all paths and ports are user-accessible
- **Credentials via environment** — never hardcoded in config or source code
- **Local by default** — UI binds to `127.0.0.1`; network access requires an explicit `api_key`
- **Permission enforcement** — all operations check the caller's `Principal`
- **`.env` gitignored** — `adb-nexus init` adds `.env` to `.gitignore` automatically

---

## Configuration

See [aperture_nexus_config.md](aperture_nexus_config.md) for the full configuration reference.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines, conventions, and PR workflow.
