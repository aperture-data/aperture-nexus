# aperture-nexus

**The Unified KMC (Knowledge, Memory, Context) Engine for Agentic State.**

aperture-nexus gives AI agents and applications a persistent, searchable memory layer that understands text, images, video, and more — powered by [ApertureDB](https://aperturedata.io)'s vector search, knowledge graph, and multimodal data capabilities.

---

## Quickstart

```bash
# One-time setup — creates your principal and writes NEXUS_API_KEY to .env
adb-nexus init
```

```python
import os
from aperture_nexus import Memory, Context, Information

memory = Memory()
principal = memory.authenticate(
    user_id="alice",
    api_key=os.environ["NEXUS_API_KEY"],
)

ctx = Context(
    principal=principal,
    session_name="support-2024-001",
    purpose="Customer reporting missing order",
)

info = Information(context_id=ctx.id)
info.log(text="Customer says order #4821 never arrived")
info.log(image="screenshot.png")

# Raw storage — fast, no model calls
memory.commit(ctx, info)

# Or: generate embeddings and store (requires models in config)
memory.process_and_commit(ctx, info)

# Search — permissions applied automatically
results = memory.search(query="missing orders last week")
```

---

## Installation

```bash
pip install aperture-nexus
```

Optional features:

```bash
pip install aperture-nexus[video]    # video frame extraction (opencv-python)
pip install aperture-nexus[clip]     # CLIP embeddings for text and images
pip install aperture-nexus[tokens]   # token-based text chunking
pip install aperture-nexus[metrics]  # Prometheus metrics export
pip install aperture-nexus[ui]       # web UI and REST API
pip install aperture-nexus[all]      # everything
```

> **Video embedding performance:** The built-in video embedder extracts frames
> with CLIP, which is slow for anything beyond short clips — each frame is
> embedded individually. For production workloads, pass a pre-computed embedding
> directly via `info.log(video=..., embedding=my_vector, embedding_model="...")`,
> or use `memory.commit()` (raw storage, no embedding) and add a dedicated video
> model later.

---

## ApertureDB

aperture-nexus requires a running ApertureDB instance.

**Quickest option — Docker Compose:**

```bash
docker compose up -d
```

Data persists across restarts. ApertureDB web UI is available at `http://localhost:8087`.

> **Community edition note:** The Docker image (`aperturedata/aperturedb-community`) is licensed for evaluation and development only — no production use. See the [ApertureDB community edition license](https://www.aperturedata.io/docker-license) and [contact ApertureData](https://aperturedata.io/contact) for production licensing.

**Mac users:** If port 55555 conflicts with another service:

```bash
APERTUREDB_PORT=15555 docker compose up -d
```

Then set the same port in your `.env`:
```
APERTUREDB_HOST=localhost
APERTUREDB_PORT=15555
```

---

## Setup

```bash
adb-nexus init      # interactive setup — creates aperture_nexus.json and .env
adb-nexus validate  # test your connection and config
```

---

## Documentation

Full documentation is in the [`docs/`](docs/) folder:

| Page | Contents |
|------|----------|
| [Concepts](docs/concepts.md) | The KMC model, ApertureDB storage mapping, architecture diagrams |
| [API Reference](docs/api-reference.md) | `Memory`, `Context`, `Information`, `MemoryTask`, exceptions |
| [Configuration](docs/configuration.md) | Every field in `aperture_nexus.json` with defaults and env var overrides |

---

## CLI

```bash
adb-nexus init                             # interactive setup wizard
adb-nexus init --defaults                  # non-interactive, all defaults
adb-nexus validate                         # test connection and config
adb-nexus validate --config path/to/x.json
adb-nexus stats                            # session stats
adb-nexus stats --detailed
adb-nexus ui                               # launch web UI (local only)
adb-nexus ui --host 0.0.0.0 --port 8080   # launch for network access
```

---

## Security

- No root required — all paths and ports are user-accessible
- Credentials via environment — never hardcoded in config or source
- UI binds to `127.0.0.1` by default — network access requires an explicit `api_key`
- `.env` is automatically added to `.gitignore` by `adb-nexus init`

---

## Enterprise design

aperture-nexus is designed for deployment at any scale — from a single
developer's Claude session to a multi-team organization.

### Credential separation

Admin credentials (ApertureDB `APERTUREDB_KEY`) are isolated to one place:
`adb-nexus init`. Everything else — application code, AI agent sessions,
search queries — uses only:

- **Regular ApertureDB credentials** — connection to the DB
- **`NEXUS_API_KEY`** — user-level credential written to `.env` by `init`

No application code ever needs admin credentials. A compromised session
cannot create or delete principals.

### Authentication flow

```
┌─────────────────────┐        ┌──────────────────────┐
│   adb-nexus init    │        │   Application / Agent │
│  (admin creds only) │        │  (regular creds only) │
│                     │        │                       │
│  create_principal() │        │  memory.authenticate()│
│  → NEXUS_API_KEY    │──.env─▶│  → Principal          │
│    written to .env  │        │  → Context            │
└─────────────────────┘        │  → memory.commit()    │
                                └──────────────────────┘
```

### Key rotation

If a principal's key needs to be replaced (new device, suspected exposure),
an admin rotates it — no DB schema changes, no session disruption:

```python
from aperture_nexus import NexusAdmin

admin = NexusAdmin()  # requires admin ApertureDB credentials
new_key = admin.rotate_key(user_id="alice")
# Deliver new_key to alice; update her .env
```

The previous key is invalidated immediately. All existing memories are
retained — rotation affects authentication only, not stored data.

### Multi-user and multi-tenant

Each principal has `user_id`, `department`, and `organization` properties
stamped on every stored object. This is the foundation for the v2
permissions model (Personal / Project / Team / Company retrieval scoping)
without requiring any schema migration.

---

## License

aperture-nexus is released under the [MIT License](LICENSE).

> **Note:** This license covers aperture-nexus only. ApertureDB is licensed separately — see the [community edition license](https://www.aperturedata.io/docker-license) and [contact ApertureData](https://aperturedata.io/contact) for production use.

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines, conventions, and PR workflow.
