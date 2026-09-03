# aperture-nexus

**The Cognition Engine for Enterprise AI.**

aperture-nexus enables AI workflows, agents, and the humans working
alongside them to establish context, capture knowledge across text,
images, audio, video, and more — and commit it to memory for search
and retrieval, powered by [ApertureDB](https://aperturedata.io)'s
vector search and knowledge graph.

The three building blocks — **Knowledge** (what was captured),
**Memory** (the engine that stores and retrieves it), and **Context**
(who did what, in which session, and why) — together form a complete
cognition layer that scales from a single developer session to a
multi-team enterprise deployment.

![aperture-nexus hello world](demo/demo.gif)

---

## Try It Now — One Command

```bash
git clone https://github.com/aperturedata/aperture-nexus
cd aperture-nexus
docker compose --profile demo run --rm nexus-demo
```

Requires [Docker Desktop](https://docs.docker.com/get-started/get-docker/)
(Mac / Windows) or [Docker Engine](https://docs.docker.com/engine/install/)
(Linux). Docker Compose is included with both.

The first run downloads Docker images if not already cached — allow
**3–5 minutes**. Subsequent runs start in seconds.

The walkthrough pauses at each step so you can read what's happening before
it proceeds. All services — ApertureDB, the Lenz gateway, and the web UI —
start automatically. Explore stored data at **http://localhost:8087**
(ApertureDB web UI) while the demo is running.

> **Note:** Use `run --rm` (above), not `docker compose up`. The `up` variant
> does not allocate a terminal, so the demo cannot pause for input and will
> scroll straight through.

When the demo finishes, the stack keeps running so you can explore the web UI.
Stop it when you're done:

```
docker compose down      # stop (data is preserved)
docker compose down -v   # stop and wipe all data
```

> **Demo vs. dev stack:** The `--profile demo` flag runs a self-contained
> guided walkthrough that creates and cleans up all demo data automatically.
> To start a persistent ApertureDB stack for your own development — without
> the demo — omit the flag: `docker compose up -d`. See the
> [ApertureDB](#aperturedb) section below.

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
info.log(text="Order was marked delivered on April 3rd")

# Store — no model calls required
memory.commit(ctx, info)

# Search by metadata — works immediately, no embedding model needed
results = memory.search(filters={"session_name": "support-2024-001"})
```

For semantic search, generate embeddings at commit time and query by content:

```python
# Requires an embedding model — see the Models section below
memory.process_and_commit(ctx, info)
results = memory.search(query="missing orders last week")
```

---

## Installation

aperture-nexus v0.1.0 is not yet on PyPI. Install directly from the repository:

```bash
git clone https://github.com/aperturedata/aperture-nexus
cd aperture-nexus
pip install .
```

Optional features:

```bash
pip install ".[video]"    # video frame extraction (opencv-python)
pip install ".[clip]"     # CLIP embeddings for text and images
pip install ".[tokens]"   # token-based text chunking
pip install ".[metrics]"  # Prometheus metrics export
pip install ".[all]"      # everything
```

> **Video embedding performance:** The built-in video embedder extracts frames
> with CLIP, which is slow for anything beyond short clips — each frame is
> embedded individually. For production workloads, pass a pre-computed embedding
> directly via `info.log(video=..., embedding=my_vector, embedding_model="...")`,
> or use `memory.commit()` (raw storage, no embedding) and add a dedicated video
> model later.

---

## Models

Models are only needed for `process_and_commit()`. If you use `commit()` for
raw storage — or supply pre-computed embeddings — you can skip this section.

### CLIP — One Model for Text, Images, and Video

CLIP works across all modalities in a shared embedding space: a text query
can find images, and an image query can find text. It is the simplest
configuration and the recommended starting point.

Install:

```bash
pip install aperture-nexus[clip]
```

Configure in `aperture_nexus.json`:

```json
{
    "models": {
        "text_embedding": "ViT-B/16",
        "image_embedding": "ViT-B/16",
        "video_embedding": "ViT-B/16"
    }
}
```

Two limitations to keep in mind:

- **77-token text limit.** CLIP truncates input at 77 tokens (~300 characters).
  Longer passages are chunked automatically using `processing.text_chunk_size`.
  For long-document workloads, a dedicated text model gives better recall.
- **Trained on image-text pairs.** CLIP retrieval quality for pure-text
  queries is lower than purpose-built text models such as BGE-M3 or
  `text-embedding-3-small`.

### Dedicated Text Model + CLIP for Images and Video

For workloads that mix document retrieval with image search, use a dedicated
text model alongside CLIP for images and video:

```json
{
    "models": {
        "text_embedding": "text-embedding-3-small",
        "image_embedding": "ViT-B/16",
        "video_embedding": "ViT-B/16"
    }
}
```

> **Note:** Text and image embeddings are stored in separate ApertureDB
> DescriptorSets. Cross-modal search (text query → image results) requires
> a shared embedding space — CLIP provides this; separate models do not.

### Pre-Computed Embeddings

Skip model configuration by supplying embeddings directly at log time:

```python
info.log(
    image="photo.jpg",
    embedding=my_vector,           # numpy array
    embedding_model="my-model",    # recorded in metadata
)
memory.commit(ctx, info)           # no model calls
```

The same pattern works for text, blob, and video entries.

---

## ApertureDB

aperture-nexus requires a running ApertureDB instance.

**Quickest option — Docker Compose:**

```bash
docker compose up -d
```

Starts ApertureDB, the Lenz TCP gateway, and the ApertureDB Web UI.
Data persists across restarts. aperture-nexus connects on port `55556`
by default.

```
docker compose down      # stop (data preserved)
docker compose down -v   # stop and wipe all data
```

**ApertureDB Web UI** is available at `http://localhost:8087` after startup.
Use it to inspect what is stored in ApertureDB — entities, connections,
descriptors, images — which is a useful way to verify that memories were
committed correctly.

> **Community edition note:** The Docker image
> (`aperturedata/aperturedb-community`) is licensed for evaluation and
> development only — no production use. See the
> [ApertureDB community edition license](https://www.aperturedata.io/docker-license)
> and [contact ApertureData](https://www.aperturedata.io/contact-us) for
> production licensing.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `APERTUREDB_KEY` | One of these three | Encoded connection key. Generated by `adb config`. Preferred for production. |
| `APERTUREDB_JSON` | | JSON connection config: `{"host":"...","port":55556,"username":"...","password":"...","use_ssl":false}`. Used for local dev and the Docker Compose setup. |
| `APERTUREDB_CONFIG` | | Name of a saved `adb config` configuration to use. |
| `NEXUS_API_KEY` | Yes (app code) | App-level key for `memory.authenticate()`. Written to `.env` by `adb-nexus init`. |
| `APERTURE_NEXUS_CONFIG` | No | Path to `aperture_nexus.json`. Overrides automatic discovery. |
| `APERTURE_NEXUS_LOG_LEVEL` | No | Internal log level: `DEBUG` `INFO` `WARNING` `ERROR`. Default: `ERROR`. |
| `APERTURE_NEXUS_UI_API_KEY` | When UI is hosted | API key for the web UI. Required when `ui.host` is not `127.0.0.1`. Never put this value in `aperture_nexus.json`. |

ApertureDB credentials are resolved in this priority order:

1. `APERTUREDB_KEY` — a single encoded key (simplest, recommended for production)
2. `APERTUREDB_JSON` — JSON string with host, port, username, password, use_ssl
3. Active `adb` configuration set via `adb config`

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
| [Customer Support Agent](docs/customer-support-agent.md) | Multi-agent multimodal pipeline with CLIP semantic search — worked example |

### Knowledge Graph at a Glance

Every `commit()` creates a `NexusCommit` entity connected to its `NexusContext`,
`NexusUser`, and content (blobs, images, videos). The full graph:

![aperture-nexus schema graph](demo/nexus_schema.png)

See [Concepts → ApertureDB Storage Mapping](docs/concepts.md#aperturedb-storage-mapping)
for the per-connection breakdown.

---

## CLI

```bash
adb-nexus init                             # interactive setup wizard
adb-nexus init --defaults                  # non-interactive, all defaults
adb-nexus validate                         # test connection and config
adb-nexus validate --config path/to/x.json
```

---

## Security

- No root required — all paths and ports are user-accessible
- Credentials via environment — never hardcoded in config or source
- UI binds to `127.0.0.1` by default — network access requires an explicit `api_key`
- `.env` is automatically added to `.gitignore` by `adb-nexus init`

---

## Enterprise Design

aperture-nexus is designed for deployment at any scale — from a single
developer's Claude session to a multi-team organization.

### Credential Separation

Admin credentials (ApertureDB `APERTUREDB_KEY`) are isolated to one place:
`adb-nexus init`. Everything else — application code, AI agent sessions,
search queries — uses only:

- **Regular ApertureDB credentials** — connection to the DB
- **`NEXUS_API_KEY`** — user-level credential written to `.env` by `init`

No application code ever needs admin credentials. A compromised session
cannot create or delete principals.

### Authentication Flow

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

### Key Rotation

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

### Multi-User and Multi-Tenant

Each principal has `user_id`, `department`, and `organization` properties
stamped on every stored object. This is the foundation for the v2
permissions model (Personal / Project / Team / Company retrieval scoping)
without requiring any schema migration.

---

## Roadmap

v0.1 ships the core KMC API — commit, search, connect, and authenticate — validated against live ApertureDB.

**Coming in v2:**

| Feature | What it enables |
|---------|----------------|
| MCP server (`adb-nexus mcp`) | Persistent memory in Claude Code, Cursor, Windsurf, and any MCP-compatible client |
| Memory updates with lineage | `Memory.update()` — supersede a memory while retaining full history via `superseded_by` |
| Preference profiles | Per-category preferences (`coding`, `writing`, `email`) stored as searchable memories |
| Visibility model | `private` / `department` / `organization` / `shared` — enforced automatically at search time |
| Hybrid search | KNN vector search + metadata filters combined in a single query |
| GraphRAG neighborhood search | `search(neighborhood=2)` — traverse the knowledge graph from matched results |
| Recency weighting | Configurable decay in search scoring; `search(lookback=30)` |
| Memory archiving + retention | Archive, TTL policies, and LLM-based consolidation of older memories |
| Web UI | Browser interface for browsing sessions, contexts, memories, and search |
| Async commit (validated) | `async_process_and_commit()` re-exposed after live integration testing |

**Shape what gets built:** [Join the v2 discussion →](https://github.com/aperturedata/aperture-nexus/discussions)

Tell us what you're building and what would unblock you. We prioritize based on real use cases.

---

## License

aperture-nexus is released under the [MIT License](LICENSE).

> **Note:** This license covers aperture-nexus only. ApertureDB is licensed separately — see the [community edition license](https://www.aperturedata.io/docker-license) and [contact ApertureData](https://www.aperturedata.io/contact-us) for production use.

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines, conventions, and PR workflow.
