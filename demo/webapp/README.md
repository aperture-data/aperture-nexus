# ask-the-codebase: a codebase-ingestion app on Nexus

A single-page web app that ingests any git repo into Nexus and lets
you query it three ways:

- **Semantic chat** — natural-language question, Nexus retrieves the
  matching chunks, an LLM synthesises an answer with per-claim citations.
- **Scoped filters** — narrow by Session (PR), Principal (author), or
  Department. Live counts shown on every chip.
- **Graph lineage** — walk the memory graph from a file to every PR /
  commit / principal that ever touched it.

The example repo is this one (aperture-nexus itself) because we had
it handy, but `python demo/webapp/ingest.py --repo /path/to/other/repo`
works on any git repo on disk.

## Setup — replicating on a fresh machine

### 1. ApertureDB

Pick one.

**Option A — Community edition in Docker (easiest for local dev):**
From `aperture-nexus/`:
```bash
docker compose up -d
```
That brings up ApertureDB (community), the Lenz TCP gateway, and the
Web UI (`http://localhost:8087`). The webapp defaults to this stack
at `localhost:55556`, `admin`/`admin`.

**Option B — ApertureDB Cloud or VPC** (production-grade, managed):
Get a connection key from the ApertureData console, then export it
before running the webapp (or write it into `aperture_nexus.json`):
```bash
export APERTUREDB_KEY=adbp_...
```
or JSON form for an on-prem/VPC deployment:
```bash
export APERTUREDB_JSON='{"host":"your-host","port":55555,
    "username":"...","password":"...","use_ssl":true}'
```
Everything the webapp does — ingest, search, graph queries, lineage —
runs identically against Cloud/VPC. The only difference is where the
TCP endpoint lives.

### 2. Python environment

Python 3.10+. From `aperture-nexus/`:
```bash
pip install -e ".[sentence-transformers,clip,ui]"
pip install anthropic openai playwright  # LLM clients + optional GIF recorder
```
`anthropic` and `openai` are both optional — install only the one you
plan to use. Install `playwright` only if you want to re-record the GIF.

### 3. Ingest

Three scripts, each independent. Run against the aperture-nexus repo:
```bash
python demo/webapp/ingest.py          # ~17 files, ~5 min
python demo/webapp/ingest_git.py      # all git commits, grouped per PR
python demo/webapp/ingest_images.py   # architecture diagrams
```

Or against any other git repo on disk:
```bash
python demo/webapp/ingest.py       --repo /path/to/other/repo
python demo/webapp/ingest_git.py   --repo /path/to/other/repo
```

Each script accepts `--reset` to wipe its own slice of memory before
reloading.

### 4. LLM key (optional)

The synthesised-answer panel is powered by an LLM. Pick one of
Anthropic, OpenAI, or any OpenAI-compatible endpoint (Together, Groq,
Fireworks, LocalAI, …). You can also run with no key — raw search
chunks and the graph still work.

**Anthropic** — workspace-scoped key (`sk-ant-wrkspc_...`) is simplest:
```bash
export ANTHROPIC_API_KEY=sk-ant-wrkspc-...
```
User-scoped key (`sk-ant-usr-...`) — also set the workspace id:
```bash
export ANTHROPIC_API_KEY=sk-ant-usr-...
export ANTHROPIC_WORKSPACE_ID=wrkspc_...
```

**OpenAI** (or an OpenAI-compatible endpoint):
```bash
export OPENAI_API_KEY=sk-...
# optional — point at Together, Groq, Fireworks, Ollama, LocalAI, etc.
export OPENAI_BASE_URL=https://api.together.xyz/v1
```

**Choosing / overriding** — if both keys are set, Anthropic wins by
default. Force a provider with:
```bash
export NEXUS_DEMO_LLM_PROVIDER=openai    # or 'anthropic'
```
Override the model with `NEXUS_DEMO_LLM`. Defaults: `claude-sonnet-5-5`
on Anthropic, `gpt-4o-mini` on OpenAI.

### 5. Start the server

```bash
uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0
```

Open `http://localhost:9001`.

## What the app shows

- **Chat (left):** chips pre-fill four demo questions. Type anything,
  hit Enter. The ANSWER card streams in token-by-token; SOURCES below
  it show the retrieved chunks with file paths, commit SHAs, authors,
  dates, scores.
- **Filters (left, above chat):** Session dropdown, Principal dropdown,
  Department chips. Live counts fetched from `/counts`. Narrow a search
  to a specific PR or author.
- **Graph (right):** reloads with each query. Hover a node for its
  connections; click for its properties and the ApertureDB query that
  fetches it. The "Lineage of memory.py" chip triggers a graph-only
  query that walks `git log --follow` + Nexus's graph edges to show
  every PR that ever modified the file.

## Endpoint contract

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | UI |
| `POST` | `/search` | `{query, k, session_name?, user_id?, department?}` → `{results[]}` |
| `GET`  | `/synthesize?q=…` | SSE stream: `sources` event (chunks), then `chunk` events (streamed LLM text), then `done`. Same filter params as /search. |
| `GET`  | `/counts` | `{sessions, principals, departments, total_blobs}` with per-bucket chunk counts. Powers the filter chip labels. |
| `GET`  | `/graph/schema` | Interactive HTML page: full workspace schema. |
| `GET`  | `/graph/query?q=…` | Interactive HTML: graph centred on the chunks that match the query. Same filter params. |
| `GET`  | `/graph/lineage?file=…` | Interactive HTML: every PR / commit / principal that ever touched a given file. |
| `GET`  | `/figure/{id}` | Serves a named architecture diagram. |

## File layout

```
demo/webapp/
├── app.py              # FastAPI backend
├── ingest.py           # Walks a repo's source + docs
├── ingest_git.py       # Walks `git log --patch`, one Session per PR
├── ingest_images.py    # Pairs each architecture diagram with its caption
├── graph_viz.py        # Reusable, schema-agnostic ApertureDB → interactive
│                       # graph page. Drop into any other ApertureDB app.
├── record_gif.py       # Playwright headless recording of the 4-beat demo
├── templates/
│   └── index.html      # Chat + filters + graph UI
├── graphs/             # Generated graph HTML (gitignored)
└── RECORDING.md        # Manual GIF playbook
```

## Hosting elsewhere

The webapp is a thin FastAPI wrapper. The ApertureDB side is the only
hosting question:

- **ApertureDB Cloud or VPC** — point `APERTUREDB_KEY` or
  `APERTUREDB_JSON` at the managed endpoint. Deploy the webapp on any
  Python host (Streamlit Cloud, Fly.io, Railway, HuggingFace Spaces).
  Clean separation of concerns and the one we'd recommend for a shared
  public demo.
- **Community in Docker, bundled on HF Spaces** — one Dockerfile with
  ApertureDB + Lenz + the webapp. Free tier works for small datasets;
  memory headroom is the main constraint.
- **Dedicated VM** — Fly.io, Railway, DigitalOcean. One modest VM
  runs the whole stack with room to grow.

Whatever path you pick, keep the one-time ingest step out of the
request cycle — bake it into the image build, or run it once with
`--reset` after deploy.
