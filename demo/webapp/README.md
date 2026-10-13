# Nexus demo web app

A single-page app that lets you ask questions about the aperture-nexus
codebase and see:

- the top semantically-matching chunks (file, snippet, score)
- an interactive graph of the Nexus memory, with the matching chunks
  highlighted

The app ingests this repo's own source and docs into Nexus, so the demo
is Nexus-storing-Nexus. Good for showing the aha: "my memory layer can
answer questions about itself, with provenance and a visible graph."

## Prerequisites

- The local ApertureDB Docker stack running (`docker compose up -d` from
  `aperture-nexus/`). The webapp defaults to `localhost:55556` with
  `admin`/`admin`.
- Python packages: `fastapi`, `uvicorn[standard]`, plus the aperture-nexus
  `[clip]` extra for CLIP text embeddings.

## One-time ingest

```bash
python demo/webapp/ingest.py              # loads the Nexus repo
python demo/webapp/ingest.py --reset      # wipe and reload
```

The script:

- Creates a demo principal (`nexus-demo-user`, workspace `nexus-demo`) and
  caches its API key under `demo/webapp/.nexus-demo.key`.
- Loads ~18 key files (README, CLAUDE.md, core src modules, user-facing
  docs), chunks each at ~1200 chars so chunk text lands inline on the
  Descriptor, and commits each file via `Memory.process_and_commit()`
  with the CLIP `ViT-B-32` text encoder.
- Is idempotent: re-running with no flag is a no-op; `--reset` wipes.

Expect ~1 minute wall-clock on first run (CLIP model download + ~326
chunks to embed).

## Run the server

```bash
# Minimal — raw results only, no synthesised answer.
uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0

# With LLM synthesis (recommended for the demo): set your Anthropic key.
#   Workspace-scoped key (sk-ant-wrkspc_...): the key alone is enough.
#   User-scoped key (sk-ant-usr-...): also set ANTHROPIC_WORKSPACE_ID.
ANTHROPIC_API_KEY=sk-ant-... \
  ANTHROPIC_WORKSPACE_ID=wrkspc_... \
  uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0
```

Open `http://localhost:9001/` in a browser.

**Model selection.** Defaults to `claude-sonnet-5-5`. Override with
`NEXUS_DEMO_LLM=claude-haiku-5-5` for faster/cheaper, or
`NEXUS_DEMO_LLM=claude-opus-4-7` for highest-quality.

**Fallback behaviour.** If the `anthropic` SDK isn't installed or the key
isn't set, the server starts fine — the UI shows a soft notice telling
you how to turn synthesis on and still displays the raw search chunks
and the graph. Good for demoing without a network connection.

- **Chat (left):** type a question, hit Enter. Returns 6 ranked
  snippets with file path, folder, score, and the chunk text.
- **Graph (right):** iframe that reloads with each search, showing the
  matched chunks highlighted inside the Nexus memory graph. Hover to
  highlight connections. Click a node for its details and the query
  that fetches it.

## What the app shows

- `GET /` — the UI
- `POST /search` — JSON `{query, k}` →
  `{answer, answer_model, results[]}`. `answer` and `answer_model` are
  `null` when the LLM is disabled; `results[]` is always present.
- `GET /graph/schema` — the whole workspace's schema (classes + edges)
- `GET /graph/query?q=...` — a graph centred on the Blob chunks that
  match the query, expanded 2 hops up the graph (Commit → Context →
  Session → User)

## Hosting it elsewhere

**The hard part is ApertureDB.** The app is a thin FastAPI wrapper and
runs anywhere Python runs — the challenge is where ApertureDB lives.

- **HuggingFace Spaces (Docker):** write a single `Dockerfile` that
  installs the aperturedb-community image + lenz + this app, bake the
  ingest into the image build, and `uvicorn` on container start. Free
  CPU tier should be enough for the demo but check memory headroom —
  ApertureDB wants ~1 GB.
- **Dedicated VM (Fly.io, Railway, DigitalOcean):** more
  predictable than Spaces; a single $5/mo VM runs the whole stack
  comfortably.
- **Managed ApertureDB (cloud):** separate the webapp from the DB. Deploy
  the webapp on any free Python host (Streamlit Cloud, Railway free
  tier), set `APERTUREDB_JSON` to point at the cloud instance.

Whatever path you pick, keep the one-time ingest step out of the request
cycle — bake it into the deploy or run it once with `--reset`.

## File layout

```
demo/webapp/
├── app.py              # FastAPI backend
├── ingest.py           # One-time repo ingest
├── graph_viz.py        # Interactive ApertureDB graph (copied from
│                       #   aperturedb-applications/notebooks/gcs-auto-annotate/)
├── templates/
│   └── index.html      # Chat + graph UI
├── graphs/             # Generated graph HTML files (served at /graphs/*)
├── static/             # (reserved for future assets)
├── .nexus-demo.key     # Cached API key (gitignore; recreated by ingest)
└── README.md
```

## Known gaps

- `Memory` does not chunk long text at commit time today despite the
  docstring on `Information.log()` saying it does — the ingest script
  pre-chunks so chunks fit under Nexus's 2000-char inline-text threshold.
  When Memory ships automatic chunking, the ingest can go back to one
  `log()` call per file.
- `SearchResult` does not expose `commit_id` or `entry_id` publicly
  (`entry_id` is internal). The graph anchors on `file_path` from the
  result's `metadata` instead; works for this demo, but any future demo
  anchoring on commit should wait for the Memory API to surface those.
- The direct-ApertureDB query path shown in the Nexus workflow diagram
  is not demonstrated in this app yet; both query paths currently go
  through `Memory.search()`.
- Semantic search uses CLIP's text tower (`ViT-B-32`) because that is
  the embedding model Nexus ships with today. CLIP's text tower is
  weak at typo tolerance (subword tokenizer, 77-token limit). Typos
  like `inforamation` land far from `information` in CLIP's text
  embedding space. A sentence-transformers model would handle this
  better and is on the roadmap (see `nexus_benchmark_resilience_metrics`
  memory for the benchmark context).
- `CLAUDE.md` is intentionally excluded from the ingest — it is the
  Claude Code dev guide, dense enough to dominate every search result.
  Add it back only if you specifically want to query dev-guide material.
