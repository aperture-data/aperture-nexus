# Ask the Codebase: an interactive demo

A small web app that lets you ask natural-language questions about an
aperture-nexus codebase and see:

- a Claude-synthesised **answer** that cites specific files and commits
- the **source chunks** that fed the answer — file snippets, git
  commits (author, date, diff), and architecture diagrams rendered inline
- an interactive **memory graph** on the right, re-rendered on every
  search, showing exactly which pieces of Nexus's memory match

![ask-the-codebase demo](https://raw.githubusercontent.com/aperture-data/aperture-nexus/main/demo/ask-the-codebase.gif)

Live locally at `http://localhost:9001` once you run the three ingest
scripts and start the server.

---

## How this is different from "Coding Assistant with Continuity"

[Coding Assistant with Continuity](./coding-assistant.md) is a *scripted
API walkthrough*: three sessions over multiple days, showing how
`Context.purpose` and `session_name` keep an agent coherent across time.
It's about the agent remembering **itself**.

**Ask the Codebase** is a *running UI over a committed memory*: it
assumes someone has already ingested the repo (files, git history,
diagrams) into Nexus and now provides a chat interface for anyone to
query it. It's about an agent — or a human — remembering **a project**.

The two are complementary. Coding Assistant shows the API primitives in
code. Ask the Codebase shows what you can build on top of them when the
memory already exists.

---

## What's ingested

The demo uses the aperture-nexus repo itself as a worked example — Nexus
remembering Nexus. Three ingest scripts populate a single Nexus session
(`aperture-nexus-codebase`) with three kinds of memory, all searchable
together:

| Script | What it loads | Nexus shape |
|---|---|---|
| `ingest.py` | Core source + user-facing docs, chunked ~1200 chars | 1 Principal, 3 Contexts (root/src/docs), 17 Commits, ~310 Blobs |
| `ingest_git.py` | Full `git log --patch` for the repo | 2 Principals (both of Vishakha's email addresses), 21 Contexts (one per git commit, with the commit subject as `purpose`), 21 Commits, ~1320 Blobs |
| `ingest_images.py` | The four architecture diagrams, each as a paired caption + image | 1 Context, 4 image Entries + 4 caption Blobs |

Everything shares the same `session_name` so a single `Memory.search()`
query sees all of it. The graph on the right shows how the pieces
connect — a question that touches the architecture surfaces the diagram
entity *and* the context it lives in *and* the session that groups it
with the codebase.

## What makes the answers good

Three choices in the stack matter:

1. **A real text embedding model.** The demo configures
   `BAAI/bge-small-en-v1.5` through Nexus's new
   `SentenceTransformersEmbedder`. CLIP's text tower — Nexus's prior
   default — was trained on image-caption pairs and ranks the same few
   diff-heavy chunks for every query. BGE-small ranks the exact
   target commit for specific-attribution questions
   like "who added the department and organization fields to Principal
   and when?" at position 1 with a clear score separation.
2. **Rich metadata on every chunk.** The ingest scripts stamp
   `file_path`, `folder`, `commit_sha`, `author`, `commit_date`,
   `commit_subject`, `figure_id` where applicable — the webapp passes
   that metadata to the LLM alongside the chunk text, so Claude can
   answer "when was X introduced" correctly rather than only "what is X".
3. **Streaming synthesis + a graph that updates.** Sources arrive from
   `Memory.search()` in ~200 ms; the LLM streams its answer token by
   token into the chat panel; the graph iframe reloads in parallel.
   The experience feels like ~500 ms instead of the 2–4 s a buffered
   LLM response takes.

---

## Running it locally

Prerequisites: the local ApertureDB Docker stack running (`docker
compose up -d` from the repo root), Python 3.10+, and the
`sentence-transformers` and `clip` extras installed:

```bash
pip install ".[sentence-transformers,clip,ui]"
```

Then, from the repo root:

```bash
# One-time ingest — ~2 min total, dominated by embedding model downloads.
python demo/webapp/ingest.py
python demo/webapp/ingest_git.py
python demo/webapp/ingest_images.py

# Start the webapp.
ANTHROPIC_API_KEY=sk-ant-... \
  ANTHROPIC_WORKSPACE_ID=wrkspc_... \
  uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0
```

Open `http://localhost:9001` and try the example chips in the UI. The
synthesised-answer panel is optional — if no `ANTHROPIC_API_KEY` is
set, the app still runs; it just shows a soft notice and skips the
synthesis step.

To use a different LLM, set `NEXUS_DEMO_LLM` (any
`claude-sonnet-5-5` / `claude-haiku-5-5` / `claude-opus-5-5` id).

### Resetting the demo workspace

Each ingest script accepts `--reset` to wipe its own slice of memory
before re-ingesting. For a full reset, bring the Docker stack down with
volumes removed:

```bash
docker compose down -v && docker compose up -d
python demo/webapp/ingest.py
python demo/webapp/ingest_git.py
python demo/webapp/ingest_images.py
```

---

## Architecture

See [`demo/webapp/README.md`](https://github.com/aperture-data/aperture-nexus/blob/main/demo/webapp/README.md)
for the file layout and endpoint contract. The short version:

- **Backend** (`app.py`) is a thin FastAPI wrapper around a single long-lived
  `Memory` instance plus an Anthropic client. Three endpoints:
  - `POST /search` — fast path, returns ranked chunks only, no LLM wait
  - `GET  /synthesize` — SSE stream: emits a `sources` event with the
    chunks first, then `chunk` events with LLM text, then `done`
  - `GET  /graph/query?q=…` — runs the same search, builds a graph
    seeded on matching Blobs (one per file / commit / figure id)
    expanded two hops upward (Commit → Context → Session → User), and
    returns an interactive HTML page
- **Frontend** (`templates/index.html`) is vanilla JS. On every search:
  reloads the graph iframe in parallel with opening the EventSource for
  the stream.
- **Graph viz** (`graph_viz.py`) is a schema-agnostic ApertureDB
  renderer — the same library used in
  [`aperturedb-applications/notebooks/gcs-auto-annotate`](https://github.com/aperture-data/aperturedb-applications/tree/main/notebooks/gcs-auto-annotate).
  It inspects live schema with `GetSchema`, builds a react-force-graph
  page, and inlines each node's hero media (an image, a video, a blob
  preview) directly into the page's details panel.

Diagram retrieval is intentionally not CLIP image search. Each diagram
is logged *twice* — once as a text caption (BGE-indexed, text-searchable)
and once as an image (CLIP-indexed, usable later for image-to-image
queries). When a caption surfaces for a text query, the UI pulls up the
paired image from Nexus and renders it inline. The caption-and-image
sibling pair lives in a dedicated "architecture diagrams" context, so a
graph query also reveals the full set of diagrams Nexus knows about.

---

## How we built it

The whole demo rests on five Nexus API calls. The full source is in
[`demo/webapp/`](https://github.com/aperture-data/aperture-nexus/tree/main/demo/webapp);
the snippets below strip the error-handling and logging so you can see
just the Nexus shape.

### 1. Ingesting a file

For each file we create one `Information` buffer under a `Context`
keyed to the folder, log the chunks, then commit — one commit per file.
`process_and_commit` runs the configured text embedding model
(sentence-transformers in this demo) and writes a `Blob` + `Descriptor`
per chunk, all linked back to the `Context`.

```python
from aperture_nexus import Context, Information, Memory, NexusAdmin

memory = Memory()
memory._cfg.models.text_embedding = "BAAI/bge-small-en-v1.5"
principal = memory.authenticate(user_id="nexus-demo-user", api_key=api_key)

ctx = Context(
    principal=principal,
    session_name="aperture-nexus-codebase",
    purpose="aperture-nexus Python API source",
    organization="nexus-demo",
    department="src",
)

info = Information(context_id=ctx.id)
for idx, chunk in enumerate(_chunk_text(file_text, size=1200, overlap=150)):
    info.log(
        text=chunk,
        metadata={
            "file_path": "src/aperture_nexus/memory.py",
            "folder": "src",
            "chunk_idx": idx,
        },
        tag="src/aperture_nexus/memory.py",
    )

memory.process_and_commit(ctx, info)
```

Everything on that `metadata` dict lands on the stored `Blob` and on
its `Descriptor`, so later searches can filter by `file_path`, `folder`,
or any custom key you add — no re-ingest needed.

### 2. Ingesting a git commit (author becomes a Principal)

The git-history ingest treats each commit's author as a distinct
Principal. The commit subject becomes `Context.purpose`, so each commit
shows up as its own `NexusContext` node in the graph with a readable
label. The commit date flows through `created_at` on every entry
automatically.

```python
# Each author becomes a Principal — provision once, cache the API key.
pid = _principal_name(commit["author_name"], commit["author_email"])
if pid not in principals:
    api_key = admin.create_principal(
        user_id=pid, user_name=commit["author_name"],
        organization="nexus-demo", department="git-history",
    )
    principals[pid] = memory.authenticate(user_id=pid, api_key=api_key)
principal = principals[pid]

# One Context per commit — purpose = commit subject.
ctx = Context(
    principal=principal,
    session_name="aperture-nexus-codebase",   # shared with the file ingest
    purpose=commit["subject"][:200],
    organization="nexus-demo",
    department="git-history",
)

info = Information(context_id=ctx.id)
# Entry 1: the commit message itself
info.log(text=commit["message"], metadata={
    "commit_sha": commit["sha"], "author": commit["author_name"],
    "commit_date": commit["date"], "kind": "commit_message",
})
# Entry 2+: the diff, chunked
for chunk in _chunk_text(commit["diff"]):
    info.log(text=chunk, metadata={
        "commit_sha": commit["sha"], "commit_subject": commit["subject"],
        "author": commit["author_name"], "commit_date": commit["date"],
        "kind": "diff",
    })

memory.process_and_commit(ctx, info)
```

Note the shared `session_name` with the file ingest — a single
`Memory.search` sees both.

### 3. Ingesting a diagram (paired caption + image)

Each architecture diagram is logged twice: a text caption (BGE-indexed,
findable by natural-language questions like "show me the architecture")
and the image bytes themselves (CLIP-indexed, usable for image-to-image
queries). Both carry the same `figure_id` metadata so the UI can pair
them for display.

```python
memory._cfg.models.image_embedding = "ViT-B/16"   # CLIP for the pixels

ctx = Context(
    principal=principal,
    session_name="aperture-nexus-codebase",
    purpose="aperture-nexus architecture diagrams",
    organization="nexus-demo", department="diagrams",
)

info = Information(context_id=ctx.id)
info.log(
    text="Diagram: Nexus write and query workflow. WRITE path: ...",
    metadata={"kind": "diagram_caption", "figure_id": "nexus_workflow"},
    tag="diagram:nexus_workflow",
)
info.log(
    image="docs/nexus_workflow.png",
    metadata={"kind": "diagram_image", "figure_id": "nexus_workflow"},
    tag="diagram:nexus_workflow",
)
memory.process_and_commit(ctx, info)
```

### 4. Searching across all three (files + commits + diagrams)

The webapp's `/search` endpoint is one `Memory.search()` call, filtered
to the demo session. The embedding model is picked up from
`memory._cfg.models.text_embedding` — no need to specify it per call.

```python
results = memory.search(
    query="Who added the department and organization fields to Principal and when?",
    modality="text",
    filters={
        "session_name": "aperture-nexus-codebase",
        "organization":  "nexus-demo",
    },
    k=8,
)

for r in results:
    meta = r.metadata or {}
    kind = meta.get("kind")   # 'diff', 'commit_message', 'diagram_caption', or None (file)
    print(f"{r.score:.3f}  {kind or 'file':16s}  "
          f"{meta.get('file_path') or meta.get('commit_short') or meta.get('figure_id')}")
    print(f"  {r.text[:120]}...")
```

Returns a flat list ranked by cosine similarity. Each `SearchResult`
carries the chunk's text, score, `context_id`, `session_id`, `user_id`,
`created_at`, and any custom metadata you stamped at log time —
everything the LLM needs to cite its sources.

### 5. Feeding the LLM

The webapp's `/synthesize` endpoint hands the top-k chunks to Claude
with a tight system prompt. Streaming the response via SSE drops
perceived latency from ~3 s to ~500 ms:

```python
with anthropic_client.messages.stream(
    model="claude-sonnet-5-5",
    max_tokens=1000,
    system="You are an expert on aperture-nexus… [full prompt in app.py] …",
    messages=[{"role": "user", "content": prompt_with_chunks}],
) as stream:
    for text in stream.text_stream:
        yield f"event: chunk\ndata: {json.dumps({'text': text})}\n\n"
```

That's the whole stack — ~250 lines of application code around the
Nexus primitives.

---

## Try it on your own repo

The three ingest scripts take their paths from `REPO_ROOT` at the top of
each script — pointing them at a different checkout works today without
code changes. A hosted version where visitors paste a public GitHub URL
and chat with that repo is on the roadmap; the open design question is
how to isolate per-visitor workspaces inside a shared ApertureDB without
duplicating the DB per user.
