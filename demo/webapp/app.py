"""
app.py — Nexus demo web app.

A single-page UI that lets you ask questions about the aperture-nexus
codebase and see:
  * the top semantically-matching chunks (file, snippet, score)
  * an interactive graph of the Nexus memory with those results highlighted

Run (from aperture-nexus/):
    python demo/webapp/ingest.py        # one-time: load the repo
    uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0
Then open http://localhost:9001
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# Connect to the local Docker ApertureDB stack by default.
# Override by exporting APERTUREDB_JSON before launching uvicorn.
os.environ.setdefault(
    "APERTUREDB_JSON",
    '{"host":"localhost","port":55556,"username":"admin","password":"admin","use_ssl":false}',
)

import json  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from aperture_nexus import Memory  # noqa: E402

# Optional: Anthropic SDK for LLM-synthesised answers. Falls back cleanly
# to raw-results-only mode if the SDK or API key is missing.
try:
    import anthropic  # noqa: E402
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parents[1]
GRAPHS_DIR = WEBAPP_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WEBAPP_DIR))
import graph_viz  # noqa: E402  — bundled copy

LOG = logging.getLogger("nexus-demo.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

DEMO_WORKSPACE = "nexus-demo"
DEMO_USER = "nexus-demo-user"
DEMO_SESSION = "aperture-nexus-codebase"
# Must match the model the ingest scripts used.
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
API_KEY_FILE = WEBAPP_DIR / ".nexus-demo.key"

LLM_MODEL = os.environ.get("NEXUS_DEMO_LLM", "claude-sonnet-5-5")
LLM_MAX_TOKENS = 1000
LLM_SYSTEM = (
    "You are an expert on aperture-nexus, a Python memory layer for AI agents built on "
    "ApertureDB. Below are chunks retrieved from Nexus itself by semantic search over the "
    "user's question. Each chunk is prefixed with its source — a FILE path, a GIT COMMIT "
    "header, or a DIAGRAM caption. Use the commit metadata when the question is about "
    "when something was introduced, who changed it, or why. Cite your sources in "
    "parentheses after each claim: file paths, short commit SHAs, or figure ids.\n"
    "\n"
    "DIAGRAM chunks are special. The web UI is already displaying the actual PNG image "
    "for each diagram in the SOURCES panel below your answer — the user can see the "
    "picture. Your job is to explain what the diagram shows and answer the question "
    "using the caption as source. Do NOT redraw the diagram with ASCII art, do NOT say "
    "'I can't show the image' or 'I only have the description', and do NOT apologise "
    "for not having pixels. Just reference the diagram by name (e.g. \"see the "
    "nexus_components diagram below\") and let the UI render it.\n"
    "\n"
    "If the chunks don't address the question, say so plainly rather than inventing "
    "an answer."
)

# Nexus brand palette for the graph nodes.
NEXUS_NODE_COLORS = {
    "NexusUser":       "#102069",
    "NexusSession":    "#172F94",
    "NexusContext":    "#4B5EAC",
    "NexusCommit":     "#828FC5",
    "_Blob":           "#1F4E8C",
    "_Image":          "#0C7A5A",   # diagram images stand out in the graph
    "_Descriptor":     "#8EC1F0",
    "_DescriptorSet":  "#C9560C",
}
NEXUS_LABEL_PROPS = {
    "NexusUser": "user_id",
    "NexusSession": "session_name",
    "NexusContext": "purpose",
    "NexusCommit": "commit_id",
    "_Blob": "file_path",
    "_Image": "figure_id",
}


# ----------------------------------------------------------------------
# Nexus setup — authenticate once at startup
# ----------------------------------------------------------------------

def _load_memory() -> Memory:
    if not API_KEY_FILE.exists():
        raise RuntimeError(
            f"No cached API key at {API_KEY_FILE}. "
            "Run: python demo/webapp/ingest.py first."
        )
    key = API_KEY_FILE.read_text().strip()
    m = Memory()
    m._cfg.models.text_embedding = TEXT_MODEL
    m.authenticate(user_id=DEMO_USER, api_key=key)
    return m


MEMORY: Optional[Memory] = None
LLM_CLIENT = None  # type: ignore[var-annotated]


def _is_connection_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in (
        "bad file descriptor", "connection", "closed", "broken pipe",
        "could not query", "tcp",
    ))


def _reconnect_memory() -> None:
    """Rebuild the shared Memory instance after a stale-connection error."""
    global MEMORY
    LOG.info("rebuilding Memory after connection error")
    MEMORY = _load_memory()


def _load_llm():
    """Return an Anthropic client if the SDK + key are available, else None.

    If ``ANTHROPIC_WORKSPACE_ID`` is set, it's passed as the
    ``anthropic-workspace-id`` default header — needed for user-scoped
    (``sk-ant-usr-...``) keys. Workspace-scoped (``sk-ant-wrkspc_...``)
    keys don't need this.
    """
    if not _ANTHROPIC_OK:
        LOG.info("anthropic SDK not installed — summary disabled")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        LOG.info("ANTHROPIC_API_KEY not set — summary disabled")
        return None
    try:
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        # Exclude zstd from Accept-Encoding: anthropic 1.4.0 ships with httpx2,
        # whose zstd decoder calls zstandard.process() with an unsupported
        # output_buffer_limit kwarg and raises TypeError that surfaces as a
        # misleading "Connection error". gzip/deflate paths are fine.
        default_headers: dict = {"accept-encoding": "gzip, deflate"}
        if workspace:
            default_headers["anthropic-workspace-id"] = workspace
            LOG.info("using workspace id %s", workspace)
        c = anthropic.Anthropic(default_headers=default_headers)
        LOG.info("LLM ready — synthesising answers with %s", LLM_MODEL)
        return c
    except Exception as e:
        LOG.warning("could not initialise anthropic client: %s", e)
        return None


def _build_user_prompt(query: str, results: list[dict]) -> str:
    """Build the LLM user-message content from retrieved chunks."""
    chunks = []
    for i, r in enumerate(results, 1):
        if r.get("source_kind") == "git":
            header = (
                f"[{i}] GIT COMMIT  {r.get('commit_short','')}  "
                f"by {r.get('author','?')}  on {r.get('commit_date','?')[:10]}  "
                f"({r.get('kind','?')})\n"
                f"    Subject: {r.get('commit_subject','?')}"
            )
        else:
            header = f"[{i}] FILE  {r['file_path']}  (score {r['score']:.3f})"
        chunks.append(f"{header}\n{r['text']}")
    return f"Question:\n{query}\n\nChunks:\n" + "\n\n".join(chunks)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

app = FastAPI(title="Nexus Demo")


@app.on_event("startup")
def _startup() -> None:
    global MEMORY, LLM_CLIENT
    MEMORY = _load_memory()
    LLM_CLIENT = _load_llm()
    LOG.info("Memory ready — authenticated as %r", DEMO_USER)


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEBAPP_DIR / "templates" / "index.html").read_text())


def _run_search(query: str, k: int) -> list[dict]:
    """Run Memory.search and format results for JSON transport.

    Retries once if the shared ApertureDB connection has gone stale.
    """
    assert MEMORY is not None
    try:
        results = MEMORY.search(
            query=query, modality="text",
            filters={"session_name": DEMO_SESSION, "organization": DEMO_WORKSPACE},
            k=k,
        )
    except Exception as e:
        if not _is_connection_error(e):
            raise
        LOG.warning("search hit stale connection, reconnecting: %s", e)
        _reconnect_memory()
        results = MEMORY.search(
            query=query, modality="text",
            filters={"session_name": DEMO_SESSION, "organization": DEMO_WORKSPACE},
            k=k,
        )
    payload = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        kind = meta.get("kind")
        if kind in ("commit_message", "diff"):
            source_kind = "git"
        elif kind == "diagram_caption":
            source_kind = "diagram"
        else:
            source_kind = "file"
        payload.append({
            "score":        float(getattr(r, "score", 0.0) or 0.0),
            "text":         (getattr(r, "text", None) or "")[:1200],
            "source_kind":  source_kind,
            "file_path":    meta.get("file_path") or "",
            "folder":       meta.get("folder", ""),
            "commit_short":   meta.get("commit_short"),
            "commit_subject": meta.get("commit_subject"),
            "author":         meta.get("author"),
            "commit_date":    meta.get("commit_date"),
            "kind":           kind,
            "context_id": getattr(r, "context_id", None),
            # diagram fields (populated when source_kind == "diagram")
            "figure_id":      meta.get("figure_id"),
            "image_url":      f"/figure/{meta['figure_id']}" if meta.get("figure_id") else None,
        })
    return payload


# ----------------------------------------------------------------------
# Diagram image serving — serves known PNGs by figure_id
# ----------------------------------------------------------------------

FIGURE_FILES: dict[str, Path] = {
    "nexus_components":  REPO_ROOT / "docs" / "nexus_components.png",
    "nexus_workflow":    REPO_ROOT / "docs" / "nexus_workflow.png",
    "cognition_stack":   REPO_ROOT / "docs" / "blog_image_cognition_stack_v2.png",
    "nexus_schema":      REPO_ROOT / "demo" / "nexus_schema.png",
}


@app.get("/figure/{figure_id}")
def figure(figure_id: str) -> FileResponse:
    path = FIGURE_FILES.get(figure_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"unknown figure id {figure_id!r}")
    return FileResponse(path, media_type="image/png")


@app.post("/search")
def search(req: SearchRequest) -> JSONResponse:
    """Fast path — just the retrieved chunks, no LLM call."""
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    return JSONResponse({"results": _run_search(req.query, req.k)})


@app.get("/synthesize")
def synthesize(q: str = Query(..., min_length=1), k: int = 6) -> StreamingResponse:
    """SSE stream — sources first, then the LLM answer token-by-token.

    Events emitted:
      sources   JSON {"results": [...]}
      chunk     JSON {"text": "..."}  (repeated)
      done      JSON {"model": "..."}
      error     JSON {"message": "..."}
    """
    def _event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    def generate():
        try:
            results = _run_search(q, k)
            yield _event("sources", {"results": results})
            if LLM_CLIENT is None or not results:
                yield _event("done", {"model": None})
                return
            prompt = _build_user_prompt(q, results)
            with LLM_CLIENT.messages.stream(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=LLM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield _event("chunk", {"text": text})
            yield _event("done", {"model": LLM_MODEL})
        except Exception as e:
            LOG.warning("synthesize failed: %s", e)
            yield _event("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/graph/schema", response_class=HTMLResponse)
def graph_schema() -> HTMLResponse:
    """Render the full schema graph of the nexus-demo workspace."""
    assert MEMORY is not None
    out = GRAPHS_DIR / "schema.html"
    try:
        graph_viz.render_schema(
            MEMORY._db, title="Nexus memory — schema",
            filename=str(out), node_colors=NEXUS_NODE_COLORS, embed=False,
        )
    except Exception as e:
        if not _is_connection_error(e):
            raise
        LOG.warning("graph/schema hit stale connection, reconnecting: %s", e)
        _reconnect_memory()
        graph_viz.render_schema(
            MEMORY._db, title="Nexus memory — schema",
            filename=str(out), node_colors=NEXUS_NODE_COLORS, embed=False,
        )
    return HTMLResponse(out.read_text())


@app.get("/graph/query", response_class=HTMLResponse)
def graph_query(q: str = Query(..., min_length=1), k: int = 8) -> HTMLResponse:
    """Render a graph where the search results are highlighted.

    Anchors the graph on the matched Blob chunks (via file_path from
    metadata), then expands one hop upward (NexusCommit → NexusContext →
    NexusSession → NexusUser).
    """
    assert MEMORY is not None
    results = MEMORY.search(
        query=q, modality="text",
        filters={"session_name": DEMO_SESSION, "organization": DEMO_WORKSPACE},
        k=k,
    )
    if not results:
        return HTMLResponse(
            "<p style='color:#eaeaea;font-family:sans-serif;padding:20px'>"
            f"No results for <b>{q}</b>.</p>"
        )
    # Count chunk hits per source: files keyed by file_path, git keyed by
    # commit_sha, diagrams keyed by figure_id. Each becomes one node
    # (deduped) in the graph.
    from collections import Counter
    file_counts: Counter[str] = Counter()
    git_counts: Counter[str] = Counter()
    diagram_counts: Counter[str] = Counter()
    git_meta: dict[str, dict] = {}
    for r in results:
        meta = r.metadata or {}
        kind = meta.get("kind")
        if kind in ("commit_message", "diff"):
            sha = meta.get("commit_sha")
            if sha:
                git_counts[sha] += 1
                git_meta[sha] = {
                    "commit_short": meta.get("commit_short", sha[:8]),
                    "commit_subject": meta.get("commit_subject", ""),
                    "author": meta.get("author", ""),
                    "commit_date": meta.get("commit_date", ""),
                }
        elif kind == "diagram_caption":
            fig = meta.get("figure_id")
            if fig:
                diagram_counts[fig] += 1
        else:
            fp = meta.get("file_path")
            if fp:
                file_counts[fp] += 1

    if not file_counts and not git_counts and not diagram_counts:
        return HTMLResponse(
            "<p style='color:#eaeaea;font-family:sans-serif;padding:20px'>"
            "Results had no anchor metadata (file_path, commit_sha, or figure_id).</p>"
        )

    seed_nodes = []
    seen_ids: set[str] = set()

    # One Blob per matched file.
    if file_counts:
        all_file_blobs = graph_viz.find_nodes(
            MEMORY._db, "_Blob",
            constraints={"file_path": ["in", list(file_counts.keys())]},
            limit=500,
            label_props=NEXUS_LABEL_PROPS,
        )
        seen_files: set[str] = set()
        for n in all_file_blobs:
            fp = (n.get("props") or {}).get("file_path")
            if fp and fp not in seen_files:
                seen_files.add(fp)
                count = file_counts[fp]
                base = os.path.basename(fp)
                n["label"] = f"{base} ({count})" if count > 1 else base
                n["props"] = {
                    "source": "file",
                    "file_path": fp,
                    "folder": (n.get("props") or {}).get("folder", ""),
                    "matched_chunks": count,
                    "chunks_in_file": (n.get("props") or {}).get("chunk_total", 1),
                }
                seed_nodes.append(n)
                seen_ids.add(n["id"])

    # One Blob per matched diagram (the caption chunk).
    if diagram_counts:
        all_diagram_blobs = graph_viz.find_nodes(
            MEMORY._db, "_Blob",
            constraints={"figure_id": ["in", list(diagram_counts.keys())]},
            limit=200,
            label_props=NEXUS_LABEL_PROPS,
        )
        seen_figs: set[str] = set()
        for n in all_diagram_blobs:
            fig = (n.get("props") or {}).get("figure_id")
            if fig and fig not in seen_figs and n["id"] not in seen_ids:
                seen_figs.add(fig)
                count = diagram_counts[fig]
                n["label"] = f"diagram · {fig}" + (f" ({count})" if count > 1 else "")
                n["props"] = {
                    "source": "diagram",
                    "figure_id": fig,
                    "image_path": (n.get("props") or {}).get("image_rel_path", ""),
                    "matched_chunks": count,
                }
                seed_nodes.append(n)
                seen_ids.add(n["id"])

    # One Blob per matched commit.
    if git_counts:
        all_git_blobs = graph_viz.find_nodes(
            MEMORY._db, "_Blob",
            constraints={"commit_sha": ["in", list(git_counts.keys())]},
            limit=1000,
            label_props=NEXUS_LABEL_PROPS,
        )
        seen_commits: set[str] = set()
        for n in all_git_blobs:
            sha = (n.get("props") or {}).get("commit_sha")
            if sha and sha not in seen_commits and n["id"] not in seen_ids:
                seen_commits.add(sha)
                g = git_meta.get(sha, {})
                count = git_counts[sha]
                subject = (g.get("commit_subject") or "")[:40]
                n["label"] = f"{g.get('commit_short', sha[:8])} · {subject}" + (f" ({count})" if count > 1 else "")
                n["props"] = {
                    "source": "git",
                    "commit_sha": sha,
                    "commit_short": g.get("commit_short", sha[:8]),
                    "commit_subject": g.get("commit_subject", ""),
                    "author": g.get("author", ""),
                    "commit_date": g.get("commit_date", ""),
                    "matched_chunks": count,
                }
                seed_nodes.append(n)
                seen_ids.add(n["id"])
    # Expand upward only — exclude every ApertureDB native type below Blob
    # AND Blob itself, so expand doesn't pull sibling chunks back in through
    # the shared NexusCommit.
    nodes, edges = graph_viz.expand(
        MEMORY._db, seed_nodes, depth=2, per_class=6,
        exclude={"_Blob", "_Descriptor", "_DescriptorSet"},
        label_props=NEXUS_LABEL_PROPS,
        max_nodes=120,
    )
    # Expand also fans out from a shared Context down to its OTHER Commits,
    # so a Context with 10 commits adds 6 unrelated commits. Keep only
    # Commits directly connected to a seed Blob.
    seed_ids = {n["id"] for n in seed_nodes}
    allowed_commits = {
        s for s, d, *_ in edges
        if (s in seed_ids and any(n["id"] == d and n["cls"] == "NexusCommit" for n in nodes))
    } | {
        d for s, d, *_ in edges
        if (d in seed_ids and any(n["id"] == s and n["cls"] == "NexusCommit" for n in nodes))
    }
    kept = []
    for n in nodes:
        if n["cls"] == "NexusCommit" and n["id"] not in allowed_commits:
            continue
        kept.append(n)
    kept_ids = {n["id"] for n in kept}
    edges = [e for e in edges if e[0] in kept_ids and e[1] in kept_ids]
    nodes = kept
    # Shorten commit labels — UUIDs are visual noise. Keep first 8 chars.
    for n in nodes:
        if n["cls"] == "NexusCommit":
            n["label"] = f"commit {n['label'][:8]}" if len(n["label"]) > 8 else n["label"]
    out = GRAPHS_DIR / f"query-{uuid.uuid4().hex[:8]}.html"
    graph_viz.render_graph(
        nodes, edges,
        title=f"Nexus memory — results for '{q}'",
        filename=str(out),
        client=MEMORY._db,
        node_colors=NEXUS_NODE_COLORS,
        highlight_ids=[n["id"] for n in seed_nodes],
        embed=False,
    )
    return HTMLResponse(out.read_text())


# Serve the generated graph HTML files directly so iframe links work.
app.mount("/graphs", StaticFiles(directory=str(GRAPHS_DIR)), name="graphs")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
