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
import subprocess
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

# Optional: LLM backends for the synthesised-answer panel. Falls back
# cleanly to raw-results-only mode if neither SDK is installed or no key
# is set. Two providers supported:
#   - Anthropic (default)         : ANTHROPIC_API_KEY (+ optional
#     ANTHROPIC_WORKSPACE_ID for user-scoped keys)
#   - OpenAI-compatible endpoints : OPENAI_API_KEY (+ optional
#     OPENAI_BASE_URL for Together / Groq / Fireworks / LocalAI)
try:
    import anthropic  # noqa: E402
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False
try:
    import openai  # noqa: E402
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

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
# Default search scope is the whole workspace (every session). The UI
# exposes session + principal + department as optional filter chips.
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
API_KEY_FILE = WEBAPP_DIR / ".nexus-demo.key"

# NEXUS_DEMO_LLM_PROVIDER: "anthropic" (default) or "openai". Auto-switched
# to "openai" when only OPENAI_API_KEY is set and ANTHROPIC_API_KEY is not.
LLM_PROVIDER = os.environ.get("NEXUS_DEMO_LLM_PROVIDER", "").strip().lower()
LLM_MODEL = os.environ.get(
    "NEXUS_DEMO_LLM",
    "claude-sonnet-5-5" if LLM_PROVIDER != "openai" else "gpt-4o-mini",
)
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
LLM_ACTIVE_PROVIDER: Optional[str] = None


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
    """Pick an LLM client based on which env vars are set.

    Resolution order:
      - explicit NEXUS_DEMO_LLM_PROVIDER=openai  → OpenAI only
      - explicit NEXUS_DEMO_LLM_PROVIDER=anthropic → Anthropic only
      - default: Anthropic if ANTHROPIC_API_KEY set, else OpenAI if
        OPENAI_API_KEY set, else None (fallback mode)

    Returns a tuple (provider, client) or (None, None).
    """
    want = LLM_PROVIDER
    have_anthropic = _ANTHROPIC_OK and bool(os.environ.get("ANTHROPIC_API_KEY"))
    have_openai    = _OPENAI_OK and bool(os.environ.get("OPENAI_API_KEY"))

    if want == "openai" and have_openai:
        return "openai", _openai_client()
    if want == "anthropic" and have_anthropic:
        return "anthropic", _anthropic_client()
    if want and (want == "openai" or want == "anthropic"):
        LOG.info("NEXUS_DEMO_LLM_PROVIDER=%r but key or SDK missing — summary disabled", want)
        return None, None

    # No explicit provider — pick what's available.
    if have_anthropic:
        return "anthropic", _anthropic_client()
    if have_openai:
        return "openai", _openai_client()
    LOG.info("no LLM key set (ANTHROPIC_API_KEY / OPENAI_API_KEY) — summary disabled")
    return None, None


def _anthropic_client():
    try:
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        # Exclude zstd from Accept-Encoding: anthropic 1.4.0 ships with httpx2,
        # whose zstd decoder calls zstandard.process() with an unsupported
        # output_buffer_limit kwarg and raises TypeError that surfaces as a
        # misleading "Connection error". gzip/deflate paths are fine.
        default_headers: dict = {"accept-encoding": "gzip, deflate"}
        if workspace:
            default_headers["anthropic-workspace-id"] = workspace
            LOG.info("using Anthropic workspace id %s", workspace)
        c = anthropic.Anthropic(default_headers=default_headers)
        LOG.info("LLM ready — synthesising answers via Anthropic %s", LLM_MODEL)
        return c
    except Exception as e:
        LOG.warning("could not initialise anthropic client: %s", e)
        return None


def _openai_client():
    try:
        base_url = os.environ.get("OPENAI_BASE_URL")  # for Together, Groq, LocalAI, etc.
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
            LOG.info("using OpenAI-compatible endpoint %s", base_url)
        c = openai.OpenAI(**kwargs)
        LOG.info("LLM ready — synthesising answers via OpenAI %s", LLM_MODEL)
        return c
    except Exception as e:
        LOG.warning("could not initialise openai client: %s", e)
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
    global MEMORY, LLM_CLIENT, LLM_ACTIVE_PROVIDER
    MEMORY = _load_memory()
    LLM_ACTIVE_PROVIDER, LLM_CLIENT = _load_llm()
    LOG.info("Memory ready — authenticated as %r", DEMO_USER)


class SearchRequest(BaseModel):
    query: str
    k: int = 6
    session_name: Optional[str] = None
    user_id: Optional[str] = None
    department: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEBAPP_DIR / "templates" / "index.html").read_text())


def _build_filters(
    session_name: Optional[str] = None,
    user_id: Optional[str] = None,
    department: Optional[str] = None,
) -> dict:
    """Build the Memory.search filters dict, scoped to the workspace."""
    f: dict = {"organization": DEMO_WORKSPACE}
    if session_name:
        f["session_name"] = session_name
    if user_id:
        f["user_id"] = user_id
    if department:
        f["department"] = department
    return f


def _run_search(
    query: str, k: int,
    session_name: Optional[str] = None,
    user_id: Optional[str] = None,
    department: Optional[str] = None,
) -> list[dict]:
    """Run Memory.search and format results for JSON transport.

    Retries once if the shared ApertureDB connection has gone stale.
    """
    assert MEMORY is not None
    filters = _build_filters(session_name, user_id, department)
    try:
        results = MEMORY.search(query=query, modality="text", filters=filters, k=k)
    except Exception as e:
        if not _is_connection_error(e):
            raise
        LOG.warning("search hit stale connection, reconnecting: %s", e)
        _reconnect_memory()
        results = MEMORY.search(query=query, modality="text", filters=filters, k=k)
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
    return JSONResponse({"results": _run_search(
        req.query, req.k,
        session_name=req.session_name,
        user_id=req.user_id,
        department=req.department,
    )})


@app.get("/synthesize")
def synthesize(
    q: str = Query(..., min_length=1),
    k: int = 6,
    session_name: Optional[str] = None,
    user_id: Optional[str] = None,
    department: Optional[str] = None,
) -> StreamingResponse:
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
            results = _run_search(
                q, k,
                session_name=session_name, user_id=user_id, department=department,
            )
            yield _event("sources", {"results": results})
            if LLM_CLIENT is None or not results:
                yield _event("done", {"model": None})
                return
            prompt = _build_user_prompt(q, results)
            if LLM_ACTIVE_PROVIDER == "anthropic":
                with LLM_CLIENT.messages.stream(
                    model=LLM_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                    system=LLM_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        yield _event("chunk", {"text": text})
            else:  # openai (or openai-compatible endpoint)
                stream = LLM_CLIENT.chat.completions.create(
                    model=LLM_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    stream=True,
                )
                for ev in stream:
                    delta = (ev.choices[0].delta.content or "") if ev.choices else ""
                    if delta:
                        yield _event("chunk", {"text": delta})
            yield _event("done", {"model": f"{LLM_ACTIVE_PROVIDER}:{LLM_MODEL}"})
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
def graph_query(
    q: str = Query(..., min_length=1),
    k: int = 8,
    session_name: Optional[str] = None,
    user_id: Optional[str] = None,
    department: Optional[str] = None,
) -> HTMLResponse:
    """Render a graph where the search results are highlighted.

    Anchors the graph on the matched Blob chunks (via file_path from
    metadata), then expands one hop upward (NexusCommit → NexusContext →
    NexusSession → NexusUser).
    """
    assert MEMORY is not None
    filters = _build_filters(session_name, user_id, department)
    results = MEMORY.search(query=q, modality="text", filters=filters, k=k)
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
        title=f"Results for: {q[:60] + ('…' if len(q) > 60 else '')}",
        filename=str(out),
        client=MEMORY._db,
        node_colors=NEXUS_NODE_COLORS,
        highlight_ids=[n["id"] for n in seed_nodes],
        embed=False,
    )
    return HTMLResponse(out.read_text())


# ----------------------------------------------------------------------
# Counts endpoint — powers the live-count filter chips
# ----------------------------------------------------------------------

@app.get("/counts")
def counts() -> JSONResponse:
    """Return the catalog of sessions, principals, and departments
    present in the demo workspace, each with its Blob entry count.
    The UI uses this to populate the filter chip strip."""
    assert MEMORY is not None
    db = MEMORY._db

    def _bucket(prop: str) -> dict:
        """Count _Blob entries grouped by one property."""
        resp, _ = db.query([{"FindBlob": {
            "constraints": {"organization": ["==", DEMO_WORKSPACE]},
            "blobs": False,
            "results": {"list": [prop]},
            "limit": 10000,
        }}])
        if not isinstance(resp, list):
            return {}
        entities = resp[0].get("FindBlob", {}).get("entities", []) or []
        out: dict[str, int] = {}
        for e in entities:
            v = e.get(prop)
            if v is None or v == "":
                continue
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    try:
        return JSONResponse({
            "sessions":    _bucket("session_name"),
            "principals":  _bucket("user_id"),
            "departments": _bucket("department"),
            "total_blobs": sum(_bucket("session_name").values()),
        })
    except Exception as e:
        if _is_connection_error(e):
            _reconnect_memory()
            return counts()
        raise


# ----------------------------------------------------------------------
# Lineage — walk the graph from a file to every PR that touched it
# ----------------------------------------------------------------------

@app.get("/graph/lineage", response_class=HTMLResponse)
def graph_lineage(file: str = Query(..., min_length=1)) -> HTMLResponse:
    """Render a graph showing every PR / commit / principal that ever
    touched a given file. The Nexus-only moment: not just "what does
    this file say" (RAG) but "how did it get this way" (graph lineage).
    """
    assert MEMORY is not None
    db = MEMORY._db

    # Find every _Blob entry whose metadata links back to this file.
    # Files appear via file_path (file ingest); git chunks that modified
    # this file appear in their diff text, but we match structurally:
    # a git chunk doesn't carry file_path, so for history we instead
    # walk from NexusCommits whose git diff mentions the file.
    #
    # Simpler, graph-native path: find Blobs with file_path==file (file
    # ingest), AND Blobs that are git diff chunks whose session is a
    # PR that touched the file. For the demo we use the straightforward
    # structural query: all Blobs tagged with this file_path, plus all
    # git-history Blobs in the same workspace whose diff text mentions
    # the file's basename.

    base = os.path.basename(file)

    # Which git commits touched this file? Ask git locally — ApertureDB
    # can't do substring queries, and the diff text in Nexus isn't
    # directly file-indexed. git log --follow handles renames.
    try:
        git_output = subprocess.check_output(
            ["git", "log", "--follow", "--format=%H", "--", file],
            cwd=str(REPO_ROOT), text=True, errors="replace",
            stderr=subprocess.DEVNULL,
        )
        commit_shas = [s for s in git_output.strip().split("\n") if s]
    except Exception as e:
        LOG.warning("git log for %s failed: %s", file, e)
        commit_shas = []

    anchor_nodes = []
    # Anchor 1: one Blob representing the file (from file-ingest chunks).
    file_blobs = graph_viz.find_nodes(
        db, "_Blob",
        constraints={"file_path": ["==", file]},
        limit=1,
        label_props=NEXUS_LABEL_PROPS,
    )
    for n in file_blobs:
        n["label"] = f"{base} (current)"
        n["props"] = {"source": "file", "file_path": file}
        anchor_nodes.append(n)

    # Anchor 2: one Blob per matching git commit (the commit_message chunk,
    # which is one per commit — avoids pulling in all N diff chunks).
    if commit_shas:
        git_blobs = graph_viz.find_nodes(
            db, "_Blob",
            constraints={
                "commit_sha": ["in", commit_shas],
                "kind": ["==", "commit_message"],
            },
            limit=200,
            label_props=NEXUS_LABEL_PROPS,
        )
        seen_shas: set[str] = set()
        for n in git_blobs:
            sha = (n.get("props") or {}).get("commit_sha")
            if sha and sha not in seen_shas:
                seen_shas.add(sha)
                p = n.get("props") or {}
                subject = (p.get("commit_subject") or "")[:40]
                n["label"] = f"{sha[:8]} · {subject}"
                n["props"] = {
                    "source": "git_commit",
                    "commit_sha": sha,
                    "commit_subject": p.get("commit_subject", ""),
                    "author": p.get("author", ""),
                    "commit_date": p.get("commit_date", ""),
                }
                anchor_nodes.append(n)

    if not anchor_nodes:
        return HTMLResponse(
            "<p style='color:#eaeaea;font-family:sans-serif;padding:20px'>"
            f"No Nexus entries found for file <b>{base}</b>.</p>"
        )

    # Expand UP: Commit → Context (= PR) → Session → User. No Blob/Image
    # fan-out — keeps the graph focused on the PR lineage.
    nodes, edges = graph_viz.expand(
        db, anchor_nodes, depth=2, per_class=4,
        exclude={"_Blob", "_Image", "_Descriptor", "_DescriptorSet"},
        label_props=NEXUS_LABEL_PROPS,
        max_nodes=200,
    )
    # Add anchor nodes back (expand excludes _Blob, which includes our
    # seeds — put them in the final node set so edges render).
    nodes = list(anchor_nodes) + nodes

    # Rename ugly UUID-y labels to the PR title (purpose) for Contexts
    # and the short SHA for Commits.
    for n in nodes:
        p = n.get("props") or {}
        if n["cls"] == "NexusContext" and p.get("purpose"):
            n["label"] = p["purpose"][:60]
        elif n["cls"] == "NexusSession" and p.get("session_name"):
            n["label"] = p["session_name"][:60]
        elif n["cls"] == "NexusCommit":
            cid = p.get("commit_id", "")
            n["label"] = f"commit {cid[:8]}" if cid else n["label"]

    out = GRAPHS_DIR / f"lineage-{uuid.uuid4().hex[:8]}.html"
    graph_viz.render_graph(
        nodes, edges,
        title=f"Lineage of {base}",
        filename=str(out),
        client=db,
        node_colors=NEXUS_NODE_COLORS,
        highlight_ids=[n["id"] for n in anchor_nodes],
        embed=False,
    )
    return HTMLResponse(out.read_text())


# Serve the generated graph HTML files directly so iframe links work.
app.mount("/graphs", StaticFiles(directory=str(GRAPHS_DIR)), name="graphs")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
