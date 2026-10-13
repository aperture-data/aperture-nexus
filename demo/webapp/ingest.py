"""
ingest.py — load the aperture-nexus repo into Nexus itself.

Walks aperture-nexus/, picks a focused set of Python source files + docs,
chunks each file into ~1200-char windows so every chunk fits inline on
its Descriptor (threshold is 2000 chars in Nexus today), and commits each
file as one Nexus commit with CLIP text embeddings so semantic search
returns readable snippets.

Idempotent: skips when the demo session already has commits.

Usage (from aperture-nexus/):
    python demo/webapp/ingest.py              # ingest if empty
    python demo/webapp/ingest.py --reset      # wipe + re-ingest
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# Connect to the local Docker ApertureDB stack this webapp assumes.
# Set explicitly so we don't rely on home-dir configs.
os.environ.setdefault(
    "APERTUREDB_JSON",
    '{"host":"localhost","port":55556,"username":"admin","password":"admin","use_ssl":false}',
)

from aperture_nexus import Context, Information, Memory, NexusAdmin  # noqa: E402

LOG = logging.getLogger("nexus-demo.ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]

INGEST_PATHS: dict[str, list[str]] = {
    # CLAUDE.md is the Claude Code dev guide — dense with every Nexus concept
    # in one file, so it matches almost every query and crowds out the
    # user-facing docs. Intentionally omitted.
    "root": ["README.md"],
    "src": [
        "src/aperture_nexus/memory.py",
        "src/aperture_nexus/context.py",
        "src/aperture_nexus/information.py",
        "src/aperture_nexus/admin.py",
        "src/aperture_nexus/auth.py",
        "src/aperture_nexus/config.py",
        "src/aperture_nexus/cli.py",
        "src/aperture_nexus/exceptions.py",
        "src/aperture_nexus/_client.py",
    ],
    "docs": [
        "docs/README.md",
        "docs/concepts.md",
        "docs/getting-started.md",
        "docs/api-reference.md",
        "docs/configuration.md",
        "docs/coding-assistant.md",
        "docs/customer-support-agent.md",
    ],
}

DEMO_WORKSPACE = "nexus-demo"
DEMO_USER = "nexus-demo-user"
DEMO_SESSION = "aperture-nexus-codebase"
# BGE-small is a purpose-built text retrieval model (384-dim, ~130MB,
# CPU-fast) that handles specific-attribution queries much better than
# CLIP's text tower. See nexus_clip_text_search_limits memory for why.
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
API_KEY_FILE = REPO_ROOT / "demo" / "webapp" / ".nexus-demo.key"

# Chunk to ~1200 chars with ~150 overlap. Below Nexus's 2000-char
# inline-text threshold, so every chunk's text lands on its Descriptor
# and search returns readable snippets without extra fetches.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows, preferring line boundaries."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        # Try to end on a newline near the window boundary (keeps code chunks readable).
        if end < n:
            nl = text.rfind("\n", i + size // 2, end)
            if nl != -1:
                end = nl + 1
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def _ensure_principal(admin: NexusAdmin) -> str:
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()
        if key:
            LOG.info("reusing principal %r (key cached)", DEMO_USER)
            return key
    try:
        key = admin.create_principal(
            user_id=DEMO_USER,
            user_name="Nexus Demo",
            organization=DEMO_WORKSPACE,
            department="demo",
        )
        LOG.info("created principal %r", DEMO_USER)
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        key = admin.rotate_key(user_id=DEMO_USER)
        LOG.info("rotated key for existing principal %r", DEMO_USER)
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_FILE.write_text(key)
    return key


def _already_ingested(memory: Memory) -> bool:
    try:
        results = memory.search(
            filters={"session_name": DEMO_SESSION, "organization": DEMO_WORKSPACE},
            k=1,
        )
        return len(results) > 0
    except Exception:
        return False


def _reset(memory: Memory) -> None:
    """Remove all demo-session memories so we can re-ingest cleanly."""
    LOG.info("resetting demo session %r…", DEMO_SESSION)
    # Query by metadata to collect session ids actually in use.
    probes = memory.search(
        filters={"session_name": DEMO_SESSION, "organization": DEMO_WORKSPACE},
        k=500,
    )
    sids = {p.session_id for p in probes if p.session_id}
    for sid in sids:
        memory.remove(session_id=sid)
    LOG.info("  removed %d session(s)", len(sids))


def _iter_files() -> Iterable[tuple[str, Path]]:
    for folder, rel_paths in INGEST_PATHS.items():
        for rel in rel_paths:
            p = REPO_ROOT / rel
            if p.exists() and p.is_file():
                yield folder, p
            else:
                LOG.warning("missing (skipped): %s", rel)


def _folder_purpose(folder: str) -> str:
    return {
        "root": "aperture-nexus project overview and dev guide",
        "src": "aperture-nexus Python API source",
        "docs": "aperture-nexus user-facing documentation",
    }.get(folder, folder)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    reset = "--reset" in argv

    LOG.info("connecting to ApertureDB and Nexus…")
    admin = NexusAdmin()
    memory = Memory()
    memory._cfg.models.text_embedding = TEXT_MODEL

    api_key = _ensure_principal(admin)
    principal = memory.authenticate(user_id=DEMO_USER, api_key=api_key)
    LOG.info("authenticated as %r (workspace=%r)", DEMO_USER, DEMO_WORKSPACE)

    if reset:
        _reset(memory)
    elif _already_ingested(memory):
        LOG.info("session %r already has entries — skipping ingest.", DEMO_SESSION)
        LOG.info("re-run with --reset to wipe and reload.")
        return 0

    files_by_folder: dict[str, list[Path]] = {}
    for folder, path in _iter_files():
        files_by_folder.setdefault(folder, []).append(path)

    total_files = sum(len(v) for v in files_by_folder.values())
    LOG.info("ingesting %d files across %d folders…", total_files, len(files_by_folder))

    committed = 0
    total_chunks = 0
    for folder, paths in files_by_folder.items():
        ctx = Context(
            principal=principal,
            session_name=DEMO_SESSION,
            purpose=_folder_purpose(folder),
            organization=DEMO_WORKSPACE,
            department=folder,
        )
        for p in paths:
            rel = p.relative_to(REPO_ROOT).as_posix()
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                LOG.warning("non-utf8, skipped: %s", rel)
                continue
            chunks = _chunk_text(text)
            info = Information(context_id=ctx.id)
            for idx, chunk in enumerate(chunks):
                info.log(
                    text=chunk,
                    metadata={
                        "file_path": rel,
                        "folder": folder,
                        "ext": p.suffix.lstrip("."),
                        "chunk_idx": idx,
                        "chunk_total": len(chunks),
                    },
                    tag=rel,
                )
            LOG.info("  commit  %-45s  %d chunk(s)  (%d chars)", rel, len(chunks), len(text))
            memory.process_and_commit(ctx, info)
            committed += 1
            total_chunks += len(chunks)

    LOG.info("done — %d files, %d total chunks committed.", committed, total_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
