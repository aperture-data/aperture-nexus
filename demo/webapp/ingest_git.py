"""
ingest_git.py — load aperture-nexus git history into Nexus.

Walks `git log --patch` for this repo, creates a Principal per author
(you + anyone else who shows up in the history), and commits each git
commit as its own Nexus commit. Commit message = Context.purpose,
diff = Information entries (chunked), timestamp = created_at.

Runs alongside ingest.py — same workspace, same session, different
department ("git-history"). Search surfaces both file snapshots and
historical commits in one list.

Usage (from aperture-nexus/):
    python demo/webapp/ingest_git.py              # add git history to current workspace
    python demo/webapp/ingest_git.py --reset      # wipe git history and re-ingest
    python demo/webapp/ingest_git.py --limit 10   # only the N most recent commits
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "APERTUREDB_JSON",
    '{"host":"localhost","port":55556,"username":"admin","password":"admin","use_ssl":false}',
)

from aperture_nexus import Context, Information, Memory, NexusAdmin  # noqa: E402

LOG = logging.getLogger("nexus-demo.ingest_git")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = Path(__file__).resolve().parent
KEYS_DIR = WEBAPP_DIR / ".keys"
KEYS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_WORKSPACE = "nexus-demo"
DEMO_SESSION = "aperture-nexus-codebase"
# Must match the model used by ingest.py — descriptor sets are keyed
# by model, so mixing within one session makes search miss half the data.
TEXT_MODEL = "BAAI/bge-small-en-v1.5"

# Shared with ingest.py — if this changes, ingest.py's chunk-sized files
# won't co-exist cleanly in the same session.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Record / field separators for git log parsing — safe because no file
# content normally contains these ASCII control codes.
RECORD_SEP = "\x1f"   # unit separator
FIELD_SEP = "\x1e"    # record separator


# ----------------------------------------------------------------------
# Chunking (duplicated from ingest.py so this script is standalone)
# ----------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            nl = text.rfind("\n", i + size // 2, end)
            if nl != -1:
                end = nl + 1
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


# ----------------------------------------------------------------------
# Git log parsing
# ----------------------------------------------------------------------

def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, errors="replace",
    )


def _principal_name(author_name: str, author_email: str) -> str:
    """Derive a stable, lowercase, slug-ish principal user_id from author."""
    # Prefer local-part of email for stability across display-name changes.
    local = author_email.split("@", 1)[0]
    slug = re.sub(r"[^a-z0-9]+", "-", local.lower()).strip("-")
    return slug or re.sub(r"[^a-z0-9]+", "-", author_name.lower()).strip("-") or "unknown"


def _walk_commits(limit: int | None = None) -> Iterable[dict]:
    """Yield one dict per git commit, newest-first."""
    fmt = FIELD_SEP.join(["%H", "%an", "%ae", "%aI", "%s", "%b"]) + RECORD_SEP
    args = ["log", f"--format={fmt}"]
    if limit:
        args.append(f"-n{limit}")
    raw = _git(args)
    for record in raw.split(RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(FIELD_SEP)
        if len(parts) < 5:
            continue
        sha, name, email, date, subject = parts[:5]
        body = parts[5] if len(parts) > 5 else ""
        yield {
            "sha": sha,
            "author_name": name,
            "author_email": email,
            "date": date,
            "subject": subject.strip(),
            "body": body.strip(),
        }


def _diff(sha: str) -> str:
    """Get the full patch for one commit (no stats header)."""
    return _git(["show", "--format=", "--patch", sha])


# ----------------------------------------------------------------------
# Principal management
# ----------------------------------------------------------------------

def _ensure_principal(admin: NexusAdmin, user_id: str, user_name: str) -> str:
    """Create (or rotate) the principal and return a cached API key."""
    key_file = KEYS_DIR / f"{user_id}.key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    try:
        key = admin.create_principal(
            user_id=user_id, user_name=user_name,
            organization=DEMO_WORKSPACE, department="git-history",
        )
        LOG.info("created principal %r", user_id)
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        key = admin.rotate_key(user_id=user_id)
        LOG.info("rotated key for existing principal %r", user_id)
    key_file.write_text(key)
    return key


# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------

def _reset(memory: Memory) -> None:
    LOG.info("resetting git-history entries from session %r…", DEMO_SESSION)
    # Query everything in the demo session with department=git-history.
    probes = memory.search(
        filters={
            "session_name": DEMO_SESSION,
            "organization": DEMO_WORKSPACE,
            "department": "git-history",
        },
        k=1000,
    )
    LOG.info("  found %d entries to remove", len(probes))
    if probes:
        memory.remove(results=probes)
    LOG.info("  removed.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="wipe git-history entries before ingesting")
    p.add_argument("--limit", type=int, default=None, help="only ingest the N most recent commits")
    args = p.parse_args(argv)

    LOG.info("connecting to ApertureDB and Nexus…")
    admin = NexusAdmin()
    memory = Memory()
    memory._cfg.models.text_embedding = TEXT_MODEL

    # Each author becomes a Principal — map authors to (memory, principal) pairs.
    principal_cache: dict[str, tuple] = {}

    def mp_for(user_id: str, user_name: str):
        if user_id in principal_cache:
            return principal_cache[user_id]
        key = _ensure_principal(admin, user_id, user_name)
        m = Memory()
        m._cfg.models.text_embedding = TEXT_MODEL
        principal = m.authenticate(user_id=user_id, api_key=key)
        principal_cache[user_id] = (m, principal)
        return m, principal

    if args.reset:
        # Reset using the main demo principal (same workspace, can remove).
        _, _ = mp_for("nexus-demo-user", "Nexus Demo")
        _reset(principal_cache["nexus-demo-user"][0])

    commits = list(_walk_commits(limit=args.limit))
    LOG.info("walking %d commits…", len(commits))

    committed = 0
    total_chunks = 0
    seen_principals: set[str] = set()

    for c in commits:
        pid = _principal_name(c["author_name"], c["author_email"])
        if pid not in seen_principals:
            seen_principals.add(pid)
            LOG.info("  principal: %s  (%s <%s>)", pid, c["author_name"], c["author_email"])
        m, principal = mp_for(pid, c["author_name"])

        # Context per commit so each shows up as its own node in the graph.
        purpose = c["subject"][:200]  # purpose field has a soft length limit
        ctx = Context(
            principal=principal,
            session_name=DEMO_SESSION,
            purpose=purpose,
            organization=DEMO_WORKSPACE,
            department="git-history",
        )
        info = Information(context_id=ctx.id)

        # Entry 1: commit message (subject + body) as its own searchable chunk.
        msg_text = c["subject"]
        if c["body"]:
            msg_text += "\n\n" + c["body"]
        for idx, chunk in enumerate(_chunk_text(msg_text)):
            info.log(
                text=chunk,
                metadata={
                    "commit_sha": c["sha"],
                    "commit_short": c["sha"][:8],
                    "commit_subject": c["subject"][:200],
                    "author": c["author_name"],
                    "author_email": c["author_email"],
                    "commit_date": c["date"],
                    "kind": "commit_message",
                    "chunk_idx": idx,
                },
                tag=f"commit:{c['sha'][:8]}",
            )

        # Entries 2+: the diff, chunked. One entry per chunk.
        try:
            diff = _diff(c["sha"])
        except subprocess.CalledProcessError as e:
            LOG.warning("  could not get diff for %s: %s", c["sha"][:8], e)
            diff = ""
        if diff:
            chunks = _chunk_text(diff)
            for idx, chunk in enumerate(chunks):
                info.log(
                    text=chunk,
                    metadata={
                        "commit_sha": c["sha"],
                        "commit_short": c["sha"][:8],
                        "commit_subject": c["subject"][:200],
                        "author": c["author_name"],
                        "commit_date": c["date"],
                        "kind": "diff",
                        "chunk_idx": idx,
                        "chunk_total": len(chunks),
                    },
                    tag=f"commit:{c['sha'][:8]}",
                )

        LOG.info("  commit  %s  %s  (%d entries)",
                 c["sha"][:8], c["subject"][:60], len(info._entries))
        try:
            m.process_and_commit(ctx, info)
            committed += 1
            total_chunks += len(info._entries)
        except Exception as e:
            LOG.warning("  commit failed: %s", e)

    LOG.info("done — %d commits ingested, %d total entries, %d principal(s).",
             committed, total_chunks, len(seen_principals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
