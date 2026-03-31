"""
quickstart.py — aperture-nexus in 30 lines

Stores two text entries under a session and searches for them.
No models required — uses commit() for raw storage.

Run:
    python examples/quickstart.py

Prerequisites:
    pip install aperture-nexus
    docker compose up -d   # or set APERTUREDB_KEY in your environment
    adb-nexus init         # or set credentials via environment variables
"""

import os
from aperture_nexus import Memory, Context, Information

# ── Connect ──────────────────────────────────────────────────────────────────

memory = Memory()
principal = memory.authenticate(
    user_id=os.environ["NEXUS_USER_ID"],
    api_key=os.environ["NEXUS_API_KEY"],
)

print(f"Authenticated as: {principal}")

# ── Describe the session ─────────────────────────────────────────────────────

ctx = Context(
    principal=principal,
    session_name="quickstart-demo",
    purpose="Testing aperture-nexus",
)

print(f"Session: {ctx.session_id}")

# ── Log information ───────────────────────────────────────────────────────────

info = Information(context_id=ctx.id)

info.log(text="aperture-nexus stores text, images, video, and blobs in ApertureDB.")
info.log(text="Sessions can have multiple participants — each gets their own Context.")
info.log(text="Use commit() for raw storage, process_and_commit() to generate embeddings.")

print(f"Buffered {len(info)} entries")

# ── Commit (raw storage — no model calls needed) ──────────────────────────────

memory_id = memory.commit(ctx, info)
print(f"Committed → memory_id: {memory_id}")

# ── Search ────────────────────────────────────────────────────────────────────

results = memory.search(
    query="embeddings and vector search",
    filters={"session_id": ctx.session_id},
)

print(f"\nSearch results ({len(results)} found):")
for r in results:
    print(f"  [{r.score:.3f}] {r.text[:80]}")

print("\nDone. Inspect stored data at http://localhost:8087 (ApertureDB web UI).")
