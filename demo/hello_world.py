"""
aperture-nexus hello world.

Demonstrates the full commit → search round-trip:
  1. Authenticate a principal created by `adb-nexus init`
  2. Open a session and commit a few memories (with CLIP embeddings)
  3. Search by session metadata and by semantic text query
"""

import os
from dotenv import load_dotenv

from aperture_nexus import Memory, Context, Information, generate_session_id

# ── Load the API key written by `adb-nexus init` ─────────────────────────────
env_file = os.environ.get("NEXUS_ENV_FILE", ".env")
load_dotenv(env_file)
api_key = os.environ["NEXUS_API_KEY"]
user_id  = os.environ.get("USER", "demo-user")

# ── Connect to ApertureDB and authenticate ────────────────────────────────────
memory = Memory()

# Use CLIP (ViT-B-32) for text and image embeddings.
memory._cfg.models.text_embedding  = "ViT-B-32"
memory._cfg.models.image_embedding = "ViT-B-32"

principal = memory.authenticate(user_id=user_id, api_key=api_key)
name = principal.user_name or principal.user_id
print(f"\n✓  Authenticated  →  {name}")

# ── Open a session and build a context ───────────────────────────────────────
sid = generate_session_id(prefix="demo")
ctx = Context(
    principal=principal,
    session_id=sid,
    purpose="aperture-nexus hello world",
)
print(f"   session : {sid[:40]}...")
print(f"   context : {ctx.id}")

# ── Log memories and commit with CLIP embeddings ──────────────────────────────
info = Information(context_id=ctx.id)
info.log(text="ApertureDB is a multimodal vector database built for AI.")
info.log(text="aperture-nexus gives AI agents persistent, searchable memory.")
info.log(text="Memories can be text, images, video, or raw blobs.")

print()
ctx_id = memory.process_and_commit(ctx, info)
print(f"✓  Committed 3 memories with CLIP embeddings")

# ── Search by metadata (no vector) ───────────────────────────────────────────
print()
results = memory.search(filters={"session_id": sid})
print(f"   Metadata search  →  {len(results)} result(s)")
for r in results:
    print(f"     • {(r.text or '')[:60]!r}")

# ── Search by semantic text query ─────────────────────────────────────────────
print()
results = memory.search(
    query="vector database for AI agents",
    modality="text",
    embedding_model="ViT-B-32",
    filters={"session_id": sid},
    k=3,
)
print(f"   Semantic search  →  top {len(results)} result(s)")
for r in results:
    print(f"     • score={r.score:.3f}  {(r.text or '')[:55]!r}")

print()
print("✓  Done.\n")
