"""
hello_nexus.py — smoke test for a live ApertureDB instance.

Runs the full stack end-to-end:
  1. Connect to ApertureDB
  2. Create a temporary principal (admin credentials required)
  3. Authenticate as that principal (regular credentials only)
  4. Log text entries into a session
  5. Commit to ApertureDB
  6. Search by session filter
  7. Clean up the test principal

No models or embeddings required — uses commit() for raw storage.

Run:
    python examples/hello_nexus.py

Prerequisites:
    pip install aperture-nexus
    docker compose up -d          # community edition, or point at a real instance
    # Admin credentials in environment (APERTUREDB_KEY or APERTUREDB_USER/PASSWORD)

If the script exits with "Hello, Nexus!" everything is working.
"""

import uuid
import sys

# ---------------------------------------------------------------------------
# 1. Connect and create a temporary principal
# ---------------------------------------------------------------------------

print("aperture-nexus hello world")
print("-" * 40)

try:
    from aperture_nexus.admin import NexusAdmin
    from aperture_nexus.memory import Memory
    from aperture_nexus.context import Context
    from aperture_nexus.information import Information
except ImportError as e:
    print(f"Import error: {e}")
    print("Run: pip install aperture-nexus")
    sys.exit(1)

user_id = f"hello-nexus-{uuid.uuid4().hex[:8]}"
admin = NexusAdmin()

try:
    api_key = admin.create_principal(user_id=user_id, user_name="Hello Nexus")
    print(f"  Principal created: {user_id!r}")
except Exception as e:
    print(f"  Failed to create principal: {e}")
    print("  Ensure admin ApertureDB credentials are set in your environment.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Authenticate — regular credentials only from here on
# ---------------------------------------------------------------------------

try:
    memory = Memory()
    principal = memory.authenticate(user_id=user_id, api_key=api_key)
    print(f"  Authenticated as: {principal}")
except Exception as e:
    print(f"  Authentication failed: {e}")
    admin.delete_principal(user_id)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Log entries into a session
# ---------------------------------------------------------------------------

session_id = f"hello-session-{uuid.uuid4().hex[:8]}"
ctx = Context(
    principal=principal,
    session_id=session_id,
    purpose="Hello Nexus smoke test",
)
print(f"  Session: {ctx.session_id}")

info = Information(context_id=ctx.id)
info.log(text="aperture-nexus is alive.")
info.log(text="Text, images, video, and blobs — all in one memory layer.")
info.log(text="Built on ApertureDB: vector search meets knowledge graph.")
print(f"  Buffered {len(info)} entries")

# ---------------------------------------------------------------------------
# 4. Commit to ApertureDB
# ---------------------------------------------------------------------------

try:
    context_id = memory.commit(ctx, info)
    print(f"  Committed → context_id: {context_id}")
except Exception as e:
    print(f"  Commit failed: {e}")
    admin.delete_principal(user_id)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Search — retrieve by session filter (no vector search needed)
# ---------------------------------------------------------------------------

try:
    results = memory.search(filters={"session_id": session_id})
    print(f"  Search returned {len(results)} result(s)")
    for r in results:
        snippet = (r.text or "")[:60]
        print(f"    session={r.session_id!r}  text={snippet!r}")
except Exception as e:
    print(f"  Search failed: {e}")
    admin.delete_principal(user_id)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Clean up
# ---------------------------------------------------------------------------

try:
    admin.delete_principal(user_id)
    print(f"  Principal {user_id!r} removed")
except Exception as e:
    print(f"  Cleanup warning: {e}")

print()
print("Hello, Nexus!")
