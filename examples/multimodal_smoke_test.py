"""
multimodal_smoke_test.py — full-stack test for aperture-nexus.

Exercises every storage, retrieval, and search path against a
live ApertureDB instance:

  1.  Text via commit()             — raw, no embedding
  2.  Image via commit()            — PNG bytes as ApertureDB Image
  3.  Blob via commit()             — arbitrary bytes
  4.  Text via process_and_commit() — CLIP embedding + vector search
  5.  Image via process_and_commit()— CLIP embedding + vector search
  6.  Video via process_and_commit()— per-clip CLIP + vector search
  7.  Pre-computed embedding        — pass vector directly, no model
  8.  retrieve() — text             — bytes decode to original string
  9.  retrieve() — image            — bytes are valid PNG
  10. retrieve() — blob             — bytes match original
  11. retrieve() — video            — non-empty bytes returned
  12. Metadata search               — session_id filter across modalities
  13. connect()                     — link two contexts
  14. remove()                      — delete context, verify absent

Prerequisites:
    pip install aperture-nexus[clip,video]
    docker compose up -d

Run:
    python examples/multimodal_smoke_test.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import uuid
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"{PASS}  {label}")
    else:
        msg = label + (f": {detail}" if detail else "")
        print(f"{FAIL}  {msg}")
        _failures.append(msg)
    return condition


def section(title: str) -> None:
    width = 56
    print()
    print(f"── {title} {'─' * max(0, width - len(title))}")


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def _png_bytes(color: tuple = (100, 149, 237), size: int = 32) -> bytes:
    """Solid-colour PNG — no disk I/O."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mp4_bytes() -> bytes:
    """Minimal synthetic MP4 via OpenCV — 90 frames at 30 fps."""
    import cv2
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    out = cv2.VideoWriter(
        tmp.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30, (64, 64),
    )
    for i in range(90):
        frame = np.full((64, 64, 3), i * 2, dtype=np.uint8)
        out.write(frame)
    out.release()
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


def _is_valid_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

print("aperture-nexus multimodal smoke test")
print("=" * 40)

try:
    from aperture_nexus.admin import NexusAdmin
    from aperture_nexus.memory import Memory
    from aperture_nexus.context import Context
    from aperture_nexus.information import Information
except ImportError as e:
    print(f"Import error: {e}")
    print("Run: pip install aperture-nexus")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Temporary config — sets CLIP as the embedding model for all modalities
# ---------------------------------------------------------------------------

_cfg_file = tempfile.NamedTemporaryFile(
    suffix=".json", delete=False, mode="w"
)
json.dump({
    "models": {
        "text_embedding": "ViT-B/16",
        "image_embedding": "ViT-B/16",
        "video_embedding": "ViT-B/16",
    }
}, _cfg_file)
_cfg_file.close()
_cfg_path = _cfg_file.name

# ---------------------------------------------------------------------------
# Setup: principal + memory engine
# ---------------------------------------------------------------------------

section("Setup")

uid = f"smoke-{uuid.uuid4().hex[:8]}"
admin = NexusAdmin()

try:
    api_key = admin.create_principal(
        user_id=uid, user_name="Smoke Test User"
    )
    print(f"  Principal: {uid!r}")
except Exception as e:
    print(f"  Cannot create principal: {e}")
    os.unlink(_cfg_path)
    sys.exit(1)

try:
    memory_raw = Memory()               # no models — for commit() tests
    memory_clip = Memory(config=_cfg_path)  # CLIP — for process_and_commit()
    principal = memory_raw.authenticate(user_id=uid, api_key=api_key)
    print(f"  Authenticated as: {principal.user_name}")
except Exception as e:
    print(f"  Authentication failed: {e}")
    admin.delete_principal(uid)
    os.unlink(_cfg_path)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Text via commit() — raw storage, no embeddings
# ---------------------------------------------------------------------------

section("1. Text  —  commit() (raw, no embedding)")

TEXT_CONTENT = "ApertureDB is a multimodal vector database."
sid_text = f"smoke-text-{uuid.uuid4().hex[:8]}"
ctx1 = Context(
    principal=principal,
    session_id=sid_text,
    purpose="text raw storage test",
)
info = Information(context_id=ctx1.id)
info.log(text=TEXT_CONTENT)
info.log(text="aperture-nexus adds a memory layer on top.")
cid1 = None

try:
    cid1 = memory_raw.commit(ctx1, info)
    check("commit() returns context_id", bool(cid1))
    check("info drained after commit", len(info) == 0)
except Exception as e:
    check("commit() text", False, str(e))

# ---------------------------------------------------------------------------
# 2. Image via commit() — PNG bytes stored as ApertureDB Image
# ---------------------------------------------------------------------------

section("2. Image  —  commit() (raw bytes)")

BLUE_PNG = _png_bytes(color=(30, 100, 200))
sid_img = f"smoke-img-{uuid.uuid4().hex[:8]}"
ctx2 = Context(
    principal=principal,
    session_id=sid_img,
    purpose="image raw storage test",
)
info = Information(context_id=ctx2.id)
info.log(image=BLUE_PNG)
cid2 = None

try:
    cid2 = memory_raw.commit(ctx2, info)
    check("commit() image returns context_id", bool(cid2))
except Exception as e:
    check("commit() image", False, str(e))

# ---------------------------------------------------------------------------
# 3. Blob via commit() — arbitrary bytes
# ---------------------------------------------------------------------------

section("3. Blob  —  commit() (arbitrary bytes)")

BLOB_CONTENT = b"%PDF-1.4 this is a fake pdf document"
sid_blob = f"smoke-blob-{uuid.uuid4().hex[:8]}"
ctx3 = Context(
    principal=principal,
    session_id=sid_blob,
    purpose="blob raw storage test",
)
info = Information(context_id=ctx3.id)
info.log(blob=BLOB_CONTENT, document_type="pdf")
cid3 = None

try:
    cid3 = memory_raw.commit(ctx3, info)
    check("commit() blob returns context_id", bool(cid3))
except Exception as e:
    check("commit() blob", False, str(e))

# ---------------------------------------------------------------------------
# 4. Text via process_and_commit() — CLIP + vector search
# ---------------------------------------------------------------------------

section("4. Text  —  process_and_commit() + CLIP + vector search")

sid_clip_text = f"smoke-cliptext-{uuid.uuid4().hex[:8]}"
ctx4 = Context(
    principal=principal,
    session_id=sid_clip_text,
    purpose="CLIP text embedding test",
)
info = Information(context_id=ctx4.id)
info.log(text="vector search meets knowledge graph in ApertureDB")
info.log(text="multimodal memory for AI agents")
cid4 = None

try:
    cid4 = memory_clip.process_and_commit(ctx4, info)
    check(
        "process_and_commit() text returns context_id",
        bool(cid4),
    )
except Exception as e:
    check("process_and_commit() text", False, str(e))

if cid4:
    try:
        results = memory_clip.search(
            query="knowledge graph vector database",
            modality="text",
            k=5,
        )
        check(
            "text vector search returns results",
            len(results) >= 1,
            f"got {len(results)}",
        )
        check(
            "top result has a score",
            bool(results) and results[0].score > 0,
        )
        check(
            "committed text session in results",
            any(r.session_id == sid_clip_text for r in results),
        )
    except Exception as e:
        check("text vector search", False, str(e))

# ---------------------------------------------------------------------------
# 5. Image via process_and_commit() — CLIP + vector search
# ---------------------------------------------------------------------------

section("5. Image  —  process_and_commit() + CLIP + vector search")

RED_PNG = _png_bytes(color=(220, 50, 50))
sid_clip_img = f"smoke-clipimg-{uuid.uuid4().hex[:8]}"
ctx5 = Context(
    principal=principal,
    session_id=sid_clip_img,
    purpose="CLIP image embedding test",
)
info = Information(context_id=ctx5.id)
info.log(image=RED_PNG)
cid5 = None

try:
    cid5 = memory_clip.process_and_commit(ctx5, info)
    check(
        "process_and_commit() image returns context_id",
        bool(cid5),
    )
except Exception as e:
    check("process_and_commit() image", False, str(e))

if cid5:
    try:
        results = memory_clip.search(
            query="red square image",
            modality="image",
            k=5,
        )
        check(
            "text→image cross-modal search returns results",
            len(results) >= 1,
            f"got {len(results)}",
        )
        check(
            "committed image session in results",
            any(r.session_id == sid_clip_img for r in results),
        )
    except Exception as e:
        check("text→image cross-modal search", False, str(e))

# ---------------------------------------------------------------------------
# 6. Video via process_and_commit() — per-clip CLIP + vector search
# ---------------------------------------------------------------------------

section("6. Video  —  process_and_commit() + per-clip CLIP + search")

sid_video = f"smoke-video-{uuid.uuid4().hex[:8]}"
ctx6 = Context(
    principal=principal,
    session_id=sid_video,
    purpose="video CLIP embedding test",
)
info = Information(context_id=ctx6.id)
cid6 = None

try:
    mp4 = _mp4_bytes()
    info.log(video=mp4)
    cid6 = memory_clip.process_and_commit(ctx6, info)
    check(
        "process_and_commit() video returns context_id",
        bool(cid6),
    )
except Exception as e:
    check("process_and_commit() video", False, str(e))

if cid6:
    try:
        results = memory_clip.search(
            query="motion over time",
            modality="video",
            k=5,
        )
        check(
            "text→video search returns results",
            len(results) >= 1,
            f"got {len(results)}",
        )
        check(
            "committed video session in results",
            any(r.session_id == sid_video for r in results),
        )
        check(
            "video result has start_frame",
            any(r.start_frame is not None for r in results),
        )
    except Exception as e:
        check("text→video search", False, str(e))

# ---------------------------------------------------------------------------
# 7. Pre-computed embedding — pass vector directly, skip model call
# ---------------------------------------------------------------------------

section("7. Pre-computed embedding  —  no model call")

sid_precomp = f"smoke-precomp-{uuid.uuid4().hex[:8]}"
ctx7 = Context(
    principal=principal,
    session_id=sid_precomp,
    purpose="pre-computed embedding test",
)
info = Information(context_id=ctx7.id)
vec = np.random.rand(512).astype(np.float32)
vec /= np.linalg.norm(vec)
info.log(
    text="pre-computed embedding test entry",
    embedding=vec,
    embedding_model="ViT-B/16",
)
cid7 = None

try:
    cid7 = memory_raw.process_and_commit(ctx7, info)
    check(
        "process_and_commit() with pre-computed embedding",
        bool(cid7),
    )
    results = memory_raw.search(
        query=vec,
        modality="text",
        embedding_model="ViT-B/16",
        k=5,
    )
    check(
        "vector search finds pre-computed entry",
        any(r.session_id == sid_precomp for r in results),
        f"got {len(results)} results",
    )
except Exception as e:
    check("pre-computed embedding round-trip", False, str(e))

# ---------------------------------------------------------------------------
# 8-11. retrieve() — verify actual content bytes round-trip
# ---------------------------------------------------------------------------

section("8. retrieve()  —  text content round-trip")

if cid1:
    try:
        entries = memory_raw.retrieve(cid1)
        text_entries = [e for e in entries if e.modality == "text"]
        check(
            "retrieve() returns text entries",
            len(text_entries) >= 1,
            f"got {len(text_entries)} text entries",
        )
        texts = [e.text for e in text_entries]
        check(
            "original text content matches",
            TEXT_CONTENT in texts,
            f"got: {texts}",
        )
    except Exception as e:
        check("retrieve() text", False, str(e))
else:
    print(f"{SKIP}  retrieve() text — commit() failed earlier")

section("9. retrieve()  —  image bytes round-trip")

if cid2:
    try:
        entries = memory_raw.retrieve(cid2)
        img_entries = [e for e in entries if e.modality == "image"]
        check(
            "retrieve() returns image entries",
            len(img_entries) >= 1,
            f"got {len(img_entries)} image entries",
        )
        check(
            "image bytes are valid PNG",
            bool(img_entries) and _is_valid_png(img_entries[0].data),
        )
        check(
            "image bytes match what was stored",
            bool(img_entries) and img_entries[0].data == BLUE_PNG,
        )
    except Exception as e:
        check("retrieve() image", False, str(e))
else:
    print(f"{SKIP}  retrieve() image — commit() failed earlier")

section("10. retrieve()  —  blob bytes round-trip")

if cid3:
    try:
        entries = memory_raw.retrieve(cid3)
        blob_entries = [e for e in entries if e.modality == "blob"]
        check(
            "retrieve() returns blob entries",
            len(blob_entries) >= 1,
            f"got {len(blob_entries)} blob entries",
        )
        check(
            "blob bytes match what was stored",
            bool(blob_entries) and blob_entries[0].data == BLOB_CONTENT,
        )
        check(
            "blob document_type preserved",
            bool(blob_entries) and blob_entries[0].document_type == "pdf",
        )
    except Exception as e:
        check("retrieve() blob", False, str(e))
else:
    print(f"{SKIP}  retrieve() blob — commit() failed earlier")

section("11. retrieve()  —  video bytes round-trip")

if cid6:
    try:
        entries = memory_clip.retrieve(cid6)
        vid_entries = [e for e in entries if e.modality == "video"]
        check(
            "retrieve() returns video entries",
            len(vid_entries) >= 1,
            f"got {len(vid_entries)} video entries",
        )
        check(
            "video bytes are non-empty",
            bool(vid_entries) and len(vid_entries[0].data) > 0,
        )
    except Exception as e:
        check("retrieve() video", False, str(e))
else:
    print(f"{SKIP}  retrieve() video — commit() failed earlier")

# ---------------------------------------------------------------------------
# 12. Metadata search — session_id filter across all modalities
# ---------------------------------------------------------------------------

section("12. Metadata search  —  session_id filter, all modalities")

for label, sid, cid in [
    ("text",  sid_text,      cid1),
    ("image", sid_img,       cid2),
    ("blob",  sid_blob,      cid3),
    ("video", sid_video,     cid6),
]:
    if not cid:
        print(f"{SKIP}  metadata search ({label}) — commit() failed")
        continue
    try:
        results = memory_raw.search(filters={"session_id": sid})
        check(
            f"metadata search finds {label} context",
            len(results) >= 1,
            f"got {len(results)}",
        )
        check(
            f"metadata result has correct session_id ({label})",
            all(r.session_id == sid for r in results),
        )
    except Exception as e:
        check(f"metadata search ({label})", False, str(e))

# ---------------------------------------------------------------------------
# 13. connect() — link two contexts
# ---------------------------------------------------------------------------

section("13. connect()  —  link two contexts")

if cid1 and cid4:
    try:
        memory_raw.connect(ctx1, ctx4, relationship="related_to")
        check("connect() two contexts succeeds", True)
    except Exception as e:
        check("connect() two contexts", False, str(e))
else:
    print(f"{SKIP}  connect() — prior contexts unavailable")

# ---------------------------------------------------------------------------
# 14. remove() — delete a context and verify it is gone
# ---------------------------------------------------------------------------

section("14. remove()  —  delete a context")

if cid3:
    try:
        memory_raw.remove(cid3)
        check("remove() blob context succeeds", True)
        results = memory_raw.search(filters={"session_id": sid_blob})
        check(
            "removed context absent from metadata search",
            len(results) == 0,
            f"still found {len(results)} results",
        )
    except Exception as e:
        check("remove()", False, str(e))
else:
    print(f"{SKIP}  remove() — blob context unavailable")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

section("Cleanup")

os.unlink(_cfg_path)

try:
    admin.delete_principal(uid)
    check("delete_principal() succeeds", True)
except Exception as e:
    check("delete_principal()", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 40)
if _failures:
    print(f"FAILED — {len(_failures)} check(s) failed:")
    for f in _failures:
        print(f"  x {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    print()
    print("Hello, Nexus!")
