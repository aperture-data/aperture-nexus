"""
multimodal_demo.py — aperture-nexus multimodal walkthrough.

Shows a support agent storing a customer complaint (text) alongside
a product photo (image) in the same commit. A second agent retrieves
everything — text and image — by searching for the session.

Then: text-query → image results via CLIP semantic search.

Prerequisites:
    pip install "aperture-nexus[clip]"
    docker compose up -d

Connection is resolved by the ApertureDB SDK's create_connector(), which
reads (in priority order) APERTUREDB_KEY, APERTUREDB_JSON, APERTUREDB_CONFIG,
or the active `adb config`. If nothing is set, this script falls back to
the local Docker stack defaults so it runs out of the box.

For a deeper dive on connection options and the JSON API, see
the aperturedb-connector skill (aperturedb-connector-skill.md) or run
`adb-nexus init` to bootstrap a persistent config + .env.

Run:
    python demo/multimodal_demo.py
"""

import io
import os
import sys

# ── Connection ────────────────────────────────────────────────────────────────
#
# create_connector() from aperturedb.CommonLibrary already handles the full
# priority chain (APERTUREDB_KEY → APERTUREDB_JSON → APERTUREDB_CONFIG → adb
# config). We only fill in a local-Docker default when nothing at all is set,
# so the demo runs zero-config against `docker compose up -d`.

def _ensure_connection() -> None:
    if any(os.environ.get(v) for v in ("APERTUREDB_KEY", "APERTUREDB_JSON", "APERTUREDB_CONFIG")):
        return
    from aperturedb.CommonLibrary import create_connector
    try:
        create_connector()  # succeeds if an active `adb config` exists
        return
    except Exception:
        pass
    os.environ["APERTUREDB_JSON"] = (
        '{"host":"localhost","port":55556,'
        '"username":"admin","password":"admin","use_ssl":false}'
    )

_ensure_connection()

from aperture_nexus import Memory, Context, Information, NexusAdmin
from aperture_nexus.exceptions import NexusStorageError, NexusValidationError

DEMO_USER = "demo-multimodal"
DEMO_ORG  = "acme-corp"

BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
DIM   = "\033[2m"
RESET = "\033[0m"

def ok(msg):   print(f"\033[32m  ✓\033[0m  {msg}")
def step(msg): print(f"\n{BOLD}  ▸  {msg}{RESET}")
def dim(msg):  print(f"  {DIM}{msg}{RESET}")


# ── Helpers ───────────────────────────────────────────────────────────────────

_FONT_CANDIDATES = (
    # Linux (Debian/Ubuntu)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # Linux (Fedora/RHEL)
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    # Windows
    "C:\\Windows\\Fonts\\arialbd.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Last resort — 8pt bitmap font. The demo will still work, just look tiny.
    return ImageFont.load_default()


def _make_product_image(label: str, color: tuple) -> bytes:
    """Generate a labelled product photo — no disk I/O. Works on Linux, macOS, Windows."""
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new("RGB", (320, 240), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 309, 229], outline=(255, 255, 255), width=3)
    draw.text((160, 120), label, fill=(255, 255, 255), font=_load_font(24), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _provision_principal(admin: NexusAdmin) -> str:
    try:
        return admin.create_principal(
            user_id=DEMO_USER,
            user_name="Multimodal Demo Agent",
            organization=DEMO_ORG,
            department="support",
        )
    except (NexusStorageError, NexusValidationError):
        admin.delete_principal(user_id=DEMO_USER)
        return admin.create_principal(
            user_id=DEMO_USER,
            user_name="Multimodal Demo Agent",
            organization=DEMO_ORG,
            department="support",
        )


# ── Demo ─────────────────────────────────────────────────────────────────────

def run():
    print()
    print(f"{BOLD}  aperture-nexus  ·  multimodal demo{RESET}")
    print(f"  {DIM}text + image  ·  commit  ·  search  ·  CLIP semantic search{RESET}")
    print()

    admin  = NexusAdmin()
    api_key = _provision_principal(admin)
    memory  = Memory()
    principal = memory.authenticate(user_id=DEMO_USER, api_key=api_key)
    ok(f"Connected  ·  principal: {DEMO_USER}  ·  org: {DEMO_ORG}")

    # ── Step 1: Agent 1 — store text + image in one commit ───────────────────

    step("Agent 1 — customer reports a defective product (text + image)")

    ctx1 = Context(
        principal=principal,
        session_name="support-ticket-8823",
        purpose="Defective product — scratched surface reported by customer",
    )

    product_photo = _make_product_image("DEFECT", color=(160, 30, 30))

    info1 = Information(context_id=ctx1.id)
    info1.log(text="Customer reports surface scratches on unit received 2025-06-01.")
    info1.log(text="Customer is on enterprise plan. Priority: high.")
    info1.log(
        text="Photo attached — scratch visible on top-left corner of unit.",
        image=product_photo,
    )

    commit_id = memory.commit(ctx1, info1)
    ok(f"Committed  ·  3 entries (2 text, 1 text+image)  ·  commit_id: {commit_id[:16]}…")
    dim("Inspect at http://localhost:8087  →  FindBlob + FindImage")

    # ── Step 2: Agent 2 — retrieve everything by session ─────────────────────

    step("Agent 2 — retrieves the full session by metadata filter")

    results = memory.search(filters={"session_name": "support-ticket-8823"})

    ok(f"{len(results)} entries retrieved:\n")
    for r in results:
        modality_tag = f"[{r.modality}]"
        if r.text:
            print(f"    {GREEN}{modality_tag:<8}{RESET}  {r.text[:80]}")
        else:
            print(f"    {GREEN}{modality_tag:<8}{RESET}  (binary content — {r.modality})")

    # ── Step 3: CLIP semantic search — text query finds the image ─────────────

    step("CLIP semantic search — text query → image result")

    try:
        import aperture_nexus._embeddings as _emb  # noqa: F401
    except ImportError:
        print("  (skipped — pip install aperture-nexus[clip] to enable)")
        _cleanup(admin, memory, ctx1)
        return

    clip_memory = Memory()
    clip_memory._cfg.models.text_embedding  = "ViT-B/16"
    clip_memory._cfg.models.image_embedding = "ViT-B/16"

    ctx2 = Context(
        principal=principal,
        session_name="support-ticket-8824",
        purpose="Product quality issue — surface damage",
    )

    info2 = Information(context_id=ctx2.id)
    info2.log(image=_make_product_image("DEFECT", color=(160, 30, 30)))
    info2.log(image=_make_product_image("OK",     color=(30, 130, 60)))
    clip_memory.process_and_commit(ctx2, info2)
    ok("Committed 2 product images with CLIP embeddings")

    query = "damaged product with scratch"
    img_results = clip_memory.search(
        query=query,
        modality="image",
        embedding_model="ViT-B/16",
        k=5,
    )
    ok(f'Query: "{query}"  →  {len(img_results)} image result(s)')
    for r in img_results:
        print(f"    {GREEN}[image]  {RESET} score: {r.score:.3f}  session: {r.session_id}")

    # ── Done ─────────────────────────────────────────────────────────────────

    print()
    print(f"  {DIM}Explore at http://localhost:8087{RESET}")
    print(f"  {DIM}FindBlob / FindImage / FindEntity (NexusContext){RESET}")
    print()

    _cleanup(admin, memory, ctx1, ctx2)


def _cleanup(admin, memory, *contexts):
    # Session-scoped removal cascades: content + NexusCommit + NexusContext + NexusSession.
    # Deduplicate in case multiple contexts share a session.
    session_ids = {ctx.session_id for ctx in contexts if ctx.session_id}
    for sid in session_ids:
        try:
            memory.remove(session_id=sid)
        except Exception:
            pass
    try:
        admin.delete_principal(user_id=DEMO_USER)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(0)
