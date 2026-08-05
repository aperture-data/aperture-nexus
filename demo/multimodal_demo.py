"""
multimodal_demo.py — aperture-nexus multimodal walkthrough.

Shows a support agent storing a customer complaint (text) alongside
a product photo (image) in the same commit. A second agent retrieves
everything — text and image — by searching for the session.

Then: text-query → image results via CLIP semantic search.

Prerequisites:
    pip install "aperture-nexus[clip]"
    docker compose up -d
    adb-nexus init   (or set APERTUREDB_JSON + NEXUS_API_KEY)

Run:
    python demo/multimodal_demo.py
"""

import io
import os
import sys

# ── Connection ────────────────────────────────────────────────────────────────

# Works against the local Docker stack out of the box.
if not os.environ.get("APERTUREDB_KEY") and not os.environ.get("APERTUREDB_JSON"):
    os.environ["APERTUREDB_JSON"] = (
        '{"host":"localhost","port":55556,'
        '"username":"admin","password":"admin","use_ssl":false}'
    )

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

def _make_product_image(label: str, color: tuple) -> bytes:
    """Generate a labelled product photo — no disk I/O."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img = PILImage.new("RGB", (320, 240), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 309, 229], outline=(255, 255, 255), width=3)
    # Draw label in the centre — use default font if truetype unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((160, 120), label, fill=(255, 255, 255), font=font, anchor="mm")
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
    for ctx in contexts:
        try:
            memory.remove(context_id=ctx.id)
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
