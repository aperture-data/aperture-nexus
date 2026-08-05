"""
present.py — aperture-nexus presentation demo.

Knowledge → Memory → Context → Cognition

Four auto-running acts with manual Continue prompts between them.
A silent seed phase pre-loads org knowledge before the demo begins.
Increase your terminal font to 18–20pt before running.

    python demo/present.py

ApertureDB must be running (docker compose up -d).
Requires: pip install "aperture-nexus[clip]"
"""

import io
import logging
import os
import sys
import time
import subprocess
import warnings
import uuid

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ── Connection defaults (local Docker stack) ──────────────────────────────────

if not os.environ.get("APERTUREDB_KEY") and not os.environ.get("APERTUREDB_JSON"):
    os.environ["APERTUREDB_JSON"] = (
        '{"host":"localhost","port":55556,'
        '"username":"admin","password":"admin","use_ssl":false}'
    )

# ── Terminal styling ──────────────────────────────────────────────────────────

BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
WHITE  = "\033[97m"
RESET  = "\033[0m"

def W():
    import shutil
    return min(shutil.get_terminal_size(fallback=(80, 24)).columns, 90)

def clear():
    print("\033[2J\033[H", end="")

def blank(n=1):
    print("\n" * (n - 1))

def slow(text, delay=0.022, indent="    "):
    print(indent, end="", flush=True)
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def reveal(text, indent="    ", pause=0.5):
    print(f"{indent}{text}")
    time.sleep(pause)

def banner(title, subtitle=""):
    w = W()
    bar = "═" * (w - 4)
    clear()
    blank()
    print(f"  {BOLD}{CYAN}╔{bar}╗{RESET}")
    print(f"  {BOLD}{CYAN}║{title.center(w - 4)}║{RESET}")
    if subtitle:
        print(f"  {BOLD}{CYAN}║{subtitle.center(w - 4)}║{RESET}")
    print(f"  {BOLD}{CYAN}╚{bar}╝{RESET}")
    blank()

def section(title):
    w = W()
    blank()
    print(f"  {BOLD}{YELLOW}{'─' * 4}  {title}  {'─' * max(0, w - len(title) - 10)}{RESET}")
    blank()

def code(line, indent="    "):
    print(f"{indent}{BOLD}{CYAN}", end="", flush=True)
    for ch in line:
        print(ch, end="", flush=True)
        time.sleep(0.018)
    print(RESET)

def result(label, value, indent="      "):
    time.sleep(0.15)
    print(f"{indent}{GREEN}✓{RESET}  {BOLD}{label}{RESET}  {DIM}{value}{RESET}")

def web_ui_prompt():
    w = W()
    blank()
    print(f"  {BOLD}{YELLOW}{'─' * (w - 4)}{RESET}")
    print(f"  {BOLD}{WHITE}  👉  Open  http://localhost:8087  in your browser{RESET}")
    print(f"  {DIM}  Log in: admin / admin   →   FindImage or FindBlob{RESET}")
    print(f"  {BOLD}{YELLOW}{'─' * (w - 4)}{RESET}")
    blank()

def cont(label="next"):
    blank()
    try:
        input(f"  {BOLD}▸  Press Enter to {label}…{RESET}  ")
    except EOFError:
        pass
    blank()

def ok(msg):
    time.sleep(0.3)
    print(f"\n  {GREEN}{BOLD}✓{RESET}  {msg}")


# ── Docker stack ──────────────────────────────────────────────────────────────

def ensure_stack():
    banner("aperture-nexus", "Starting services…")
    print(f"  {DIM}Checking Docker stack…{RESET}", flush=True)
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        time.sleep(2)
        print(f"  {GREEN}✓{RESET}  ApertureDB  ·  Lenz  ·  Web UI — all running")
        print(f"  {DIM}Web UI:  http://localhost:8087  (admin / admin){RESET}")
    except Exception as e:
        print(f"  {YELLOW}⚠  Could not start Docker stack: {e}{RESET}")
        print(f"  {DIM}Run manually: docker compose up -d{RESET}")
    time.sleep(1)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _make_image(label, color, size=(400, 300)):
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img  = PILImage.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size[0] - 9, size[1] - 9], outline=(255, 255, 255), width=4)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((size[0] // 2, size[1] // 2), label, fill=(255, 255, 255),
              font=font, anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Principal management ──────────────────────────────────────────────────────

ORG  = "acme-corp"
DEPT = "support"

SEED_AGENTS = [
    ("alice-support", "Alice Chen"),
    ("bob-support",   "Bob Patel"),
    ("carol-support", "Carol Wu"),
]

DEMO_USER = f"presenter-{uuid.uuid4().hex[:6]}"

_all_principals = []   # user_id strings — for cleanup
_all_contexts   = []   # context objects — for cleanup


def _create_principal(admin, user_id, user_name):
    from aperture_nexus.exceptions import NexusStorageError, NexusValidationError
    try:
        return admin.create_principal(
            user_id=user_id, user_name=user_name,
            organization=ORG, department=DEPT,
        )
    except (NexusStorageError, NexusValidationError):
        admin.delete_principal(user_id=user_id)
        return admin.create_principal(
            user_id=user_id, user_name=user_name,
            organization=ORG, department=DEPT,
        )


def cleanup(admin, memory):
    for ctx in _all_contexts:
        try:
            memory.remove(context_id=ctx.id)
        except Exception:
            pass
    for uid in _all_principals:
        try:
            admin.delete_principal(user_id=uid)
        except Exception:
            pass


# ── Seed phase (silent) ───────────────────────────────────────────────────────

PRIOR_REPORTS = [
    {
        "agent":   "alice-support",
        "session": "ticket-7101",
        "purpose": "Surface scratch on SkyDock Pro unit — customer complaint",
        "entries": [
            "Customer received SkyDock Pro with a visible surface scratch across the lid.",
            "Unit photographed upon receipt. Damage appears pre-shipping.",
        ],
        "image_label": "SCRATCH",
        "image_color": (140, 50, 50),
    },
    {
        "agent":   "bob-support",
        "session": "ticket-7245",
        "purpose": "SkyDock Pro coating defect reported — paint flaking at corners",
        "entries": [
            "Customer reports paint flaking at all four corners of the SkyDock Pro.",
            "Third report this month. Engineering flagged for QA review.",
        ],
        "image_label": "FLAKING",
        "image_color": (130, 80, 20),
    },
    {
        "agent":   "carol-support",
        "session": "ticket-7389",
        "purpose": "SkyDock Pro discoloration — heat damage suspected",
        "entries": [
            "Discoloration on top surface, customer suspects heat damage from shipping.",
            "Similar to tickets 7101 and 7245. Pattern: same product batch.",
        ],
        "image_label": "DISCOLOR",
        "image_color": (90, 60, 130),
    },
]


def seed_knowledge(admin, memory):
    """Pre-load 3 prior support reports from 3 different agents into org memory."""
    from aperture_nexus import Context, Information

    clip_mem = _make_clip_memory()
    agent_map = {uid: uname for uid, uname in SEED_AGENTS}

    for report in PRIOR_REPORTS:
        user_id   = report["agent"]
        user_name = agent_map[user_id]

        api_key   = _create_principal(admin, user_id, user_name)
        _all_principals.append(user_id)

        principal = memory.authenticate(user_id=user_id, api_key=api_key)

        ctx = Context(
            principal=principal,
            session_name=report["session"],
            purpose=report["purpose"],
        )
        _all_contexts.append(ctx)

        info = Information(context_id=ctx.id)
        for text in report["entries"]:
            info.log(text=text)
        info.log(
            text=f"Product photo — {report['image_label'].lower()} visible on unit surface.",
            image=_make_image(report["image_label"], report["image_color"]),
        )

        clip_mem.process_and_commit(ctx, info)


def _make_clip_memory():
    from aperture_nexus import Memory as _Mem
    m = _Mem()
    m._cfg.models.text_embedding  = "ViT-B/16"
    m._cfg.models.image_embedding = "ViT-B/16"
    return m


# ── Act 1 — Knowledge ─────────────────────────────────────────────────────────

def act_knowledge(memory):
    banner("KNOWLEDGE", "What your organization already knows")

    section("Before this call — org memory already holds prior reports")
    reveal(
        f"  {DIM}Three support agents handled similar cases this month.{RESET}",
        indent="", pause=0.5,
    )
    blank()
    code('results = memory.search(filters={"organization": "acme-corp"})')
    results = memory.search(filters={"organization": ORG})
    blank()

    text_entries  = [r for r in results if r.modality == "text"]
    image_entries = [r for r in results if r.modality == "image"]

    ok(
        f"{len(results)} entries found  ·  "
        f"{len(text_entries)} text  ·  {len(image_entries)} images\n"
    )
    time.sleep(0.3)

    # Group by session for display
    by_session = {}
    for r in results:
        by_session.setdefault(r.session_id, []).append(r)

    agent_names = {uid: uname for uid, uname in SEED_AGENTS}
    for sid, entries in list(by_session.items())[:3]:
        user_id    = entries[0].user_id or "unknown"
        name       = agent_names.get(user_id, user_id)
        modalities = ", ".join(sorted({e.modality for e in entries}))
        texts      = [e for e in entries if e.text]
        first      = texts[0].text if texts else ""
        short      = (first[:62] + "…") if len(first) > 63 else first
        reveal(
            f"    {GREEN}▸{RESET}  {BOLD}{name}{RESET}  {DIM}({modalities}){RESET}",
            indent="", pause=0.2,
        )
        reveal(f"       {DIM}{short}{RESET}", indent="", pause=0.5)

    time.sleep(0.8)
    web_ui_prompt()
    print(f"  {DIM}  FindBlob → see the support notes{RESET}")
    print(f"  {DIM}  FindImage → see the defect photos from all three agents{RESET}")


# ── Act 2 — Memory ────────────────────────────────────────────────────────────

def act_memory(memory, principal):
    from aperture_nexus import Context, Information

    banner("MEMORY", "A new call comes in — capture it")

    section("A fourth agent opens a new session")
    code("ctx = Context(")
    code(f'    principal   = principal,')
    code('    session_name= "ticket-7502",')
    code('    purpose     = "SkyDock Pro lid — crack reported after drop",')
    code(")")
    time.sleep(0.8)

    ctx = Context(
        principal=principal,
        session_name="ticket-7502",
        purpose="SkyDock Pro lid — crack reported after drop",
    )
    _all_contexts.append(ctx)
    result("context_id", ctx.id)
    result("session   ", ctx.session_name)
    time.sleep(0.5)

    section("Log what happens on the call")
    entries = [
        ("Customer says SkyDock Pro lid cracked after a single drop from desk height.",
         None),
        ("Customer is on enterprise plan — priority escalated to P1.",
         None),
        ("Customer sent photo — crack runs along the hinge seam.",
         _make_image("CRACKED", (50, 50, 160))),
    ]

    code("info = Information(context_id=ctx.id)")
    blank()
    info = Information(context_id=ctx.id)

    for text, img in entries:
        short = (text[:55] + "…") if len(text) > 56 else text
        if img is not None:
            code(f'info.log(text="{short}", image=<photo>)')
        else:
            code(f'info.log(text="{short}")')
        if img is not None:
            info.log(text=text, image=img)
        else:
            info.log(text=text)
        time.sleep(0.5)

    blank()
    section("Commit to ApertureDB — no model calls, instant")
    code("commit_id = memory.commit(ctx, info)")
    commit_id = memory.commit(ctx, info)
    blank()
    result("commit_id", commit_id[:24] + "…")
    result("stored   ", "2 text entries  ·  1 text+image entry  ·  0 model calls")
    time.sleep(1)

    web_ui_prompt()
    print(f"  {DIM}  FindImage → 4 defect photos now in ApertureDB{RESET}")
    print(f"  {DIM}  FindEntity with_class NexusContext → 4 sessions in the graph{RESET}")


# ── Act 3 — Context ───────────────────────────────────────────────────────────

def act_context(memory):
    banner("CONTEXT", "Every session, every agent — one search")

    section("Search across all sessions in the organization")
    reveal(
        f"  {DIM}No session ID required. Filter by org — Memory handles the rest.{RESET}",
        indent="", pause=0.5,
    )
    blank()
    code('results = memory.search(filters={"organization": "acme-corp"})')
    results = memory.search(filters={"organization": ORG})
    blank()

    text_r  = [r for r in results if r.modality == "text"]
    image_r = [r for r in results if r.modality == "image"]
    users   = {r.user_id for r in results}
    sessions = {r.session_id for r in results}

    ok(
        f"{len(results)} entries  ·  "
        f"{len(text_r)} text  ·  {len(image_r)} images  ·  "
        f"{len(users)} contributors  ·  {len(sessions)} sessions\n"
    )
    time.sleep(0.3)

    agent_names = {uid: uname for uid, uname in SEED_AGENTS}
    agent_names[DEMO_USER] = "You (presenter)"

    seen_users = set()
    for r in results:
        uid = r.user_id or "unknown"
        if uid in seen_users:
            continue
        seen_users.add(uid)
        name  = agent_names.get(uid, uid)
        count = sum(1 for x in results if x.user_id == uid)
        reveal(
            f"    {GREEN}▸{RESET}  {BOLD}{name}{RESET}"
            f"  {DIM}{count} entries  ·  org: {ORG}{RESET}",
            indent="", pause=0.4,
        )

    time.sleep(1.2)

    section("Context wraps every entry with identity, session, and purpose")
    reveal(
        f"  {DIM}No joins, no foreign keys — it's graph-native.{RESET}",
        indent="", pause=0.5,
    )
    reveal(
        f"  {DIM}Permissions, provenance, and routing all follow automatically.{RESET}",
        indent="", pause=0.8,
    )


# ── Act 4 — Cognition ─────────────────────────────────────────────────────────

def act_cognition(memory):
    banner("COGNITION", "Pluggable models  ·  Semantic search  ·  Any embedding")

    section("CLIP: text query  →  ranked image results")
    reveal(
        f"  {DIM}Text and images share the same embedding space.{RESET}",
        indent="", pause=0.4,
    )
    reveal(
        f"  {DIM}One natural-language query finds matching images.{RESET}",
        indent="", pause=0.6,
    )
    blank()

    query = "cracked or scratched product surface"
    code("clip_memory = Memory()")
    code('clip_memory._cfg.models.image_embedding = "ViT-B/16"')
    blank()
    code(f'results = clip_memory.search(')
    code(f'    query    = "{query}",')
    code(f'    modality = "image",')
    code(f')')
    blank()
    time.sleep(0.8)

    clip_mem = _make_clip_memory()
    img_results = clip_mem.search(
        query=query,
        modality="image",
        embedding_model="ViT-B/16",
        k=10,
    )

    ok(f'"{query}"  →  {len(img_results)} image result(s)\n')
    for r in img_results:
        bar_len = int(r.score * 40)
        bar = f"{GREEN}{'█' * bar_len}{RESET}{DIM}{'░' * (40 - bar_len)}{RESET}"
        reveal(
            f"    {GREEN}[image]{RESET}  score: {BOLD}{r.score:.3f}{RESET}  {bar}",
            indent="", pause=0.4,
        )
    time.sleep(1.2)

    blank()
    section("search_contexts()  —  find sessions by purpose")
    reveal(
        f"  {DIM}Contexts are graph nodes. Embed their purpose — search by meaning.{RESET}",
        indent="", pause=0.6,
    )
    blank()
    code('ctx_results = memory.search_contexts("SkyDock surface defect")')
    blank()
    time.sleep(0.5)

    ctx_results = clip_mem.search_contexts(
        "SkyDock surface defect",
        embedding_model="ViT-B/16",
        k=10,
    )

    ok(f"{len(ctx_results)} context(s) matched  ·  ranked by semantic similarity\n")
    for r in ctx_results[:4]:
        short = (r.purpose[:58] + "…") if r.purpose and len(r.purpose) > 59 else (r.purpose or "—")
        reveal(
            f"    {GREEN}▸{RESET}  score: {BOLD}{r.score:.3f}{RESET}"
            f"  {DIM}{short}{RESET}",
            indent="", pause=0.4,
        )
    time.sleep(1.2)

    blank()
    section("Any model plugs in here — one config change")
    models = [
        ("Vision / multimodal", "CLIP  ViT-B/16  ·  ViT-L/14"),
        ("Dense text",          "BGE-M3  ·  E5-large"),
        ("OpenAI",              "text-embedding-3-small  ·  text-embedding-3-large"),
        ("Local / air-gapped",  "Ollama  ·  any HuggingFace model"),
    ]
    for category, examples in models:
        reveal(
            f"    {GREEN}▸{RESET}  {BOLD}{category}{RESET}  {DIM}{examples}{RESET}",
            indent="", pause=0.4,
        )
    time.sleep(1)

    blank()
    reveal(
        f"  {DIM}Switch models in aperture_nexus.json — stored data stays intact.{RESET}",
        indent="", pause=1.0,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_stack()
    cont("begin")

    from aperture_nexus import NexusAdmin, Memory

    admin  = NexusAdmin()
    memory = Memory()

    # Silent seed — pre-load org knowledge before the demo starts
    print(f"  {DIM}Seeding organization knowledge (3 prior reports)…{RESET}", flush=True)
    try:
        seed_knowledge(admin, memory)
        print(f"  {GREEN}✓{RESET}  3 prior support sessions loaded  ·  org: {ORG}")
    except Exception as e:
        print(f"\n  {YELLOW}Seed failed: {e}{RESET}")
        sys.exit(1)

    # Demo principal for live act
    print(f"  {DIM}Creating demo principal…{RESET}", flush=True)
    try:
        api_key = _create_principal(admin, DEMO_USER, "Demo Support Agent")
        _all_principals.append(DEMO_USER)
        principal = memory.authenticate(user_id=DEMO_USER, api_key=api_key)
    except Exception as e:
        print(f"\n  {YELLOW}Auth failed: {e}{RESET}")
        cleanup(admin, memory)
        sys.exit(1)

    print(f"  {GREEN}✓{RESET}  Ready  ·  4 principals  ·  org: {ORG}")
    time.sleep(1)

    try:
        act_knowledge(memory)
        cont("Act 2 — Memory: new call comes in")

        act_memory(memory, principal)
        cont("Act 3 — Context: cross-session retrieval")

        act_context(memory)
        cont("Act 4 — Cognition: semantic search + pluggable models")

        act_cognition(memory)

    finally:
        blank()
        print(f"  {DIM}Cleaning up demo data…{RESET}", flush=True)
        cleanup(admin, memory)
        print(f"  {GREEN}✓{RESET}  Done.  Stack still running at http://localhost:8087")
        blank()

    w = W()
    blank()
    print(f"  {BOLD}{CYAN}{'═' * (w - 4)}{RESET}")
    print(f"  {BOLD}  github.com/aperturedata/aperture-nexus{RESET}")
    print(f"  {BOLD}{CYAN}{'═' * (w - 4)}{RESET}")
    blank()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        blank()
        print(f"  {DIM}Interrupted.{RESET}")
        sys.exit(0)
