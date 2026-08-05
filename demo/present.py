"""
present.py — aperture-nexus presentation demo.

Knowledge → Memory → Context → Cognition

Four auto-running sections with manual Continue prompts between them.
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


# ── Damage photo generators ───────────────────────────────────────────────────
#
# Each pattern draws a visually distinct damage type so CLIP can separate them.
# Text labels in the image reinforce the CLIP match on targeted queries.

def _make_damage_image(label, base, pattern, size=(420, 320)):
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img  = PILImage.new("RGB", size, color=base)
    draw = ImageDraw.Draw(img)
    W, H = size

    if pattern == "scratch":
        # Parallel diagonal silver lines + one prominent scratch
        for x in range(-H, W + H, 22):
            draw.line([(x, 0), (x + H, H)], fill=(140, 145, 162), width=1)
        draw.line([(28, 58), (W-28, H-58)], fill=(182, 190, 212), width=6)
        draw.line([(30, 58), (W-26, H-56)], fill=(228, 232, 245), width=2)
        draw.rectangle([10, 10, W-11, H-11], outline=(158, 164, 188), width=3)

    elif pattern == "flaking":
        # Corner paint chips exposing lighter substrate + scattered spots
        sub = tuple(min(255, c + 88) for c in base)
        chips = [(12,12,64,40),(W-76,12,64,40),(12,H-52,64,40),(W-76,H-52,64,40)]
        for cx, cy, cw, ch in chips:
            draw.rectangle([cx, cy, cx+cw, cy+ch], fill=sub)
            draw.polygon(
                [(cx, cy+ch),(cx+14, cy+ch-11),(cx+30, cy+ch+6),
                 (cx+48, cy+ch-8),(cx+cw, cy+ch)],
                fill=base,
            )
        for x, y, r in [(118,88,9),(198,200,8),(292,112,10),(158,222,7),(332,186,6),(212,132,5)]:
            draw.ellipse([x-r,y-r,x+r,y+r], fill=sub)
        draw.rectangle([10, 10, W-11, H-11], outline=(192, 175, 148), width=3)

    elif pattern == "discolor":
        # Amber-brown blotch — concentric ellipses fading from center
        cx, cy = int(W * 0.52), int(H * 0.44)
        for r in range(105, 5, -5):
            t = 1.0 - r / 105.0
            c = (
                int(base[0] + (212 - base[0]) * t * 0.82),
                int(base[1] + (118 - base[1]) * t * 0.66),
                int(base[2] + (16  - base[2]) * t * 0.48),
            )
            draw.ellipse([cx-r, cy-int(r*0.62), cx+r, cy+int(r*0.62)], fill=c)
        draw.rectangle([10, 10, W-11, H-11], outline=(158, 145, 188), width=3)

    elif pattern == "crack":
        # Jagged fracture line with drop shadow
        pts = [(30,148),(88,96),(134,162),(190,82),(246,150),(304,72),(360,132),(W-22,120)]
        draw.line([(x+5,y+5) for x,y in pts], fill=(15, 17, 28), width=7)
        draw.line(pts, fill=(198, 208, 230), width=4)
        draw.line(pts, fill=(238, 242, 255), width=1)
        draw.rectangle([10, 10, W-11, H-11], outline=(168, 174, 204), width=3)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26,
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((W // 2, H - 22), label, fill=(255, 255, 255), font=font, anchor="mm")

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


# ── Seed data ─────────────────────────────────────────────────────────────────
#
# Three support tickets with 3 photos each, committed via process_and_commit()
# so CLIP embeddings are indexed before the demo starts.

PRIOR_REPORTS = [
    {
        "agent":   "alice-support",
        "session": "ticket-7101",
        "purpose": "Surface scratch damage on SkyDock Pro lid — pre-shipping",
        "notes": [
            "Customer received SkyDock Pro with a diagonal scratch running the full length of the lid.",
            "Scratch depth varies — heavier at the entry and exit points, tool or edge contact.",
            "Unit was new in sealed original box. Damage is pre-shipping. Three photos attached.",
        ],
        "photos": [
            ("scratch abrasion",     (50, 55, 70), "scratch"),
            ("scratch mark",         (46, 51, 66), "scratch"),
            ("surface scratch",      (54, 59, 74), "scratch"),
        ],
    },
    {
        "agent":   "bob-support",
        "session": "ticket-7245",
        "purpose": "Paint flaking at corners — coating defect, SkyDock Pro batch A2",
        "notes": [
            "Paint flaking at all four corners, exposing bare aluminum underneath.",
            "Customer reports flaking is progressing inward — started at corners two weeks ago.",
            "Third identical report this month, all from batch A2-2025. QA team notified.",
        ],
        "photos": [
            ("paint flaking chips",  (78, 54, 28), "flaking"),
            ("coating peeling",      (82, 58, 32), "flaking"),
            ("flaking edge damage",  (74, 50, 24), "flaking"),
        ],
    },
    {
        "agent":   "carol-support",
        "session": "ticket-7389",
        "purpose": "Heat discoloration on SkyDock Pro top surface — shipping damage suspected",
        "notes": [
            "Amber-brown discoloration covers approximately 40% of the top surface.",
            "Customer suspects heat exposure during shipping — unit stored in vehicle.",
            "Pattern matches tickets 7101 and 7245. Same batch. Escalated to engineering.",
        ],
        "photos": [
            ("heat discoloration",   (44, 40, 74), "discolor"),
            ("heat stain amber",     (48, 44, 78), "discolor"),
            ("thermal burn mark",    (40, 36, 70), "discolor"),
        ],
    },
]


def seed_knowledge(admin, memory):
    """Pre-load 3 prior support reports with 3 photos each into org memory (CLIP-indexed)."""
    from aperture_nexus import Context, Information

    clip_mem = _make_clip_memory()
    agent_map = {uid: uname for uid, uname in SEED_AGENTS}

    # Pre-generate all PIL images so the CLIP commits don't stall on image I/O
    print(f"  {DIM}  Pre-generating damage photos…{RESET}", end=" ", flush=True)
    for report in PRIOR_REPORTS:
        report["_images"] = [
            _make_damage_image(label, color, pattern)
            for label, color, pattern in report["photos"]
        ]
    print(f"{GREEN}done{RESET}")

    # Warm up CLIP so the first ticket doesn't absorb the model-load cost
    print(f"  {DIM}  Loading CLIP model…{RESET}", end=" ", flush=True)
    t_clip = time.time()
    try:
        clip_mem.search(query="warmup", modality="text",
                        embedding_model="ViT-B/16", k=1)
    except Exception:
        pass
    print(f"{GREEN}done{RESET}  {DIM}{time.time() - t_clip:.1f}s{RESET}")

    n = len(PRIOR_REPORTS)
    for i, report in enumerate(PRIOR_REPORTS, 1):
        user_id   = report["agent"]
        user_name = agent_map[user_id]
        bar_done  = "█" * i
        bar_left  = "░" * (n - i)

        print(
            f"  {DIM}  [{bar_done}{bar_left}] {i}/{n}  {user_name:<14}{RESET}",
            end=" ", flush=True,
        )
        t0 = time.time()

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
        for note in report["notes"]:
            info.log(text=note)
        for (photo_label, _, _), img_bytes in zip(report["photos"], report["_images"]):
            info.log(
                text=f"Customer photo: {photo_label.strip().lower()}",
                image=img_bytes,
            )

        clip_mem.process_and_commit(ctx, info)
        print(f"{GREEN}✓{RESET}  {DIM}{time.time() - t0:.1f}s{RESET}")


def _make_clip_memory():
    from aperture_nexus import Memory as _Mem
    m = _Mem()
    m._cfg.models.text_embedding  = "ViT-B/16"
    m._cfg.models.image_embedding = "ViT-B/16"
    return m


# ── Schema graph ─────────────────────────────────────────────────────────────

SCHEMA_PATH = "/tmp/nexus_schema"

def open_schema_graph():
    """Render the live ApertureDB schema to a PNG and open it."""
    try:
        from aperturedb.CommonLibrary import create_connector
        from aperturedb.Utils import Utils
        conn  = create_connector()
        utils = Utils(conn)
        src   = utils.visualize_schema(filename=SCHEMA_PATH, format="png")
        png   = SCHEMA_PATH + ".png"
        subprocess.Popen(["xdg-open", png],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  {GREEN}✓{RESET}  Schema diagram → {DIM}{png}{RESET}")
    except Exception as e:
        print(f"  {YELLOW}⚠  Schema graph skipped: {e}{RESET}")


# ── Knowledge ─────────────────────────────────────────────────────────────────

def show_knowledge(memory):
    banner("KNOWLEDGE", "What your organization already knows")

    section("Before this call — 3 prior sessions already in memory")
    reveal(
        f"  {DIM}Three support engineers handled similar cases this month.{RESET}",
        indent="", pause=0.5,
    )
    blank()
    code('results = memory.search(filters={"organization": "acme-corp"}, k=50)')
    results = memory.search(filters={"organization": ORG}, k=50)
    blank()

    text_r  = [r for r in results if r.modality == "text"]
    image_r = [r for r in results if r.modality == "image"]
    sessions = {r.session_id for r in results}

    ok(
        f"{len(results)} entries  ·  "
        f"{len(text_r)} notes  ·  {len(image_r)} photos  ·  "
        f"{len(sessions)} sessions\n"
    )
    time.sleep(0.3)

    # One line per session
    by_session: dict = {}
    for r in results:
        by_session.setdefault(r.session_id, []).append(r)

    agent_names = {uid: uname for uid, uname in SEED_AGENTS}
    damage_type = {
        "alice-support": "scratch",
        "bob-support":   "flaking",
        "carol-support": "discolor",
    }

    for sid, entries in list(by_session.items())[:3]:
        uid   = entries[0].user_id or "unknown"
        name  = agent_names.get(uid, uid)
        dtype = damage_type.get(uid, "unknown")
        texts = [e for e in entries if e.text]
        note  = texts[0].text if texts else ""
        short = (note[:64] + "…") if len(note) > 65 else note
        imgs  = sum(1 for e in entries if e.modality == "image")
        reveal(
            f"    {GREEN}▸{RESET}  {BOLD}{name}{RESET}  "
            f"{DIM}{len(texts)} notes · {imgs} photos · {dtype}{RESET}",
            indent="", pause=0.2,
        )
        reveal(f"       {DIM}{short}{RESET}", indent="", pause=0.5)

    time.sleep(0.8)
    web_ui_prompt()
    print(f"  {DIM}  FindBlob → read the support notes for each ticket{RESET}")
    print(f"  {DIM}  FindImage → browse all 9 defect photos side by side{RESET}")
    blank()
    open_schema_graph()
    print(f"  {DIM}  Schema diagram shows NexusPrincipal → NexusContext → Descriptors{RESET}")


# ── Memory ────────────────────────────────────────────────────────────────────

def show_memory(memory, principal):
    from aperture_nexus import Context, Information

    banner("MEMORY", "A new call comes in — capture it")

    section("Open a session for the new ticket")
    code("ctx = Context(")
    code(f'    principal   = principal,')
    code('    session_name= "ticket-7502",')
    code('    purpose     = "SkyDock Pro lid cracked after drop — P1 enterprise",')
    code(")")
    time.sleep(0.8)

    ctx = Context(
        principal=principal,
        session_name="ticket-7502",
        purpose="SkyDock Pro lid cracked after drop — P1 enterprise",
    )
    _all_contexts.append(ctx)
    result("context_id", ctx.id)
    result("session   ", ctx.session_name)
    time.sleep(0.5)

    section("Log everything that happens on the call")
    entries = [
        ("Lid cracked on first drop from desk height — 75 cm fall, single impact.",
         None),
        ("Enterprise customer — 500-unit deployment. Priority escalated to P1.",
         None),
        ("Customer sent two photos: full crack and hinge detail.",
         _make_damage_image("CRACK  full",  (35, 38, 58), "crack")),
        (None,
         _make_damage_image("CRACK  hinge", (32, 35, 55), "crack")),
    ]

    code("info = Information(context_id=ctx.id)")
    blank()
    info = Information(context_id=ctx.id)

    for text, img in entries:
        if text and img is not None:
            short = (text[:54] + "…") if len(text) > 55 else text
            code(f'info.log(text="{short}", image=<photo>)')
            info.log(text=text, image=img)
        elif text:
            short = (text[:54] + "…") if len(text) > 55 else text
            code(f'info.log(text="{short}")')
            info.log(text=text)
        else:
            code('info.log(image=<second photo>)')
            info.log(image=img)
        time.sleep(0.45)

    blank()
    section("Commit — raw storage, no model calls, instant")
    code("commit_id = memory.commit(ctx, info)")
    commit_id = memory.commit(ctx, info)
    blank()
    result("commit_id", commit_id[:24] + "…")
    result("stored   ", "2 text entries  ·  1 text+image  ·  1 image  ·  0 model calls")
    time.sleep(1)

    web_ui_prompt()
    print(f"  {DIM}  FindImage → 11 defect photos now stored across 4 sessions{RESET}")
    print(f"  {DIM}  FindEntity with_class NexusContext → 4 sessions in the graph{RESET}")


# ── Context ───────────────────────────────────────────────────────────────────

def show_context(memory):
    banner("CONTEXT", "Every session, every contributor — one search")

    section("Pull all sessions from the organization")
    reveal(
        f"  {DIM}No session ID needed. Memory knows who captured what and why.{RESET}",
        indent="", pause=0.5,
    )
    blank()
    code('results = memory.search(filters={"organization": "acme-corp"}, k=50)')
    results = memory.search(filters={"organization": ORG}, k=50)
    blank()

    text_r   = [r for r in results if r.modality == "text"]
    image_r  = [r for r in results if r.modality == "image"]
    users    = {r.user_id for r in results}
    sessions = {r.session_id for r in results}

    ok(
        f"{len(results)} entries  ·  "
        f"{len(text_r)} notes  ·  {len(image_r)} photos  ·  "
        f"{len(users)} contributors  ·  {len(sessions)} sessions\n"
    )
    time.sleep(0.3)

    agent_names = {uid: uname for uid, uname in SEED_AGENTS}
    agent_names[DEMO_USER] = "You (new ticket)"

    seen: set = set()
    for r in results:
        uid = r.user_id or "unknown"
        if uid in seen:
            continue
        seen.add(uid)
        name  = agent_names.get(uid, uid)
        count = sum(1 for x in results if x.user_id == uid)
        reveal(
            f"    {GREEN}▸{RESET}  {BOLD}{name}{RESET}"
            f"  {DIM}{count} entries  ·  org: {ORG}{RESET}",
            indent="", pause=0.4,
        )

    time.sleep(1.2)

    section("Context is graph-native — identity, session, and purpose travel with every entry")
    reveal(
        f"  {DIM}No joins. No foreign keys.{RESET}",
        indent="", pause=0.4,
    )
    reveal(
        f"  {DIM}Permissions, provenance, and routing follow automatically.{RESET}",
        indent="", pause=0.8,
    )


# ── Cognition ─────────────────────────────────────────────────────────────────

def show_cognition(memory):
    banner("COGNITION", "Pluggable models  ·  Semantic search  ·  Any embedding")

    clip_mem = _make_clip_memory()

    section("CLIP: three queries, three damage types — same index, different results")
    reveal(
        f"  {DIM}Text and images share the same vector space.{RESET}",
        indent="", pause=0.4,
    )
    reveal(
        f"  {DIM}k=3 — each query pulls the photos that match best.{RESET}",
        indent="", pause=0.6,
    )
    blank()

    agent_names = {uid: uname for uid, uname in SEED_AGENTS}

    queries = [
        ("surface scratch abrasion on product", "scratch"),
        ("paint flaking chips coating failure", "flaking"),
        ("heat discoloration thermal stain",    "discolor"),
    ]

    for query, expected in queries:
        code(f'memory.search(query="{query}", modality="image", k=3)')
        blank()
        time.sleep(0.5)

        img_results = clip_mem.search(
            query=query,
            modality="image",
            embedding_model="ViT-B/16",
            k=3,
        )

        ok(f'"{query}"\n')
        for r in img_results:
            bar_len = int(r.score * 32)
            bar = f"{GREEN}{'█' * bar_len}{RESET}{DIM}{'░' * (32 - bar_len)}{RESET}"
            name = agent_names.get(r.user_id or "", r.user_id or "unknown")
            reveal(
                f"    {GREEN}[image]{RESET}  score: {BOLD}{r.score:.3f}{RESET}"
                f"  {bar}  {DIM}{name}{RESET}",
                indent="", pause=0.35,
            )
        blank()
        time.sleep(0.8)

    blank()
    section("search_contexts()  —  sessions matched by purpose, not keywords")
    reveal(
        f"  {DIM}Context graph nodes have purpose embeddings. Query by meaning.{RESET}",
        indent="", pause=0.6,
    )
    blank()
    code('ctx_results = memory.search_contexts("SkyDock surface defect", k=5)')
    blank()
    time.sleep(0.5)

    ctx_results = clip_mem.search_contexts(
        "SkyDock surface defect",
        embedding_model="ViT-B/16",
        k=5,
    )

    ok(f"{len(ctx_results)} session(s) matched  ·  ranked by semantic similarity\n")
    for r in ctx_results:
        short = (r.purpose[:60] + "…") if r.purpose and len(r.purpose) > 61 else (r.purpose or "—")
        reveal(
            f"    {GREEN}▸{RESET}  score: {BOLD}{r.score:.3f}{RESET}"
            f"  {DIM}{short}{RESET}",
            indent="", pause=0.4,
        )
    time.sleep(1.2)

    blank()
    section("Any model plugs in here — one line in aperture_nexus.json")
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
        f"  {DIM}Switch models — stored data stays intact.{RESET}",
        indent="", pause=1.0,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_stack()
    cont("begin")

    from aperture_nexus import NexusAdmin, Memory

    admin  = NexusAdmin()
    memory = Memory()

    print(f"  {DIM}Seeding organization knowledge (3 tickets × 3 photos)…{RESET}", flush=True)
    try:
        seed_knowledge(admin, memory)
        print(f"  {GREEN}✓{RESET}  9 defect photos + notes loaded  ·  org: {ORG}")
    except Exception as e:
        print(f"\n  {YELLOW}Seed failed: {e}{RESET}")
        sys.exit(1)

    print(f"  {DIM}Creating demo principal…{RESET}", flush=True)
    try:
        api_key   = _create_principal(admin, DEMO_USER, "Demo Support Agent")
        _all_principals.append(DEMO_USER)
        principal = memory.authenticate(user_id=DEMO_USER, api_key=api_key)
    except Exception as e:
        print(f"\n  {YELLOW}Auth failed: {e}{RESET}")
        cleanup(admin, memory)
        sys.exit(1)

    print(f"  {GREEN}✓{RESET}  Ready  ·  4 principals  ·  org: {ORG}")
    time.sleep(1)

    try:
        show_knowledge(memory)
        cont("Memory: new ticket comes in")

        show_memory(memory, principal)
        cont("Context: cross-session retrieval")

        show_context(memory)
        cont("Cognition: semantic search")

        show_cognition(memory)

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
