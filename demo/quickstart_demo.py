"""
quickstart_demo.py — aperture-nexus walkthrough demo.

Run interactively (pauses at each step, recommended):

    docker compose --profile demo run --rm nexus-demo

Ctrl+C at any point cleans up all demo data from ApertureDB before exiting.
After the demo, explore stored data at http://localhost:8087 (ApertureDB web UI).
"""

import shutil
import signal
import socket
import sys
import time

from aperture_nexus import Context, Information, Memory, NexusAdmin
from aperture_nexus.exceptions import NexusStorageError

# ── Module-level state — always accessible from the signal handler ────────────

_admin: NexusAdmin | None = None
_memory: Memory | None = None
_context_ids: list[str] = []          # tracked as we create them

DEMO_USER = "nexus-demo-agent"
DEMO_ORG  = "acme-corp"
DEMO_DEPT = "support"

# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup(silent: bool = False) -> None:
    """Remove all demo data from ApertureDB and delete the demo principal."""
    if not silent:
        print("  Cleaning up demo data…", flush=True)

    if _memory is not None:
        # Remove contexts we tracked during this run
        for ctx_id in _context_ids:
            try:
                _memory.remove(ctx_id)
            except Exception:
                pass
        # Sweep for any leftover data from a previous interrupted run
        try:
            results = _memory.search(filters={"user_id": DEMO_USER})
            for ctx_id in {r.context_id for r in results if r.context_id}:
                try:
                    _memory.remove(ctx_id)
                except Exception:
                    pass
        except Exception:
            pass

    if _admin is not None:
        try:
            _admin.delete_principal(user_id=DEMO_USER)
        except Exception:
            pass

    if not silent:
        print("  Done.", flush=True)


def _on_signal(sig, frame) -> None:
    print()
    print("  Interrupted.")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ── Terminal helpers ──────────────────────────────────────────────────────────

W = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100)

BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _c(code, text):
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def blank():
    print()


def hr():
    print(_c(DIM, "  " + "─" * (W - 2)))


def banner():
    blank()
    print(_c(BOLD, "  ┌" + "─" * (W - 4) + "┐"))
    print(_c(BOLD, "  │" + "  aperture-nexus  ·  walkthrough demo".center(W - 4) + "│"))
    print(_c(BOLD, "  │" + "  Knowledge  ·  Memory  ·  Context".center(W - 4) + "│"))
    print(_c(BOLD, "  └" + "─" * (W - 4) + "┘"))
    blank()


def section(title):
    blank()
    hr()
    print(_c(BOLD + CYAN, f"  {title}"))
    hr()
    blank()


def ok(msg):
    print(_c(GREEN, "  ✓") + f"  {msg}")


def show(label, value):
    print(f"    {_c(DIM, label + ':')}  {value}")


def code_block(lines):
    blank()
    print(_c(DIM, "  ┌─ call " + "─" * (W - 10) + "┐"))
    for line in lines:
        print(_c(DIM, "  │ ") + f"  {_c(BOLD, line)}")
    print(_c(DIM, "  └" + "─" * (W - 4) + "┘"))
    blank()


def json_block(lines):
    blank()
    for line in lines:
        print(f"    {_c(DIM, line)}")
    blank()


def pause(label="next step"):
    blank()
    try:
        input(_c(BOLD, f"  ▸  Press Enter to {label}…"))
    except EOFError:
        blank()
        print("  stdin is not interactive.")
        print("  Run: docker compose --profile demo run --rm nexus-demo")
        cleanup()
        sys.exit(0)
    blank()


# ── Wait for ApertureDB ───────────────────────────────────────────────────────

def _tcp_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def wait_for_aperturedb(host="lenz", port=55551, max_wait=120, interval=2):
    print(f"  Waiting for ApertureDB", end="", flush=True)
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if _tcp_ready(host, port):
            print(_c(GREEN, " ready"), flush=True)
            time.sleep(3)
            return
        print(".", end="", flush=True)
        time.sleep(interval)
    print("\n  ✗  ApertureDB did not start in time.")
    sys.exit(1)


# ── Principal management ──────────────────────────────────────────────────────

def provision_principal() -> str:
    """Return a fresh api_key. Wipes any leftover data from a prior run first."""
    global _admin, _memory

    # Sweep leftover data before (re)creating the principal
    if _memory is not None:
        try:
            results = _memory.search(filters={"user_id": DEMO_USER})
            for ctx_id in {r.context_id for r in results if r.context_id}:
                try:
                    _memory.remove(ctx_id)
                except Exception:
                    pass
        except Exception:
            pass

    try:
        return _admin.create_principal(
            user_id=DEMO_USER,
            user_name="Demo Support Agent",
            organization=DEMO_ORG,
            department=DEMO_DEPT,
        )
    except NexusStorageError:
        _admin.delete_principal(user_id=DEMO_USER)
        return _admin.create_principal(
            user_id=DEMO_USER,
            user_name="Demo Support Agent",
            organization=DEMO_ORG,
            department=DEMO_DEPT,
        )


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    global _admin, _memory, _context_ids

    banner()
    wait_for_aperturedb()

    _admin  = NexusAdmin()
    api_key = provision_principal()
    _memory = Memory()
    principal = _memory.authenticate(user_id=DEMO_USER, api_key=api_key)

    ok(f"Connected  ·  principal: {DEMO_USER}  ·  org: {DEMO_ORG}")
    blank()
    print(_c(DIM, "  (Ctrl+C at any point will clean up and exit)"))

    # ── Scenario ──────────────────────────────────────────────────────────────

    section("Scenario")

    print("  An enterprise support agent takes a call, captures context,")
    print("  and commits it to memory. A second agent opens a follow-up")
    print("  session and retrieves that context — without being told")
    print("  which session, context, or agent captured it.")
    blank()
    print(_c(DIM,
        "    Principal\n"
        "    │\n"
        "    ├─ Context: Session 1  →  4 text entries  ─┐\n"
        "    │                                           │ nexus_link\n"
        "    └─ Context: Session 2  →  1 text entry   ←─┘\n"
        "\n"
        "    search(filters={\"organization\": \"acme-corp\"})\n"
        "    └─ returns all entries across both sessions"
    ))

    pause("start Session 1")

    # ── Session 1 ─────────────────────────────────────────────────────────────

    section("Session 1  ·  Agent 1 captures context")

    ctx1 = Context(
        principal=principal,
        session_name="acme-tkt-1042-initial",
        purpose="Support call — /export endpoint timeout",
    )
    _context_ids.append(ctx1.id)

    show("session_name", ctx1.session_name)
    show("context_id  ", ctx1.id)
    blank()

    entries = [
        "Customer reports timeout on /export when file size exceeds 500 MB.",
        "Customer is on the enterprise plan. SLA: 4-hour response.",
        "Root cause: worker_timeout set to 30 s — below the file transfer window.",
        "Fix: increase worker_timeout to 120 s in worker.conf.",
    ]

    code_block([
        "info = Information(context_id=ctx1.id)",
        "info.log(text=...)  # ×4",
        "memory.commit(ctx1, info)",
    ])

    info1 = Information(context_id=ctx1.id)
    for entry in entries:
        info1.log(text=entry)
        print(f"    {_c(DIM, '·')}  {entry}")

    _memory.commit(ctx1, info1)
    blank()
    ok(f"Committed  ·  1 NexusSession  ·  1 NexusContext  ·  {len(entries)} Blobs")

    pause("open Session 2")

    # ── Session 2 ─────────────────────────────────────────────────────────────

    section("Session 2  ·  Agent 2 retrieves context")

    ctx2 = Context(
        principal=principal,
        session_name="acme-tkt-1042-followup",
        purpose="Follow-up: verify resolution of TKT-1042",
    )
    _context_ids.append(ctx2.id)

    show("session_name", ctx2.session_name)
    show("context_id  ", ctx2.id)
    blank()

    # Commit a minimal entry so ctx2's NexusContext entity exists before connect()
    info2 = Information(context_id=ctx2.id)
    info2.log(text="Follow-up session opened. Verifying resolution of TKT-1042.")
    _memory.commit(ctx2, info2)
    ok("Follow-up session committed")
    blank()

    code_block(['memory.search(filters={"organization": "acme-corp"})'])

    results = _memory.search(filters={"organization": DEMO_ORG})

    if results:
        ok(f"{len(results)} entries retrieved across all sessions:\n")
        for r in results:
            if r.text:
                print(f"    {_c(GREEN, '▸')}  {r.text}")
    else:
        ok("Entries committed. (Search returned empty — DB may still be indexing.)")

    pause("link the sessions in the graph")

    # ── Connect ───────────────────────────────────────────────────────────────

    section("Graph  ·  linking sessions with a nexus_link")

    code_block(['memory.connect(ctx1, ctx2, relationship="followed_by")'])

    _memory.connect(ctx1, ctx2, relationship="followed_by")
    ok("Edge created")
    blank()
    print(_c(DIM,
        "    NexusContext [acme-tkt-1042-initial]\n"
        "        └─[followed_by]─▸  NexusContext [acme-tkt-1042-followup]"
    ))

    pause("verify in the web UI")

    # ── UI verification ───────────────────────────────────────────────────────

    section("Verify in the ApertureDB web UI")

    print("  1. Open  http://localhost:8087")
    print("  2. Log in:  admin  /  admin")
    print("  3. Navigate to  Custom Query")
    print("  4. Paste and run:\n")

    json_block([
        "[",
        "  {",
        '    "FindEntity": {',
        '      "with_class": "NexusContext",',
        '      "results": { "all_properties": true }',
        "    }",
        "  }",
        "]",
    ])

    print("  You should see 2 NexusContext entities — one per session above.")
    print("  Each has context_id, session_id, user_id, organization, and purpose.")
    blank()
    print("  Also try  FindBlob  to see the 5 text entries,")
    print("  and  FindConnection  to see the followed_by edge.")

    pause("clean up and finish")

    # ── Done ─────────────────────────────────────────────────────────────────

    section("Done")

    cleanup()
    blank()
    print("  To stop the ApertureDB stack:")
    print("    docker compose down          # stop (data preserved)")
    print("    docker compose down -v       # stop and wipe all data")
    blank()
    print("  Next steps:")
    print("    git clone https://github.com/aperture-data/aperture-nexus")
    print("    cd aperture-nexus && pip install .")
    print("    docker compose up -d         # start ApertureDB stack")
    print("    adb-nexus init               # create your principal")
    blank()
    print("  Docs: https://github.com/aperture-data/aperture-nexus")
    blank()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        pass   # _on_signal already handled it
