#!/usr/bin/env bash
# record.sh — drives the aperture-nexus hello-world demo for asciinema
#
# Usage (from the repo root):
#   COLUMNS=88 LINES=28 asciinema rec demo/demo.cast \
#       -c "bash demo/record.sh" -t "aperture-nexus hello world" --overwrite
#
# Then convert to GIF:
#   agg --speed 1.25 --font-size 14 demo/demo.cast demo/demo.gif

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$DEMO_DIR/.." && pwd)"
cd "$REPO_DIR"

# Point aperture-nexus and adb-nexus at the local Docker ApertureDB.
export APERTUREDB_JSON='{"host":"localhost","port":55556,"username":"admin","password":"admin","use_ssl":false}'

# Keep a clean .env for this demo (written by adb-nexus init).
export NEXUS_ENV_FILE="demo/.env.demo"
rm -f "$NEXUS_ENV_FILE"

# ── Helpers ───────────────────────────────────────────────────────────────────

BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[0;36m'; RESET='\033[0m'

run() {
    local cmd="$1"
    echo -en "${BOLD}${CYAN}\$ ${RESET}"
    for ((i=0; i<${#cmd}; i++)); do
        echo -n "${cmd:$i:1}"
        sleep 0.04
    done
    echo
    sleep 0.3
    eval "$cmd"
    sleep 0.6
}

comment() {
    echo
    sleep 0.3
    echo -e "${DIM}# $*${RESET}"
    sleep 0.4
}

wait_for_aperturedb() {
    echo -en "${DIM}  waiting for ApertureDB"
    for i in $(seq 1 30); do
        if python3 -c "
import json, os
from aperturedb.Connector import Connector
cfg = json.loads(os.environ['APERTUREDB_JSON'])
c = Connector(host=cfg['host'], port=cfg['port'],
              user=cfg['username'], password=cfg['password'],
              use_ssl=False)
c.query([{'GetStatus': {}}])
" 2>/dev/null; then
            echo -e "${RESET}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo -e " timed out${RESET}"
    return 1
}

# ── Scene 1: Start ApertureDB ─────────────────────────────────────────────────
clear
sleep 0.8

comment "Start ApertureDB with Docker Compose"
run "ADB_PORT=55556 DB_TCP_CN=localhost DB_HTTP_CN=localhost docker compose up --detach"

wait_for_aperturedb

# Silently delete any existing demo principal so init always creates fresh.
python3 -c "
import os
try:
    from aperture_nexus import NexusAdmin
    NexusAdmin().delete_principal(user_id=os.environ.get('USER', 'demo-user'))
except Exception:
    pass
" 2>/dev/null
sleep 0.4

comment "Verify the connection"
run "adb-nexus validate"

# ── Scene 2: Initialise aperture-nexus ───────────────────────────────────────

comment "Create a principal and write NEXUS_API_KEY to .env"
run "adb-nexus init --defaults --env-file demo/.env.demo"

# ── Scene 3: Hello world ──────────────────────────────────────────────────────

comment "Run the hello-world demo"
export PYTHONWARNINGS=ignore
NEXUS_ENV_FILE="demo/.env.demo" run "python3 demo/hello_world.py"

# ── Fin ───────────────────────────────────────────────────────────────────────
echo
echo -e "${DIM}  ApertureDB WebUI → http://localhost:8087  (inspect stored memories)${RESET}"
echo
sleep 2
