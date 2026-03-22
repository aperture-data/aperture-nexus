#!/usr/bin/env bash
# aperture-nexus setup script
#
# Gets you from zero to a running ApertureDB + configured aperture-nexus.
# Safe to run multiple times — skips steps that are already done.
#
# Usage:
#   bash setup.sh              # full setup
#   bash setup.sh --no-docker  # skip ApertureDB (you're running it elsewhere)

set -euo pipefail

SKIP_DOCKER=false
for arg in "$@"; do
  case "$arg" in
    --no-docker) SKIP_DOCKER=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
step() { echo; echo "── $*"; }

# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
step "Checking Python version"

PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
  fail "Python not found. Install Python 3.10 or later and try again."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  fail "Python 3.10 or later is required (found $PY_VERSION)."
fi
ok "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# 2. Install aperture-nexus
# ---------------------------------------------------------------------------
step "Installing aperture-nexus"

if "$PYTHON" -c "import aperture_nexus" 2>/dev/null; then
  ok "aperture-nexus already installed"
else
  "$PYTHON" -m pip install --quiet aperture-nexus
  ok "aperture-nexus installed"
fi

# ---------------------------------------------------------------------------
# 3. ApertureDB via Docker Compose
# ---------------------------------------------------------------------------
if [ "$SKIP_DOCKER" = false ]; then
  step "Starting ApertureDB"

  if ! command -v docker &>/dev/null; then
    warn "Docker not found. Install Docker and re-run, or use --no-docker if ApertureDB is running elsewhere."
    warn "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
  fi

  if ! docker info &>/dev/null; then
    fail "Docker is not running. Start Docker and try again."
  fi

  if [ ! -f docker-compose.yml ]; then
    fail "docker-compose.yml not found. Run this script from the aperture-nexus directory."
  fi

  docker compose up -d --quiet-pull
  ok "ApertureDB started"

  # Wait for ApertureDB to accept connections (up to 30s)
  echo -n "   Waiting for ApertureDB to be ready"
  for i in $(seq 1 30); do
    if docker compose exec -T aperturedb true 2>/dev/null; then
      echo
      ok "ApertureDB is accepting connections"
      break
    fi
    echo -n "."
    sleep 1
    if [ "$i" -eq 30 ]; then
      echo
      fail "ApertureDB did not become ready within 30 seconds. Check 'docker compose logs aperturedb'."
    fi
  done
else
  ok "Skipping Docker (--no-docker)"
fi

# ---------------------------------------------------------------------------
# 4. Credentials / .env
# ---------------------------------------------------------------------------
step "Checking credentials"

if [ -f .env ]; then
  ok ".env already exists"
elif [ -f .env.example ]; then
  cp .env.example .env
  warn ".env created from .env.example — edit it with your ApertureDB credentials if needed"
else
  warn "No .env.example found. If you are using APERTUREDB_KEY, set it in your environment."
fi

# ---------------------------------------------------------------------------
# 5. aperture-nexus config
# ---------------------------------------------------------------------------
step "Checking aperture-nexus config"

if [ -f aperture_nexus.json ]; then
  ok "aperture_nexus.json already exists"
else
  echo "   Running 'adb-nexus init --defaults' to create a default config..."
  "$PYTHON" -m aperture_nexus.cli init --defaults
  ok "aperture_nexus.json created with defaults"
fi

# ---------------------------------------------------------------------------
# 6. Validate connection
# ---------------------------------------------------------------------------
step "Validating connection"

if "$PYTHON" -m aperture_nexus.cli validate; then
  ok "Connection validated"
else
  warn "Validation failed. Check your credentials and ApertureDB status."
  warn "Run 'adb-nexus validate' after fixing the issue."
  exit 1
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}Setup complete.${NC}"
echo
echo "  Try the quickstart:     python examples/quickstart.py"
echo "  Browse stored data:     http://localhost:8087  (ApertureDB web UI)"
echo "  Full docs:              docs/getting-started.md"
echo
