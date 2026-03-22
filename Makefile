# aperture-nexus development targets
#
# Usage:
#   make test              — unit tests (no live DB required)
#   make test-all          — everything (integration tests skip if no DB)
#   make test-integration  — integration tests only (requires live ApertureDB)
#   make test-fast         — unit tests, excluding slow ones
#   make test-cov          — unit tests with coverage report
#   make test-module m=test_client   — one module by name
#   make lint              — mypy type checking
#   make fmt               — autopep8 formatting

PYTHON   ?= python3
PYTEST   ?= $(PYTHON) -m pytest
SRC      := src/aperture_nexus
TESTS    := tests

.PHONY: test test-all test-integration test-fast test-cov test-module lint fmt help

# ── Test targets ────────────────────────────────────────────────────────────

test:
	$(PYTEST) $(TESTS) -m "not integration" -v

test-all:
	$(PYTEST) $(TESTS) -v

test-integration:
	$(PYTEST) $(TESTS)/integration -v

test-fast:
	$(PYTEST) $(TESTS) -m "not integration and not slow" -v

test-cov:
	$(PYTEST) $(TESTS) -m "not integration" --cov=$(SRC) --cov-report=term-missing -v

# Run a single module: make test-module m=test_client
test-module:
	$(PYTEST) $(TESTS)/$(m).py -v

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m mypy $(SRC)

fmt:
	$(PYTHON) -m autopep8 --in-place --recursive $(SRC) $(TESTS)

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make test              unit tests (no live DB)"
	@echo "  make test-all          all tests (integration skips if no DB)"
	@echo "  make test-integration  integration tests only"
	@echo "  make test-fast         unit tests, skip slow ones"
	@echo "  make test-cov          unit tests with coverage"
	@echo "  make test-module m=X   run tests/X.py"
	@echo "  make lint              mypy type check"
	@echo "  make fmt               autopep8 format"
	@echo ""
