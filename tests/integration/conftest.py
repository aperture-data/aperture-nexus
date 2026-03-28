"""
Integration test fixtures for aperture-nexus.

Spins up ApertureDB community edition in Docker for the test session,
registers it as the active ``adb`` configuration, then tears both
down cleanly — even on test failure.

Lifecycle (session-scoped):
    1. Start ApertureDB community container (random host port via testcontainers)
    2. Run  ``adb config create nexus-integ-<hex>`` — all defaults except
       port (dynamic) and ssl=False — and activate it immediately
    3. Memory() / NexusAdmin() resolve the connector through the normal
       ``create_connector()`` path, exactly as a real user would
    4. On teardown:
       a. ``adb config remove nexus-integ-<hex>`` restores the previous
          active config (or leaves the file clean if none existed)
       b. testcontainers stops and removes the Docker container

No containers or config entries are left behind.

Requirements:
    pip install testcontainers docker
    docker daemon running

Skip integration tests without Docker:
    pytest -m "not integration"
    make test
"""

import json
import time
import socket
import uuid

import pytest


# ---------------------------------------------------------------------------
# Auto-mark every test in this directory as @pytest.mark.integration
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_port(host: str, port: int, timeout: float = 90.0) -> None:
    """Poll TCP until host:port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(
        f"ApertureDB did not become available at {host}:{port} "
        f"within {timeout:.0f}s. Is Docker running?"
    )


def _wait_for_aperturedb(host: str, port: int, timeout: float = 90.0) -> None:
    """Poll until ApertureDB accepts authenticated queries or timeout expires.

    The TCP port opens before ApertureDB finishes internal initialisation.
    This probe retries the actual auth handshake so callers do not need
    an arbitrary sleep after _wait_for_port.
    """
    from aperturedb.Connector import Connector

    deadline = time.monotonic() + timeout
    last_exc = None
    while time.monotonic() < deadline:
        try:
            c = Connector(
                host=host, port=port,
                user="admin", password="admin",
                use_ssl=False,
            )
            c.query([{"GetStatus": {}}])
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"ApertureDB at {host}:{port} did not become ready within "
        f"{timeout:.0f}s. Last error: {last_exc}"
    )


def _adb_config_path(as_global: bool = True):
    """Return the Path to the adb config JSON file."""
    from aperturedb.cli.configure import _config_file_path
    return _config_file_path(as_global=as_global)


def _load_configs(config_path):
    """Load (configs_dict, active_name) from the adb config file.

    Returns ({}, None) if the file does not exist or is malformed.
    """
    try:
        from aperturedb.cli.configure import get_configurations
        return get_configurations(str(config_path))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}, None


def _save_configs(config_path, configs: dict, active: str | None) -> None:
    """Write configs dict back to the adb config file."""
    from aperturedb.cli.configure import _write_config
    if active is not None:
        configs["active"] = active
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_config(config_path, configs)


# ---------------------------------------------------------------------------
# Container fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def aperturedb_container():
    """Start ApertureDB community edition; yield {host, port, user, password}.

    testcontainers manages the full Docker lifecycle — the container is
    stopped and removed when the ``with`` block exits, regardless of whether
    tests passed or failed.
    """
    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip(
            "testcontainers is not installed. "
            "Run: pip install aperture-nexus[integration]"
        )

    with (
        DockerContainer("aperturedata/aperturedb-community:latest")
        .with_env("ADB_MASTER_KEY", "admin")
        .with_env("ADB_FORCE_SSL", "false")
        .with_exposed_ports(55555)
    ) as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(55555))

        try:
            _wait_for_port(host, port, timeout=90.0)
            _wait_for_aperturedb(host, port, timeout=90.0)
        except TimeoutError as exc:
            pytest.fail(str(exc))

        yield {"host": host, "port": port, "user": "admin", "password": "admin"}

        # Container stop + rm happens here via context manager __exit__


# ---------------------------------------------------------------------------
# adb config create / remove fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def adb_config_name(aperturedb_container):
    """Create a named adb config for the test container; activate it.

    Uses all ``adb config create`` defaults except:
    - port  → dynamic port assigned by the container
    - ssl   → False  (community edition without TLS)

    The previous active config (if any) is restored on teardown.
    The test config entry is always removed on teardown.

    Yields:
        str — the config name (e.g. ``"nexus-integ-a3f7c2d1"``)
    """
    from aperturedb.Configuration import Configuration

    config_name = f"nexus-integ-{uuid.uuid4().hex[:8]}"
    config_path = _adb_config_path(as_global=True)

    # Snapshot the current state so we can restore it later
    configs, prev_active = _load_configs(config_path)

    # Register the test config and make it active
    configs[config_name] = Configuration(
        name=config_name,
        host=aperturedb_container["host"],
        port=aperturedb_container["port"],
        username="admin",
        password="admin",
        use_ssl=False,
        use_rest=False,
        verify_hostname=False,
    )
    _save_configs(config_path, configs, active=config_name)

    yield config_name

    # ---- Teardown: remove test config, restore previous active ----
    try:
        configs_now, _ = _load_configs(config_path)
        configs_now.pop(config_name, None)

        # Restore the config that was active before we started, if it
        # still exists.  If nothing is left, remove the active key.
        if prev_active and prev_active in configs_now:
            _save_configs(config_path, configs_now, active=prev_active)
        elif configs_now:
            _save_configs(
                config_path, configs_now,
                active=next(iter(configs_now))
            )
        else:
            # Nothing left — write an empty-ish file without "active"
            config_path.write_text("{}")
    except Exception:
        pass  # best-effort; container is being torn down anyway


# ---------------------------------------------------------------------------
# High-level fixtures that depend on adb_config_name
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def nexus_admin(adb_config_name):
    """Return a NexusAdmin that resolves its connector from the active adb config.

    No db_client is injected — this exercises the full create_connector()
    resolution path, exactly as a real user would call NexusAdmin().
    """
    from aperture_nexus.admin import NexusAdmin
    return NexusAdmin()


@pytest.fixture(scope="session")
def test_principal(nexus_admin, adb_config_name):
    """Create a test principal once for the session; delete it on teardown."""
    from aperture_nexus.memory import Memory
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    api_key = nexus_admin.create_principal(
        user_id=user_id,
        user_name="Integration Test User",
    )
    # authenticate via Memory — no admin credentials needed at session time
    principal = Memory().authenticate(user_id=user_id, api_key=api_key)
    yield principal
    try:
        nexus_admin.delete_principal(user_id=user_id)
    except Exception:
        pass


@pytest.fixture()
def memory_engine(adb_config_name):
    """Return a Memory instance that resolves its connector from the active adb config."""
    from aperture_nexus.memory import Memory
    return Memory()
