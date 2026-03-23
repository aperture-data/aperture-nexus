"""
Integration test fixtures for aperture-nexus.

Spins up ApertureDB community edition in Docker for the test session.
The container is stopped and removed automatically when the session ends —
even on test failure. No containers are left running.

Requires:
    pip install testcontainers docker
    docker daemon running

Skip all integration tests if Docker is unavailable:
    pytest -m "not integration"
"""

import time
import socket
import pytest
from unittest.mock import MagicMock

# Mark all tests in this directory as integration
def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    """Poll until host:port accepts TCP connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(
        f"ApertureDB did not become available at {host}:{port} "
        f"within {timeout:.0f}s."
    )


@pytest.fixture(scope="session")
def aperturedb_container():
    """Start ApertureDB community edition and yield connection params.

    Uses testcontainers to manage the Docker lifecycle. The container is
    stopped and removed when the session ends — no cleanup required by
    individual tests.

    Yields:
        dict with keys: host, port, user, password
    """
    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip(
            "testcontainers not installed. "
            "Install with: pip install testcontainers docker"
        )

    with (
        DockerContainer("aperturedata/aperturedb-community:latest")
        .with_env("ADB_MASTER_KEY", "admin")
        .with_env("ADB_FORCE_SSL", "false")
        .with_env("ADB_PORT", "55553")
        .with_exposed_ports(55553)
    ) as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(55553))

        # Wait for ApertureDB to accept connections (up to 60s)
        try:
            _wait_for_port(host, port, timeout=60.0)
        except TimeoutError as e:
            pytest.fail(str(e))

        # Give the DB a moment to finish internal initialisation
        time.sleep(2.0)

        yield {"host": host, "port": port, "user": "admin", "password": "admin"}
        # Container is stopped and removed here by the context manager


@pytest.fixture(scope="session")
def db_connector(aperturedb_container):
    """Return an ApertureDB Connector connected to the test container."""
    from aperturedb.Connector import Connector
    conn_params = aperturedb_container
    connector = Connector(
        host=conn_params["host"],
        port=conn_params["port"],
        user=conn_params["user"],
        password=conn_params["password"],
        use_ssl=False,
    )
    return connector


@pytest.fixture(scope="session")
def nexus_admin(db_connector):
    """Return a NexusAdmin wired to the test container."""
    from aperture_nexus.admin import NexusAdmin
    return NexusAdmin(db_client=db_connector)


@pytest.fixture(scope="session")
def test_principal(nexus_admin):
    """Create a test principal and return it. Cleaned up at session end."""
    import uuid
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    api_key = nexus_admin.create_principal(
        user_id=user_id,
        user_name="Integration Test User",
        department="nexus_default_dept",
        organization="nexus_default_org",
    )
    principal = nexus_admin.authenticate(user_id=user_id, api_key=api_key)
    yield principal
    # Best-effort cleanup
    try:
        nexus_admin.delete_principal(user_id=user_id)
    except Exception:
        pass


@pytest.fixture()
def memory_engine(db_connector):
    """Return a Memory instance wired to the test container."""
    from aperture_nexus.memory import Memory
    return Memory(db_client=db_connector)
