"""
Fixtures for integration tests that require a live ApertureDB instance.

These tests are excluded from the default pytest run:
    pytest tests/                        # unit tests only
    pytest tests/integration/            # integration tests only

ApertureDB connection is read from environment variables or
aperture_nexus.json — same as production. Run ApertureDB locally with:
    docker run -p 55555:55555 aperturedata/aperturedb-community
"""

import pytest
from aperturedb.CommonLibrary import create_connector


@pytest.fixture(scope="session")
def live_connector():
    """
    A real ApertureDB Connector for integration tests.

    Reads connection from APERTUREDB_KEY or individual env vars.
    Raises a clear error if ApertureDB is unreachable.
    """
    try:
        connector = create_connector()
        # Verify connection is alive
        resp, _ = connector.query([{"GetSchema": {}}])
        return connector
    except Exception as e:
        pytest.skip(
            f"ApertureDB not reachable — skipping integration tests. "
            f"Start ApertureDB with: docker run -p 55555:55555 aperturedata/aperturedb-community. "
            f"Error: {e}"
        )
