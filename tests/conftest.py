"""
Shared test fixtures for aperture-nexus unit tests.

Unit tests must not require a live ApertureDB instance. Use the
mock_connector fixture to inject a fake DB connection.

Integration tests that require a live DB go in tests/integration/
and must be decorated with @pytest.mark.integration.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_connector():
    """
    A mock ApertureDB Connector for unit testing.

    Simulates successful query responses by default. Override
    return_value on mock_connector.query to test error paths.

    Example:
        def test_commit_stores_context(mock_connector):
            mock_connector.query.return_value = (
                [{"AddEntity": {"status": 0}}], []
            )
            memory = Memory(db=mock_connector)
            ...
    """
    connector = MagicMock()
    connector.query.return_value = ([{"status": 0}], [])
    connector.last_query_ok.return_value = True
    return connector


@pytest.fixture
def mock_principal():
    """A minimal authenticated Principal for testing."""
    principal = MagicMock()
    principal.user_id = "test-user"
    principal.user_name = "Test User"
    return principal
