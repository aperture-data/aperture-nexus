"""
Unit tests for aperture_nexus._client.

All tests use mock objects — no live ApertureDB instance required.
Integration tests that exercise a real connection live in
tests/integration/test_client_integration.py.
"""

import pytest
from unittest.mock import MagicMock, patch

from aperturedb.Connector import UnauthorizedException, UnauthenticatedException

from aperture_nexus._client import (
    get_connector,
    validate_connection,
    connection_description,
    _parse_status_response,
)
from aperture_nexus.exceptions import NexusConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connector(host="localhost", port=55555, use_ssl=True):
    """Return a mock Connector whose config object mirrors ApertureDB's Connector."""
    c = MagicMock()
    # Connector stores connection params on connector.config (the source of truth).
    c.config.host = host
    c.config.port = port
    c.config.use_ssl = use_ssl
    return c


def _status_ok_response():
    """GetStatus response indicating a healthy ApertureDB instance."""
    return [{"GetStatus": {"status": 0, "info": "Server is running"}}]


def _status_error_response(status=-1, info="Server error"):
    return [{"GetStatus": {"status": status, "info": info}}]


# ---------------------------------------------------------------------------
# get_connector
# ---------------------------------------------------------------------------

class TestGetConnector:
    def test_passthrough_when_db_provided(self):
        """When a Connector is passed, it is returned unchanged (no create_connector call)."""
        existing = _make_connector()
        result = get_connector(db_client=existing)
        assert result is existing

    def test_passthrough_does_not_call_create_connector(self):
        """Providing db_client= must short-circuit create_connector entirely."""
        existing = _make_connector()
        with patch("aperture_nexus._client.create_connector") as mock_create:
            result = get_connector(db_client=existing)
        mock_create.assert_not_called()
        assert result is existing

    def test_creates_connector_when_db_is_none(self):
        """When db_client=None, create_connector() is called and its result returned."""
        new_connector = _make_connector()
        with patch("aperture_nexus._client.create_connector", return_value=new_connector) as mock_create:
            result = get_connector()
        mock_create.assert_called_once_with()
        assert result is new_connector

    def test_raises_nexus_connection_error_on_assert_error(self):
        """AssertionError from create_connector (no config found) → NexusConnectionError."""
        with patch(
            "aperture_nexus._client.create_connector",
            side_effect=AssertionError("No configuration found."),
        ):
            with pytest.raises(NexusConnectionError) as exc_info:
                get_connector()

        msg = str(exc_info.value)
        assert "APERTUREDB_KEY" in msg
        assert "adb-nexus validate" in msg

    def test_nexus_connection_error_chains_original_assert(self):
        """NexusConnectionError must chain the original AssertionError."""
        original = AssertionError("No configuration found.")
        with patch("aperture_nexus._client.create_connector", side_effect=original):
            with pytest.raises(NexusConnectionError) as exc_info:
                get_connector()

        assert exc_info.value.__cause__ is original

    def test_raises_nexus_connection_error_on_unexpected_exception(self):
        """Any other exception from create_connector → NexusConnectionError."""
        with patch(
            "aperture_nexus._client.create_connector",
            side_effect=RuntimeError("something unexpected"),
        ):
            with pytest.raises(NexusConnectionError) as exc_info:
                get_connector()

        msg = str(exc_info.value)
        assert "Unexpected error" in msg
        assert "adb-nexus validate" in msg

    def test_unexpected_exception_is_chained(self):
        original = RuntimeError("boom")
        with patch("aperture_nexus._client.create_connector", side_effect=original):
            with pytest.raises(NexusConnectionError) as exc_info:
                get_connector()

        assert exc_info.value.__cause__ is original

    def test_db_none_is_default(self):
        """get_connector() with no args behaves the same as get_connector(db_client=None)."""
        connector = _make_connector()
        with patch("aperture_nexus._client.create_connector", return_value=connector):
            result = get_connector()
        assert result is connector


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------

class TestValidateConnection:
    def test_succeeds_on_status_zero(self):
        """validate_connection does not raise when GetStatus returns status=0."""
        connector = _make_connector()
        connector.query.return_value = (_status_ok_response(), [])
        # Should not raise
        validate_connection(connector)
        connector.query.assert_called_once_with([{"GetStatus": {}}])

    def test_raises_on_connection_error(self):
        """Python ConnectionError from connector.query → NexusConnectionError."""
        connector = _make_connector()
        connector.query.side_effect = ConnectionError("connection refused")
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        msg = str(exc_info.value)
        assert "unreachable" in msg.lower()
        assert "adb-nexus validate" in msg

    def test_raises_on_os_error(self):
        """OSError (e.g. network unreachable) → NexusConnectionError."""
        connector = _make_connector()
        connector.query.side_effect = OSError("network unreachable")
        with pytest.raises(NexusConnectionError):
            validate_connection(connector)

    def test_raises_on_unauthorized_exception(self):
        """UnauthorizedException → NexusConnectionError mentioning credentials."""
        connector = _make_connector()
        connector.query.side_effect = UnauthorizedException("bad token")
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        msg = str(exc_info.value)
        assert "unauthorized" in msg.lower() or "credentials" in msg.lower()

    def test_raises_on_unauthenticated_exception(self):
        """UnauthenticatedException → NexusConnectionError mentioning credentials."""
        connector = _make_connector()
        connector.query.side_effect = UnauthenticatedException("not authenticated")
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        msg = str(exc_info.value)
        assert "authenticate" in msg.lower() or "credentials" in msg.lower()

    def test_raises_on_unexpected_query_exception(self):
        """Unexpected exception from query → NexusConnectionError."""
        connector = _make_connector()
        connector.query.side_effect = RuntimeError("unexpected")
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        assert "Unexpected error" in str(exc_info.value)

    def test_raises_on_non_zero_status(self):
        """A GetStatus response with status != 0 → NexusConnectionError."""
        connector = _make_connector()
        connector.query.return_value = (_status_error_response(status=-1, info="DB error"), [])
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        msg = str(exc_info.value)
        assert "non-zero status" in msg or "status=-1" in msg

    def test_error_message_includes_host_port_and_scheme(self):
        """Error messages include the target host:port (scheme) for easier debugging."""
        connector = _make_connector(host="mydb.internal", port=9999, use_ssl=False)
        connector.query.side_effect = ConnectionError("refused")
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        assert "mydb.internal:9999 (tcp)" in str(exc_info.value)

    def test_exceptions_are_chained(self):
        """Original exceptions must be chained on NexusConnectionError."""
        original = ConnectionError("refused")
        connector = _make_connector()
        connector.query.side_effect = original
        with pytest.raises(NexusConnectionError) as exc_info:
            validate_connection(connector)

        assert exc_info.value.__cause__ is original

    def test_empty_response_raises(self):
        """An empty response list (no GetStatus body) → NexusConnectionError."""
        connector = _make_connector()
        connector.query.return_value = ([], [])
        with pytest.raises(NexusConnectionError):
            validate_connection(connector)

    def test_malformed_response_raises(self):
        """A response with no GetStatus key → NexusConnectionError."""
        connector = _make_connector()
        connector.query.return_value = ([{"SomethingElse": {}}], [])
        with pytest.raises(NexusConnectionError):
            validate_connection(connector)


# ---------------------------------------------------------------------------
# connection_description
# ---------------------------------------------------------------------------

class TestConnectionDescription:
    def test_includes_scheme_ssl(self):
        connector = _make_connector(host="db.example.com", port=12345, use_ssl=True)
        assert connection_description(connector) == "db.example.com:12345 (ssl)"

    def test_includes_scheme_tcp(self):
        connector = _make_connector(host="db.example.com", port=12345, use_ssl=False)
        assert connection_description(connector) == "db.example.com:12345 (tcp)"

    def test_returns_localhost_default(self):
        connector = _make_connector(host="localhost", port=55555, use_ssl=True)
        assert connection_description(connector) == "localhost:55555 (ssl)"

    def test_falls_back_on_missing_config(self):
        """If the connector has no config attribute, return a safe fallback string."""
        connector = MagicMock(spec=[])  # no attributes at all
        result = connection_description(connector)
        assert result == "<ApertureDB>"

    def test_does_not_expose_credentials(self):
        """Description must not contain passwords, tokens, or usernames."""
        connector = _make_connector(host="myhost", port=55555)
        connector.config.password = "supersecret"
        connector.config.token = "mytoken"
        connector.config.username = "admin"
        desc = connection_description(connector)
        assert "supersecret" not in desc
        assert "mytoken" not in desc
        assert "admin" not in desc


# ---------------------------------------------------------------------------
# _parse_status_response (internal helper)
# ---------------------------------------------------------------------------

class TestParseStatusResponse:
    def test_extracts_get_status_body(self):
        response = [{"GetStatus": {"status": 0, "info": "ok"}}]
        result = _parse_status_response(response)
        assert result == {"status": 0, "info": "ok"}

    def test_returns_empty_dict_for_empty_list(self):
        assert _parse_status_response([]) == {}

    def test_returns_empty_dict_for_none(self):
        assert _parse_status_response(None) == {}

    def test_returns_empty_dict_when_no_get_status_key(self):
        response = [{"AddEntity": {"status": 0}}]
        result = _parse_status_response(response)
        assert result == {}

    def test_returns_empty_dict_for_non_dict_element(self):
        result = _parse_status_response(["not a dict"])
        assert result == {}

    def test_uses_first_element_only(self):
        """Only the first response element is inspected."""
        response = [
            {"GetStatus": {"status": 0}},
            {"GetStatus": {"status": -1}},  # second element ignored
        ]
        result = _parse_status_response(response)
        assert result == {"status": 0}
