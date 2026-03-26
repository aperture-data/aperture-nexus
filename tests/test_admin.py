"""
Unit tests for aperture_nexus.admin (NexusAdmin).

Tests cover:
- create_principal(): happy path, returns api_key, duplicate raises,
  empty user_id raises
- delete_principal(): happy path, non-existent raises
- authenticate(): valid credentials return Principal, wrong key raises,
  unknown user raises, DB error raises NexusConnectionError
- _ensure_defaults(): creates schema indexes on first call; idempotent

All tests use mock_connector — no live ApertureDB required.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from aperture_nexus.admin import NexusAdmin, _hash_key
from aperture_nexus.auth import Principal
from aperture_nexus.exceptions import (
    NexusConnectionError,
    NexusPermissionError,
    NexusStorageError,
    NexusValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_admin(mock_connector):
    """Return a NexusAdmin wired to the given mock connector."""
    return NexusAdmin(db_client=mock_connector)


def _find_response(
    count: int = 0, entities: list | None = None
) -> tuple:
    """Build a FindEntity mock response (connector.query() 2-tuple)."""
    body: dict = {"status": 0, "count": count}
    if entities is not None:
        body["entities"] = entities
    return ([{"FindEntity": body}], [])


def _ok_response(cmd_name: str = "AddEntity") -> tuple:
    return ([{cmd_name: {"status": 0}}], [])


def _defaults_side_effects() -> list:
    """1 CreateIndex + nothing else (no org/dept entities)."""
    return [_ok_response("CreateIndex")]


# ---------------------------------------------------------------------------
# create_principal()
# ---------------------------------------------------------------------------


class TestCreatePrincipal:
    def test_returns_api_key(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("CreateIndex"),  # _ensure_defaults: 1 CreateIndex
            _find_response(count=0),      # alice doesn't exist
            _ok_response("AddEntity"),    # add alice
        ]
        admin = _make_admin(mock_connector)
        key = admin.create_principal("alice")
        assert isinstance(key, str)
        assert len(key) > 20  # urlsafe token is at least 43 chars

    def test_stores_hash_not_plaintext(self, mock_connector):
        """The stored properties must contain api_key_hash, not api_key."""
        mock_connector.query.side_effect = [
            _ok_response("CreateIndex"),  # _ensure_defaults: 1 CreateIndex
            _find_response(count=0),
            _ok_response("AddEntity"),
        ]

        admin = _make_admin(mock_connector)
        api_key = admin.create_principal("alice", user_name="Alice")

        # Find the AddEntity call
        all_calls = mock_connector.query.call_args_list
        add_entity_call = None
        for c in all_calls:
            cmd = c[0][0]
            if any("AddEntity" in item for item in cmd):
                add_entity_call = cmd
        assert add_entity_call is not None
        props = add_entity_call[0]["AddEntity"]["properties"]
        assert "api_key_hash" in props
        assert "api_key" not in props
        assert props["api_key_hash"] == _hash_key(api_key)
        assert props["user_name"] == "Alice"

    def test_duplicate_user_id_raises(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("CreateIndex"),  # _ensure_defaults: 1 CreateIndex
            _find_response(count=1),      # alice already exists
        ]
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError, match="already exists"):
            admin.create_principal("alice")

    def test_empty_user_id_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError):
            admin.create_principal("")

    def test_default_dept_and_org_used(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("CreateIndex"),  # _ensure_defaults: 1 CreateIndex
            _find_response(count=0),      # bob doesn't exist
            _ok_response("AddEntity"),    # add bob
        ]
        admin = _make_admin(mock_connector)
        admin.create_principal("bob")
        add_call = mock_connector.query.call_args_list[-1][0][0]
        props = add_call[0]["AddEntity"]["properties"]
        assert props["department"] == "nexus_default_dept"
        assert props["organization"] == "nexus_default_org"


# ---------------------------------------------------------------------------
# delete_principal()
# ---------------------------------------------------------------------------


class TestDeletePrincipal:
    def test_deletes_existing_principal(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),   # user exists
            _ok_response("DeleteEntity"),
        ]
        admin = _make_admin(mock_connector)
        admin._defaults_ensured = True  # skip defaults for this test
        admin.delete_principal("alice")
        assert mock_connector.query.call_count == 2

    def test_non_existent_raises(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=0),
        ]
        admin = _make_admin(mock_connector)
        admin._defaults_ensured = True
        with pytest.raises(NexusValidationError, match="does not exist"):
            admin.delete_principal("ghost")

    def test_empty_user_id_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        admin._defaults_ensured = True
        with pytest.raises(NexusValidationError, match="non-empty"):
            admin.delete_principal("")


# ---------------------------------------------------------------------------
# authenticate()
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_valid_credentials_return_principal(self, mock_connector):
        api_key = "test-key-12345"
        record = {
            "user_id": "alice",
            "user_name": "Alice Chen",
            "department": "support",
            "organization": "AcmeCorp",
            "api_key_hash": _hash_key(api_key),
        }
        mock_connector.query.return_value = _find_response(
            count=1, entities=[record]
        )
        admin = _make_admin(mock_connector)
        principal = admin.authenticate("alice", api_key)
        assert isinstance(principal, Principal)
        assert principal.user_id == "alice"
        assert principal.user_name == "Alice Chen"
        assert principal.department == "support"
        assert principal.organization == "AcmeCorp"

    def test_wrong_api_key_raises(self, mock_connector):
        record = {
            "user_id": "alice",
            "api_key_hash": _hash_key("correct-key"),
        }
        mock_connector.query.return_value = _find_response(
            count=1, entities=[record]
        )
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusPermissionError, match="Invalid credentials"):
            admin.authenticate("alice", "wrong-key")

    def test_unknown_user_raises(self, mock_connector):
        mock_connector.query.return_value = _find_response(
            count=0, entities=[]
        )
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusPermissionError, match="does not exist"):
            admin.authenticate("ghost", "any-key")

    def test_db_error_raises_connection_error(self, mock_connector):
        mock_connector.query.side_effect = ConnectionError("timeout")
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusConnectionError, match="query failed"):
            admin.authenticate("alice", "key")

    def test_empty_user_id_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError):
            admin.authenticate("", "key")

    def test_empty_api_key_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError):
            admin.authenticate("alice", "")


# ---------------------------------------------------------------------------
# _ensure_defaults()
# ---------------------------------------------------------------------------


class TestEnsureDefaults:
    def test_creates_schema_indexes(self, mock_connector):
        # 1 CreateIndex for NexusUser.user_id
        mock_connector.query.side_effect = _defaults_side_effects()
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        assert mock_connector.query.call_count == 1

    def test_idempotent_on_first_call(self, mock_connector):
        # Same 1 query regardless of whether index exists
        mock_connector.query.side_effect = _defaults_side_effects()
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        assert mock_connector.query.call_count == 1

    def test_idempotent_second_call(self, mock_connector):
        mock_connector.query.side_effect = _defaults_side_effects()
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        admin._ensure_defaults()   # second call — no more DB queries
        assert mock_connector.query.call_count == 1
