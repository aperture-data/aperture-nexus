"""
Unit tests for aperture_nexus.admin (NexusAdmin).

Tests cover:
- create_department(): happy path, duplicate is no-op, empty name raises
- create_principal(): happy path, returns api_key, duplicate raises,
  empty user_id raises
- delete_principal(): happy path, non-existent raises
- authenticate(): valid credentials return Principal, wrong key raises,
  unknown user raises, DB error raises NexusConnectionError
- _ensure_defaults(): creates default org + dept on first call; idempotent

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


# ---------------------------------------------------------------------------
# create_department()
# ---------------------------------------------------------------------------


class TestCreateDepartment:
    def test_creates_department(self, mock_connector):
        # First call: ensure_defaults (org find → add, dept find → add)
        # Then: find dept → not found, add dept
        mock_connector.query.side_effect = [
            _find_response(count=0),   # find default org
            _ok_response(),            # add default org
            _find_response(count=0),   # find default dept
            _ok_response(),            # add default dept
            _find_response(count=0),   # find "support" dept
            _ok_response(),            # add "support" dept
        ]
        admin = _make_admin(mock_connector)
        admin.create_department("support", organization="AcmeCorp")
        # 6 queries issued
        assert mock_connector.query.call_count == 6

    def test_duplicate_is_no_op(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),   # find default org → exists
            _find_response(count=1),   # find default dept → exists
            _find_response(count=1),   # find "support" → exists
        ]
        admin = _make_admin(mock_connector)
        admin.create_department("support")
        # no AddEntity called for support
        assert mock_connector.query.call_count == 3

    def test_empty_name_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            admin.create_department("")

    def test_whitespace_name_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            admin.create_department("   ")

    def test_storage_error_raised(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _find_response(count=0),
            ([{"AddEntity": {"status": -1, "info": "schema error"}}], []),
        ]
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusStorageError, match="schema error"):
            admin.create_department("bad-dept")


# ---------------------------------------------------------------------------
# create_principal()
# ---------------------------------------------------------------------------


class TestCreatePrincipal:
    def test_returns_api_key(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),   # default org exists
            _find_response(count=1),   # default dept exists
            _find_response(count=0),   # user doesn't exist
            _ok_response(),            # add user
        ]
        admin = _make_admin(mock_connector)
        key = admin.create_principal("alice")
        assert isinstance(key, str)
        assert len(key) > 20  # urlsafe token is at least 43 chars

    def test_stores_hash_not_plaintext(self, mock_connector):
        """The stored properties must contain api_key_hash, not api_key."""
        captured = []

        def capture(cmd, blobs=None):
            captured.extend(cmd)
            return _ok_response(), []

        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _find_response(count=0),
        ]
        # patch the final AddEntity call
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _find_response(count=0),
            _ok_response(),
        ]
        add_calls = []
        original = mock_connector.query.side_effect

        def spy(cmd, blobs=None):
            add_calls.append(cmd)
            idx = mock_connector.query.call_count
            return original[idx - 1] if idx <= len(original) else (_ok_response(), [])

        mock_connector.query.side_effect = original

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
            _find_response(count=1),
            _find_response(count=1),
            _find_response(count=1),   # user already exists
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
            _find_response(count=1),
            _find_response(count=1),
            _find_response(count=0),
            _ok_response(),
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
    def test_creates_org_and_dept_when_absent(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=0),   # org not found
            _ok_response(),            # add org
            _find_response(count=0),   # dept not found
            _ok_response(),            # add dept
        ]
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        assert mock_connector.query.call_count == 4

    def test_skips_when_already_exist(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),   # org exists
            _find_response(count=1),   # dept exists
        ]
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        assert mock_connector.query.call_count == 2

    def test_idempotent_second_call(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
        ]
        admin = _make_admin(mock_connector)
        admin._ensure_defaults()
        admin._ensure_defaults()   # second call — no more DB queries
        assert mock_connector.query.call_count == 2
