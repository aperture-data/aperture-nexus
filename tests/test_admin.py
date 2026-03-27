"""
Unit tests for aperture_nexus.admin (NexusAdmin).

Tests cover:
- create_principal(): happy path, returns api_key, duplicate raises,
  empty user_id raises
- delete_principal(): happy path, non-existent raises
- rotate_key(): happy path, non-existent raises, empty user_id raises
- _ensure_defaults(): creates schema indexes on first call; idempotent

All tests use mock_connector — no live ApertureDB required.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from aperture_nexus.admin import NexusAdmin, _hash_key
from aperture_nexus.exceptions import (
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
# rotate_key()
# ---------------------------------------------------------------------------


class TestRotateKey:
    def test_rotate_key_returns_new_key(self, mock_connector):
        # exists check → True, then UpdateEntity → ok
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _ok_response("UpdateEntity"),
        ]
        admin = _make_admin(mock_connector)
        new_key = admin.rotate_key("alice")
        assert isinstance(new_key, str)
        assert len(new_key) > 0

    def test_rotate_key_new_key_differs_each_call(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _ok_response("UpdateEntity"),
            _find_response(count=1),
            _ok_response("UpdateEntity"),
        ]
        admin = _make_admin(mock_connector)
        key1 = admin.rotate_key("alice")
        key2 = admin.rotate_key("alice")
        assert key1 != key2

    def test_rotate_key_nonexistent_raises(self, mock_connector):
        mock_connector.query.return_value = _find_response(count=0)
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError, match="does not exist"):
            admin.rotate_key("ghost")

    def test_rotate_key_empty_user_id_raises(self, mock_connector):
        admin = _make_admin(mock_connector)
        with pytest.raises(NexusValidationError):
            admin.rotate_key("")


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
