"""
Unit tests for aperture_nexus.auth.

Covers Principal construction, immutability, identity semantics, and
validate_credentials() input validation. No live ApertureDB required.
"""

import pytest

from aperture_nexus.auth import Principal, validate_credentials
from aperture_nexus.exceptions import NexusValidationError


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

class TestPrincipal:
    def test_creates_with_user_id_only(self):
        p = Principal(user_id="alice")
        assert p.user_id == "alice"
        assert p.user_name is None

    def test_creates_with_user_name(self):
        p = Principal(user_id="alice", user_name="Alice Chen")
        assert p.user_id == "alice"
        assert p.user_name == "Alice Chen"

    def test_is_immutable(self):
        """Principal is a frozen dataclass — attributes cannot be changed."""
        p = Principal(user_id="alice")
        with pytest.raises(Exception):  # FrozenInstanceError
            p.user_id = "bob"  # type: ignore[misc]

    def test_is_hashable(self):
        """Principals can be used as dict keys and stored in sets."""
        p1 = Principal(user_id="alice")
        p2 = Principal(user_id="bob")
        d = {p1: "first", p2: "second"}
        assert d[p1] == "first"
        s = {p1, p2}
        assert len(s) == 2

    def test_equality_by_value(self):
        """Two Principals with the same fields are equal."""
        p1 = Principal(user_id="alice", user_name="Alice")
        p2 = Principal(user_id="alice", user_name="Alice")
        assert p1 == p2

    def test_inequality_different_user_id(self):
        assert Principal(user_id="alice") != Principal(user_id="bob")

    def test_inequality_different_user_name(self):
        p1 = Principal(user_id="alice", user_name="Alice")
        p2 = Principal(user_id="alice", user_name="Alice C.")
        assert p1 != p2

    def test_repr_without_user_name(self):
        p = Principal(user_id="alice")
        assert repr(p) == "Principal(user_id='alice')"

    def test_repr_with_user_name(self):
        p = Principal(user_id="alice", user_name="Alice Chen")
        assert repr(p) == "Principal(user_id='alice', user_name='Alice Chen')"

    def test_repr_does_not_contain_api_key(self):
        """Repr must never expose credentials — Principal doesn't store them,
        but this documents the contract explicitly."""
        p = Principal(user_id="alice")
        assert "key" not in repr(p).lower()
        assert "secret" not in repr(p).lower()
        assert "password" not in repr(p).lower()

    def test_str_returns_user_name_when_available(self):
        p = Principal(user_id="alice", user_name="Alice Chen")
        assert str(p) == "Alice Chen"

    def test_str_falls_back_to_user_id(self):
        p = Principal(user_id="alice")
        assert str(p) == "alice"

    def test_service_account_without_display_name(self):
        """Service accounts and AI agents typically have no user_name."""
        agent = Principal(user_id="support-agent-v2")
        assert agent.user_name is None
        assert str(agent) == "support-agent-v2"


# ---------------------------------------------------------------------------
# validate_credentials
# ---------------------------------------------------------------------------

class TestValidateCredentials:
    def test_returns_normalized_user_id(self):
        result = validate_credentials("alice", "secret-key")
        assert result == "alice"

    def test_strips_whitespace_from_user_id(self):
        result = validate_credentials("  alice  ", "secret-key")
        assert result == "alice"

    def test_strips_tab_and_newline_from_user_id(self):
        result = validate_credentials("\talice\n", "secret-key")
        assert result == "alice"

    def test_does_not_return_api_key(self):
        """validate_credentials returns only the user_id — never the key."""
        result = validate_credentials("alice", "super-secret")
        assert result == "alice"
        assert "super-secret" not in result

    def test_raises_on_empty_user_id(self):
        with pytest.raises(NexusValidationError) as exc_info:
            validate_credentials("", "secret-key")
        assert "user_id" in str(exc_info.value)

    def test_raises_on_whitespace_only_user_id(self):
        with pytest.raises(NexusValidationError) as exc_info:
            validate_credentials("   ", "secret-key")
        assert "user_id" in str(exc_info.value)

    def test_raises_on_non_string_user_id(self):
        with pytest.raises(NexusValidationError):
            validate_credentials(None, "secret-key")  # type: ignore[arg-type]

    def test_raises_on_int_user_id(self):
        with pytest.raises(NexusValidationError):
            validate_credentials(42, "secret-key")  # type: ignore[arg-type]

    def test_raises_on_empty_api_key(self):
        with pytest.raises(NexusValidationError) as exc_info:
            validate_credentials("alice", "")
        assert "api_key" in str(exc_info.value)

    def test_raises_on_whitespace_only_api_key(self):
        with pytest.raises(NexusValidationError) as exc_info:
            validate_credentials("alice", "   ")
        assert "api_key" in str(exc_info.value)

    def test_raises_on_non_string_api_key(self):
        with pytest.raises(NexusValidationError):
            validate_credentials("alice", None)  # type: ignore[arg-type]

    def test_error_message_suggests_env_vars(self):
        """Error for empty api_key should guide users away from hardcoding."""
        with pytest.raises(NexusValidationError) as exc_info:
            validate_credentials("alice", "")
        assert "environment variable" in str(exc_info.value).lower()

    def test_accepts_uuid_user_id(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_credentials(uid, "key") == uid

    def test_accepts_email_user_id(self):
        uid = "alice@example.com"
        assert validate_credentials(uid, "key") == uid

    def test_accepts_service_account_user_id(self):
        uid = "support-agent-v2"
        assert validate_credentials(uid, "key") == uid
