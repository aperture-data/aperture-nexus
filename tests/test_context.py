"""
Unit tests for aperture_nexus.context.

Tests cover:
- Happy-path construction (session_id, session_name, all fields)
- Auto-generated id uniqueness
- Validation errors (missing principal, missing session, bad types)
- Optional field defaults

No live ApertureDB instance required.
"""

import pytest

from aperture_nexus.auth import Principal
from aperture_nexus.context import Context
from aperture_nexus.exceptions import NexusValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def principal():
    return Principal(user_id="alice", user_name="Alice Chen")


# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------


class TestContextConstruction:
    def test_session_id_only(self, principal):
        ctx = Context(principal=principal, session_id="sess-123")
        assert ctx.session_id == "sess-123"
        assert ctx.session_name is None

    def test_session_name_only(self, principal):
        ctx = Context(principal=principal, session_name="my-session")
        assert ctx.session_name == "my-session"
        assert ctx.session_id is None

    def test_session_id_takes_precedence(self, principal):
        ctx = Context(
            principal=principal,
            session_id="sess-123",
            session_name="fallback",
        )
        assert ctx.session_id == "sess-123"
        assert ctx.session_name == "fallback"

    def test_all_optional_fields(self, principal):
        ctx = Context(
            principal=principal,
            session_name="s",
            purpose="Q1 analysis",
            organization="AcmeCorp",
            priority=5,
            restrictions={"local": ["no-pii"], "global": []},
        )
        assert ctx.purpose == "Q1 analysis"
        assert ctx.organization == "AcmeCorp"
        assert ctx.priority == 5
        assert ctx.restrictions == {"local": ["no-pii"], "global": []}

    def test_default_priority_is_zero(self, principal):
        ctx = Context(principal=principal, session_name="s")
        assert ctx.priority == 0

    def test_default_optional_fields_are_none(self, principal):
        ctx = Context(principal=principal, session_name="s")
        assert ctx.purpose is None
        assert ctx.organization is None
        assert ctx.department is None
        assert ctx.restrictions is None

    def test_department_field(self, principal):
        ctx = Context(
            principal=principal,
            session_name="s",
            department="support",
        )
        assert ctx.department == "support"

    def test_id_is_auto_generated(self, principal):
        ctx = Context(principal=principal, session_name="s")
        assert isinstance(ctx.id, str)
        assert len(ctx.id) > 0

    def test_id_is_unique_per_instance(self, principal):
        ctx_a = Context(principal=principal, session_name="s")
        ctx_b = Context(principal=principal, session_name="s")
        assert ctx_a.id != ctx_b.id

    def test_id_not_in_constructor(self, principal):
        """id must be auto-generated, not accepted as a parameter."""
        with pytest.raises(TypeError):
            Context(
                principal=principal,
                session_name="s",
                id="custom-id",  # type: ignore[call-arg]
            )

    def test_principal_stored(self, principal):
        ctx = Context(principal=principal, session_name="s")
        assert ctx.principal is principal


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestContextValidation:
    def test_missing_principal_raises(self):
        with pytest.raises(NexusValidationError, match="principal"):
            Context(principal=None, session_name="s")  # type: ignore

    def test_wrong_principal_type_raises(self):
        with pytest.raises(NexusValidationError, match="principal"):
            Context(principal="alice", session_name="s")  # type: ignore

    def test_no_session_raises(self, principal):
        with pytest.raises(NexusValidationError, match="session"):
            Context(principal=principal)

    def test_empty_session_id_raises(self, principal):
        with pytest.raises(NexusValidationError, match="session_id"):
            Context(principal=principal, session_id="   ")

    def test_empty_session_name_raises(self, principal):
        with pytest.raises(NexusValidationError, match="session_name"):
            Context(principal=principal, session_name="")

    def test_priority_float_raises(self, principal):
        with pytest.raises(NexusValidationError, match="priority"):
            Context(
                principal=principal,
                session_name="s",
                priority=1.5,  # type: ignore[arg-type]
            )

    def test_priority_string_raises(self, principal):
        with pytest.raises(NexusValidationError, match="priority"):
            Context(
                principal=principal,
                session_name="s",
                priority="high",  # type: ignore[arg-type]
            )

    def test_restrictions_list_raises(self, principal):
        with pytest.raises(NexusValidationError, match="restrictions"):
            Context(
                principal=principal,
                session_name="s",
                restrictions=["no-pii"],  # type: ignore[arg-type]
            )

    def test_restrictions_none_accepted(self, principal):
        ctx = Context(
            principal=principal, session_name="s", restrictions=None
        )
        assert ctx.restrictions is None

    def test_restrictions_empty_dict_accepted(self, principal):
        ctx = Context(
            principal=principal, session_name="s", restrictions={}
        )
        assert ctx.restrictions == {}
