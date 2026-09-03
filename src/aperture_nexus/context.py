"""
Context — captures who is doing what, in which session, and why.

A Context is a pure data object. It does not write to ApertureDB.
Writing happens only when Memory.commit() is called.

Every commit is tagged with the Context that produced it, linking
the stored memory to a session, a principal, and an optional purpose
and organization. This enables precise filtering at search time.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from aperture_nexus.auth import Principal
from aperture_nexus.exceptions import NexusValidationError

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Who is doing what, in which session, and why.

    A Context is a pure data object — it never writes to ApertureDB.
    Pass it to Memory.commit() to store information under this context.

    Either ``session_id`` or ``session_name`` is required. If both are
    provided, ``session_id`` takes precedence. If only ``session_name``
    is given and no session with that name exists, a new session is
    created on first ``Memory.commit()``.

    Attributes:
        principal: The authenticated identity. Required.
        session_id: Unique session identifier. Use
            ``generate_session_id()`` to generate one.
        session_name: Human-readable session name. Must be unique
            within the principal's scope.
        purpose: Why this interaction is happening. Stored as
            metadata; searchable.
        organization: Group scope for permission and search filtering.
        department: Department this context belongs to. Inherited
            from ``principal.department`` if not set explicitly.
        priority: Relative processing priority. Higher values are
            processed first in batch operations. Default: 0.
        restrictions: Access constraints applied during search.
            Format: ``{"local": [...], "global": [...]}``.
        id: Auto-generated context ID. Read-only; not set by caller.

    Example:
        from aperture_nexus import Memory, Context, generate_session_id

        memory = Memory()
        principal = memory.authenticate(user_id="alice", api_key="...")

        # Single participant
        ctx = Context(
            principal=principal,
            session_name="support-2024-001",
            purpose="Customer reporting missing order",
            organization="AcmeCorp",
        )

        # One of many participants in a shared session
        sid = generate_session_id()
        ctx_a = Context(principal=principal_a, session_id=sid)
        ctx_b = Context(principal=principal_b, session_id=sid)
    """

    principal: Principal
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    purpose: Optional[str] = None
    organization: Optional[str] = None
    department: Optional[str] = None
    priority: int = 0
    restrictions: Optional[dict] = None
    id: str = field(
        default_factory=lambda: str(uuid.uuid4()), init=False
    )

    def __post_init__(self) -> None:
        self._validate()
        logger.debug(
            "Context created: id=%r session_id=%r session_name=%r",
            self.id,
            self.session_id,
            self.session_name,
        )

    def _validate(self) -> None:
        if not isinstance(self.principal, Principal):
            raise NexusValidationError(
                "principal must be a Principal instance. "
                "Call memory.authenticate() to get one."
            )

        if not self.session_id and not self.session_name:
            raise NexusValidationError(
                "Either session_id or session_name is required. "
                "Provide session_id for an existing session, or "
                "session_name to create or join a named session."
            )

        if (
            self.session_id is not None
            and not str(self.session_id).strip()
        ):
            raise NexusValidationError(
                "session_id must be a non-empty string. "
                "Use generate_session_id() to create one."
            )

        if (
            self.session_name is not None
            and not str(self.session_name).strip()
        ):
            raise NexusValidationError(
                "session_name must be a non-empty string. "
                "Provide a human-readable name, e.g. 'support-2024-001'."
            )

        if not isinstance(self.priority, int):
            raise NexusValidationError(
                "priority must be an integer. "
                f"Got {type(self.priority).__name__!r}."
            )

        if (
            self.restrictions is not None
            and not isinstance(self.restrictions, dict)
        ):
            raise NexusValidationError(
                "restrictions must be a dict or None. "
                'Format: {"local": [...], "global": [...]}.'
            )
