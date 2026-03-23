"""
Principal identity and credential validation for aperture-nexus.

A Principal represents an authenticated identity — a user, AI agent, or
service account — that can create Contexts and commit memories. Principals
are produced by NexusAdmin.authenticate() and are intentionally lightweight:
they carry only the information needed for attribution and permission checks.

API keys and raw credentials are NEVER stored on the Principal after
validation. It is safe to log or persist Principal objects.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from aperture_nexus.exceptions import NexusValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    """An authenticated identity that can create Contexts and commit memories.

    Principals are immutable and hashable — they can be used as dict keys,
    stored in sets, and safely shared across threads. They carry no credentials
    after construction; the API key used during authentication is discarded
    immediately and never stored here.

    Attributes:
        user_id: Unique, stable identifier for this principal. Used for
            attribution, permission checks, and linking sessions. Should not
            change over time (e.g. an email address, UUID, or service account
            name).
        user_name: Optional human-readable display name. Used in UI and log
            output. May be ``None`` for service accounts or AI agents where a
            display name is not meaningful.
        department: Department this principal belongs to. Determines which
            ApertureDB DB user credentials are used for writes. Set by
            ``NexusAdmin.create_principal()``.
        organization: Organization this principal belongs to. Used for
            metadata and search filtering.

    Example:
        principal = admin.authenticate(
            user_id="alice@example.com", api_key="..."
        )
        ctx = Context(principal=principal, session_name="support-001")

        # Safe to log — no credentials
        logger.info("Session started by %s", principal)
    """

    user_id: str
    user_name: Optional[str] = None
    department: Optional[str] = None
    organization: Optional[str] = None

    def __repr__(self) -> str:
        if self.user_name:
            return (
                f"Principal(user_id={self.user_id!r},"
                f" user_name={self.user_name!r})"
            )
        return f"Principal(user_id={self.user_id!r})"

    def __str__(self) -> str:
        return self.user_name or self.user_id


def validate_credentials(user_id: str, api_key: str) -> str:
    """Validate authentication inputs and return the normalized user_id.

    Called internally by ``NexusAdmin.authenticate()`` before any DB interaction.
    Validates that both fields are non-empty after stripping whitespace, and
    returns the stripped ``user_id`` so the caller uses a consistent value.

    The ``api_key`` is validated here but never returned or stored. It is
    passed separately to the DB authentication layer and discarded after use.

    Args:
        user_id: Unique identifier for the principal. Must be a non-empty
            string after stripping whitespace.
        api_key: Secret credential. Must be a non-empty string after stripping.
            Never logged or stored.

    Returns:
        The normalized (stripped) ``user_id``.

    Raises:
        NexusValidationError: If ``user_id`` or ``api_key`` is empty,
            whitespace-only, or not a string.

    Example:
        normalized_id = validate_credentials("  alice  ", "secret-key")
        # normalized_id == "alice"
        principal = Principal(user_id=normalized_id)
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise NexusValidationError(
            "user_id must be a non-empty string. "
            "Provide a stable unique identifier for this principal "
            "(e.g. an email address, UUID, or service account name)."
        )

    if not isinstance(api_key, str) or not api_key.strip():
        raise NexusValidationError(
            "api_key must be a non-empty string. "
            "Provide the API key or token for this principal. "
            "Never hardcode credentials — use environment variables "
            "or a secrets manager."
        )

    normalized = user_id.strip()
    logger.debug("Credentials validated for user_id=%r", normalized)
    return normalized
