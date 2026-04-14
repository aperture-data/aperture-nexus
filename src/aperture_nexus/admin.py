"""
NexusAdmin — management interface for aperture-nexus.

NexusAdmin handles system-level operations: principal provisioning,
cleanup, and maintenance. It requires admin-level ApertureDB credentials
and is intended to be used by operators, not by application code.

In normal use, NexusAdmin is driven entirely through the CLI:
    adb-nexus init   # creates principal, writes NEXUS_API_KEY to .env

Application code authenticates via Memory.authenticate(), which needs
only regular (non-admin) ApertureDB credentials and the NEXUS_API_KEY
from .env — no NexusAdmin involvement at session time.

NexusAdmin is also importable for programmatic administration:
    admin = NexusAdmin()
    api_key = admin.create_principal(user_id="alice", ...)
    admin.rotate_key(user_id="alice")
    admin.delete_principal(user_id="alice")

ApertureDB entity classes used here:
    NexusUser — app-level Principal records (hashed api_key)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Optional

from aperture_nexus._client import get_connector
from aperture_nexus.auth import validate_credentials
from aperture_nexus.config import load_config
from aperture_nexus.exceptions import (
    NexusStorageError,
    NexusValidationError,
)

logger = logging.getLogger(__name__)

# ApertureDB entity class names
_CLASS_USER = "NexusUser"

# ApertureDB permission model note:
# ApertureDB does not have per-object-type permission grants.
# Granting a DB user access gives read/write/update/delete on ALL object types.
# The only permission distinctions are:
#   - Admin users: can create/remove DB users and roles.
#   - Regular users: can read/write/update/delete all objects (including indexes).
# There is no way to grant "entity read" without also granting "blob read", etc.


def _hash_key(api_key: str) -> str:
    """Return a SHA-256 hex digest of the given api_key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _check_response(response, operation: str) -> None:
    """Raise NexusStorageError if any command in the response failed.

    Handles both the normal list-of-dicts format and the bare-dict format
    that ApertureDB returns for schema/parameter errors.
    """
    # ApertureDB returns a bare dict (not wrapped in a list) for invalid queries
    if isinstance(response, dict):
        status = response.get("status", -1)
        if status != 0:
            raise NexusStorageError(
                f"{operation} failed (status={status}): "
                f"{response.get('info', 'no details')}. "
                f"Check your ApertureDB connection and schema."
            )
        return
    for item in response:
        for cmd_name, body in item.items():
            status = body.get("status", -1) if isinstance(body, dict) else -1
            if status != 0:
                info = (
                    body.get("info", "no details")
                    if isinstance(body, dict)
                    else str(body)
                )
                raise NexusStorageError(
                    f"{operation} failed (status={status}): {info}. "
                    f"Check your ApertureDB connection and schema."
                )


def _entity_exists(db, entity_class: str, constraints: dict) -> bool:
    """Return True if at least one entity matching constraints exists."""
    cmd = [{
        "FindEntity": {
            "with_class": entity_class,
            "constraints": constraints,
            "results": {"count": True},
        }
    }]
    response, _ = db.query(cmd)
    if isinstance(response, dict):
        return False  # error response — entity does not exist
    body = response[0].get("FindEntity", {})
    return body.get("count", 0) > 0


class NexusAdmin:
    """Management interface for aperture-nexus system operations.

    Handles principal provisioning, key rotation, cleanup, and other
    administrative tasks. Requires admin-level ApertureDB credentials.
    Not used in application hot paths — see ``Memory.authenticate()``
    for session-time credential verification.

    In the common single-user case, ``adb-nexus init`` drives this
    automatically. Use ``NexusAdmin`` directly for programmatic
    multi-user administration (enterprise deployments, scripted
    provisioning, automated cleanup).

    Args:
        config: Path to ``aperture_nexus.json``. Discovered automatically
            if ``None``.
        db_client: Inject an existing admin ApertureDB ``Connector``. If
            ``None``, a connector is created from environment variables or
            the active ``adb`` configuration.

    Raises:
        NexusConnectionError: If admin credentials cannot be resolved.
        NexusConfigError: If the config file is invalid.

    Example:
        admin = NexusAdmin()
        api_key = admin.create_principal(
            user_id="alice",
            user_name="Alice Chen",
            department="engineering",
            organization="AcmeCorp",
        )
        # Deliver api_key to alice securely (e.g. write to her .env).
        # To issue a replacement key: admin.rotate_key(user_id="alice")
    """

    def __init__(
        self,
        config: Optional[str] = None,
        db_client=None,
    ) -> None:
        self._cfg = load_config(path=config, validate_deps=False)
        self._db = get_connector(db_client)
        self._defaults_ensured = False
        logger.debug("NexusAdmin initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_principal(
        self,
        user_id: str,
        user_name: Optional[str] = None,
        department: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """Register a new app-level Principal.

        Generates a random API key, stores its SHA-256 hash in ApertureDB
        as a ``NexusUser`` entity, and returns the plaintext key. The
        plaintext key is never stored — deliver it to the user out-of-band
        and discard it after this call.

        Args:
            user_id: Unique, stable identifier for this principal. Must
                not already exist.
            user_name: Optional human-readable display name.
            department: Department this principal belongs to. Defaults
                to the configured default department.
            organization: Organization this principal belongs to.
                Defaults to the configured default organization.

        Returns:
            The generated plaintext API key. Show once; store securely.

        Raises:
            NexusValidationError: If user_id is empty or already exists.
            NexusStorageError: If ApertureDB rejects the write.

        Example:
            api_key = admin.create_principal(
                user_id="alice",
                user_name="Alice Chen",
                department="engineering",
                organization="AcmeCorp",
            )
            # Deliver api_key to alice securely (e.g. write to .env).
            # To issue a replacement: admin.rotate_key(user_id="alice")
        """
        user_id = validate_credentials(user_id, "placeholder")
        self._ensure_defaults()

        if _entity_exists(
            self._db, _CLASS_USER, {"user_id": ["==", user_id]}
        ):
            raise NexusValidationError(
                f"Principal {user_id!r} already exists. "
                f"Call delete_principal() first to replace it."
            )

        dept = department or self._cfg.admin.default_department
        org = organization or self._cfg.admin.default_organization
        api_key = secrets.token_urlsafe(32)
        key_hash = _hash_key(api_key)

        props: dict = {
            "user_id": user_id,
            "api_key_hash": key_hash,
            "department": dept,
            "organization": org,
        }
        if user_name:
            props["user_name"] = user_name

        cmd = [{"AddEntity": {"class": _CLASS_USER, "properties": props}}]
        response, _ = self._db.query(cmd)
        _check_response(response, f"create_principal({user_id!r})")
        logger.debug(
            "Created principal %r in dept %r org %r", user_id, dept, org
        )
        return api_key

    def delete_principal(self, user_id: str) -> None:
        """Remove a Principal from ApertureDB.

        Existing memories written by this user are retained. The user
        will no longer be able to authenticate.

        Args:
            user_id: The user_id of the Principal to remove.

        Raises:
            NexusValidationError: If user_id is empty or does not exist.
            NexusStorageError: If ApertureDB rejects the delete.

        Example:
            admin.delete_principal(user_id="alice")
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise NexusValidationError(
                "user_id must be a non-empty string. "
                "Pass the user_id used when calling create_principal()."
            )
        if not _entity_exists(
            self._db, _CLASS_USER, {"user_id": ["==", user_id]}
        ):
            raise NexusValidationError(
                f"Principal {user_id!r} does not exist. "
                "Check the user_id or call admin.create_principal() first."
            )

        cmd = [{
            "DeleteEntity": {
                "with_class": _CLASS_USER,
                "constraints": {"user_id": ["==", user_id]},
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"delete_principal({user_id!r})")
        logger.debug("Deleted principal %r", user_id)

    def rotate_key(self, user_id: str) -> str:
        """Issue a replacement API key for an existing Principal.

        Generates a new random key, replaces the stored hash, and
        returns the new plaintext key. The previous key is immediately
        invalidated. Identity verification of the user is the caller's
        responsibility (out-of-band).

        Args:
            user_id: The user_id of the Principal whose key to rotate.

        Returns:
            The new plaintext API key. Deliver securely; store in .env.

        Raises:
            NexusValidationError: If user_id is empty or does not exist.
            NexusStorageError: If ApertureDB rejects the update.

        Example:
            new_key = admin.rotate_key(user_id="alice")
            # Deliver new_key to alice securely; update her .env.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise NexusValidationError(
                "user_id must be a non-empty string. "
                "Pass the user_id used when calling create_principal()."
            )
        if not _entity_exists(
            self._db, _CLASS_USER, {"user_id": ["==", user_id]}
        ):
            raise NexusValidationError(
                f"Principal {user_id!r} does not exist. "
                "Check the user_id or call admin.create_principal() first."
            )

        logger.warning(
            "rotate_key() has not been validated against a live "
            "ApertureDB instance. Use with caution."
        )
        new_key = secrets.token_urlsafe(32)
        cmd = [{
            "UpdateEntity": {
                "with_class": _CLASS_USER,
                "constraints": {"user_id": ["==", user_id]},
                "properties": {"api_key_hash": _hash_key(new_key)},
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"rotate_key({user_id!r})")
        logger.debug("Rotated key for principal %r", user_id)
        return new_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_defaults(self) -> None:
        """Create property indexes — idempotent."""
        if self._defaults_ensured:
            return
        self._ensure_schema()
        self._defaults_ensured = True

    def _ensure_schema(self) -> None:
        """Create property indexes for fast constraint lookups — idempotent."""
        indexes = [
            (_CLASS_USER, "user_id"),
        ]
        for cls, prop in indexes:
            cmd = [{"CreateIndex": {
                "index_type": "entity",
                "class": cls,
                "property_key": prop,
            }}]
            response, _ = self._db.query(cmd)
            for item in response:
                for _, body in item.items():
                    status = body.get("status", -1) if isinstance(body, dict) else -1
                    if status not in (0, 2):
                        info = body.get("info", "no details") if isinstance(body, dict) else str(body)
                        logger.warning(
                            "CreateIndex %r.%r returned status=%d: %s",
                            cls, prop, status, info,
                        )
            logger.debug("Ensured index on %s.%s", cls, prop)
