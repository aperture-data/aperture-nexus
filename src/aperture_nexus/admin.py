"""
NexusAdmin — identity authority for aperture-nexus.

NexusAdmin manages departments and Principals. It is the only component
that creates ApertureDB DB users and app-level identity records. Memory
never authenticates — it receives an already-authenticated Principal
via Context and trusts it.

Usage pattern:
    # Setup (once per deployment)
    admin = NexusAdmin()
    admin.create_department("support", organization="AcmeCorp")
    api_key = admin.create_principal(
        user_id="alice",
        user_name="Alice Chen",
        department="support",
        organization="AcmeCorp",
    )
    # Deliver api_key to alice out-of-band (not logged, not stored)

    # Per-session (at session start)
    principal = admin.authenticate(user_id="alice", api_key="...")

ApertureDB entity classes used here:
    NexusOrganization — organisation records
    NexusDepartment   — department records (one ApertureDB DB user each)
    NexusUser         — app-level Principal records (hashed api_key)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Optional

from aperture_nexus._client import get_connector
from aperture_nexus.auth import Principal, validate_credentials
from aperture_nexus.config import load_config
from aperture_nexus.exceptions import (
    NexusConnectionError,
    NexusPermissionError,
    NexusStorageError,
    NexusValidationError,
)

logger = logging.getLogger(__name__)

# ApertureDB entity class names
_CLASS_ORG = "NexusOrganization"
_CLASS_DEPT = "NexusDepartment"
_CLASS_USER = "NexusUser"

# ApertureDB permissions granted to department DB users
_DEPT_PERMISSIONS = ["EntityRead", "EntityWrite",
                     "ConnectionRead", "ConnectionWrite",
                     "DescriptorRead", "DescriptorWrite",
                     "BlobRead", "BlobWrite"]


def _hash_key(api_key: str) -> str:
    """Return a SHA-256 hex digest of the given api_key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _check_response(response: list, operation: str) -> None:
    """Raise NexusStorageError if any command in the response failed."""
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
            "class": entity_class,
            "constraints": constraints,
            "results": {"count": True},
        }
    }]
    response, _ = db.query(cmd)
    body = response[0].get("FindEntity", {})
    return body.get("count", 0) > 0


class NexusAdmin:
    """Identity authority for aperture-nexus.

    Creates and manages departments (ApertureDB DB users) and app-level
    Principals (stored as ``NexusUser`` entities with hashed credentials).
    Requires admin-level ApertureDB credentials.

    On first use, ``NexusAdmin`` ensures the default organization and
    department exist in ApertureDB (configurable via ``aperture_nexus.json``
    under ``admin.default_organization`` and ``admin.default_department``).

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
        admin.create_department("support", organization="AcmeCorp")
        api_key = admin.create_principal(
            user_id="alice",
            user_name="Alice Chen",
            department="support",
            organization="AcmeCorp",
        )
        principal = admin.authenticate(user_id="alice", api_key=api_key)
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

    def create_department(
        self,
        department: str,
        organization: Optional[str] = None,
    ) -> None:
        """Create a department entity in ApertureDB.

        If a department with this name already exists, this is a no-op.

        In a full deployment, this also creates an ApertureDB DB user for
        the department. That capability depends on the ApertureDB edition
        and is handled separately from the entity record.

        Args:
            department: Department name (e.g. ``"support"``).
            organization: Organization this department belongs to. If
                ``None``, uses the configured default organization.

        Raises:
            NexusValidationError: If department is empty.
            NexusStorageError: If ApertureDB rejects the write.

        Example:
            admin.create_department("support", organization="AcmeCorp")
        """
        if not isinstance(department, str) or not department.strip():
            raise NexusValidationError(
                "department must be a non-empty string."
            )
        self._ensure_defaults()
        org = organization or self._cfg.admin.default_organization

        if _entity_exists(
            self._db, _CLASS_DEPT, {"name": ["==", department]}
        ):
            logger.debug(
                "Department %r already exists — skipping create", department
            )
            return

        cmd = [{
            "AddEntity": {
                "class": _CLASS_DEPT,
                "properties": {
                    "name": department,
                    "organization": org,
                },
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"create_department({department!r})")
        logger.debug("Created department %r in org %r", department, org)

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
                department="support",
                organization="AcmeCorp",
            )
            # Deliver api_key to alice — it cannot be recovered later
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
                "user_id must be a non-empty string."
            )
        if not _entity_exists(
            self._db, _CLASS_USER, {"user_id": ["==", user_id]}
        ):
            raise NexusValidationError(
                f"Principal {user_id!r} does not exist."
            )

        cmd = [{
            "DeleteEntity": {
                "class": _CLASS_USER,
                "constraints": {"user_id": ["==", user_id]},
            }
        }]
        response, _ = self._db.query(cmd)
        _check_response(response, f"delete_principal({user_id!r})")
        logger.debug("Deleted principal %r", user_id)

    def authenticate(self, user_id: str, api_key: str) -> Principal:
        """Validate credentials and return an authenticated Principal.

        Looks up the ``NexusUser`` entity for ``user_id`` and compares
        the SHA-256 hash of ``api_key`` against the stored hash.

        Args:
            user_id: The user's unique identifier.
            api_key: The API key issued by ``create_principal()``.

        Returns:
            An authenticated ``Principal`` ready to be attached to a
            ``Context``.

        Raises:
            NexusPermissionError: If credentials are invalid or the
                user does not exist.
            NexusConnectionError: If ApertureDB is unreachable.

        Example:
            principal = admin.authenticate(
                user_id="alice", api_key="..."
            )
            ctx = Context(principal=principal, session_name="s-001")
        """
        user_id = validate_credentials(user_id, api_key)

        cmd = [{
            "FindEntity": {
                "class": _CLASS_USER,
                "constraints": {"user_id": ["==", user_id]},
                "results": {"all_properties": True},
            }
        }]
        try:
            response, _ = self._db.query(cmd)
        except Exception as e:
            raise NexusConnectionError(
                f"ApertureDB query failed during authentication: {e}. "
                f"Run 'adb-nexus validate' to check your connection."
            ) from e

        body = response[0].get("FindEntity", {})
        entities = body.get("entities", [])

        if not entities:
            raise NexusPermissionError(
                f"Authentication failed for user_id={user_id!r}. "
                f"User does not exist or credentials are invalid."
            )

        record = entities[0]
        stored_hash = record.get("api_key_hash", "")
        if stored_hash != _hash_key(api_key):
            raise NexusPermissionError(
                f"Authentication failed for user_id={user_id!r}. "
                f"Invalid credentials."
            )

        principal = Principal(
            user_id=record["user_id"],
            user_name=record.get("user_name"),
            department=record.get("department"),
            organization=record.get("organization"),
        )
        logger.debug("Authenticated principal %r", user_id)
        return principal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_defaults(self) -> None:
        """Create default org and dept entities if they don't exist."""
        if self._defaults_ensured:
            return

        default_org = self._cfg.admin.default_organization
        default_dept = self._cfg.admin.default_department

        if not _entity_exists(
            self._db, _CLASS_ORG, {"name": ["==", default_org]}
        ):
            cmd = [{
                "AddEntity": {
                    "class": _CLASS_ORG,
                    "properties": {"name": default_org},
                }
            }]
            response, _ = self._db.query(cmd)
            _check_response(response, "ensure default organization")
            logger.debug("Created default organization %r", default_org)

        if not _entity_exists(
            self._db, _CLASS_DEPT, {"name": ["==", default_dept]}
        ):
            cmd = [{
                "AddEntity": {
                    "class": _CLASS_DEPT,
                    "properties": {
                        "name": default_dept,
                        "organization": default_org,
                    },
                }
            }]
            response, _ = self._db.query(cmd)
            _check_response(response, "ensure default department")
            logger.debug("Created default department %r", default_dept)

        self._defaults_ensured = True
