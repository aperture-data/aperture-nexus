"""
Internal ApertureDB connection management for aperture-nexus.

This module is intentionally private. Users interact with ApertureDB through
the Memory class — do not import this module directly in application code.

Connection priority (delegated to aperturedb.CommonLibrary.create_connector):
  1. Caller-provided Connector object (pass-through for testing or reuse)
  2. APERTUREDB_KEY environment variable (encoded connection key)
  3. APERTUREDB_KEY Google Colab secret
  4. APERTUREDB_JSON environment variable (JSON connection config)
  5. APERTUREDB_JSON Google Colab secret
  6. APERTUREDB_JSON secret in a .env file
  7. APERTUREDB_CONFIG environment variable (named config)
  8. Active adb configuration (set via 'adb config')

All connection errors are wrapped as NexusConnectionError with clear, actionable
messages so callers do not need to handle ApertureDB internals directly.
"""

import logging
from typing import Optional, Union

from aperturedb.CommonLibrary import create_connector
from aperturedb.Connector import (
    Connector,
    UnauthorizedException,
    UnauthenticatedException,
)

from aperture_nexus.exceptions import NexusConnectionError

logger = logging.getLogger(__name__)

# Standard ApertureDB health-check command.
_STATUS_QUERY = [{"GetStatus": {}}]


def get_connector(db: Optional[Connector] = None) -> Connector:
    """Return an ApertureDB Connector, creating one if none is provided.

    When ``db`` is given it is returned as-is — useful for testing, for
    reusing an existing connection, or for injecting a pre-configured client.
    When ``db`` is ``None``, a connector is built via
    ``aperturedb.CommonLibrary.create_connector()``, which resolves credentials
    from environment variables or the active ``adb`` configuration.

    Note:
        ApertureDB establishes the actual TCP connection on the first
        ``connector.query()`` call, not here. Use :func:`validate_connection`
        to eagerly verify that ApertureDB is reachable.

    Args:
        db: An existing ApertureDB Connector to use instead of creating a new
            one. Pass ``None`` (default) to resolve from the environment.

    Returns:
        A ready-to-use ApertureDB Connector.

    Raises:
        NexusConnectionError: If no credentials are available or the
            configuration is malformed.

    Example:
        # Credentials from environment (typical usage)
        connector = get_connector()

        # Inject an existing connector (testing / connection reuse)
        connector = get_connector(db=my_existing_connector)
    """
    if db is not None:
        logger.debug("Using caller-provided ApertureDB Connector")
        return db

    logger.debug("Creating ApertureDB connector from environment / adb config")
    try:
        connector = create_connector()
    except AssertionError as e:
        # create_connector() uses assert statements for config errors.
        raise NexusConnectionError(
            f"Could not find ApertureDB credentials. "
            f"Set APERTUREDB_KEY, or set APERTUREDB_HOST / APERTUREDB_PORT / "
            f"APERTUREDB_USER / APERTUREDB_PASSWORD environment variables, "
            f"or run 'adb config' to create a saved configuration. "
            f"Run 'adb-nexus validate' to test your setup. "
            f"Details: {e}"
        ) from e
    except Exception as e:
        raise NexusConnectionError(
            f"Unexpected error while creating ApertureDB connector: {e}. "
            f"Run 'adb-nexus validate' to test your setup."
        ) from e

    logger.debug(
        "ApertureDB connector created for %s", connection_description(connector)
    )
    return connector


def validate_connection(connector: Connector) -> None:
    """Verify that ApertureDB is reachable and the credentials are accepted.

    Sends a ``GetStatus`` query — the standard ApertureDB health check — and
    raises :exc:`NexusConnectionError` if the query fails or returns a
    non-zero status code.

    Args:
        connector: The ApertureDB Connector to test.

    Raises:
        NexusConnectionError: If ApertureDB is unreachable, authentication
            fails, or the server returns an error status.

    Example:
        connector = get_connector()
        validate_connection(connector)  # raises NexusConnectionError if broken
        print("ApertureDB is reachable")
    """
    target = connection_description(connector)
    logger.debug("Validating connection to ApertureDB at %s", target)

    try:
        response, _ = connector.query(_STATUS_QUERY)
    except (ConnectionError, OSError) as e:
        raise NexusConnectionError(
            f"ApertureDB at {target} is unreachable. "
            f"Verify ApertureDB is running and network access is available. "
            f"Run 'adb-nexus validate' for a full connection check. "
            f"Details: {e}"
        ) from e
    except UnauthorizedException as e:
        raise NexusConnectionError(
            f"ApertureDB at {target} rejected the credentials (unauthorized). "
            f"Check your APERTUREDB_KEY or username/password. "
            f"Run 'adb-nexus validate' for a full connection check."
        ) from e
    except UnauthenticatedException as e:
        raise NexusConnectionError(
            f"ApertureDB at {target} could not authenticate. "
            f"Check your APERTUREDB_KEY or username/password. "
            f"Run 'adb-nexus validate' for a full connection check."
        ) from e
    except Exception as e:
        raise NexusConnectionError(
            f"Unexpected error while contacting ApertureDB at {target}: {e}. "
            f"Run 'adb-nexus validate' for a full connection check."
        ) from e

    # Inspect the GetStatus response body.
    status_body = _parse_status_response(response)
    status_code = status_body.get("status", -1)
    if status_code != 0:
        info = status_body.get("info", "no details returned")
        raise NexusConnectionError(
            f"ApertureDB at {target} returned a non-zero status during the "
            f"connection check (status={status_code}, info={info!r}). "
            f"Run 'adb-nexus validate' for a full connection check."
        )

    logger.debug("ApertureDB connection verified at %s", target)


def connection_description(connector: Connector) -> str:
    """Return a human-readable description of the connection target.

    Used in log messages and error strings so operators know which ApertureDB
    instance is involved without exposing credentials.

    Args:
        connector: The ApertureDB Connector to describe.

    Returns:
        A string of the form ``"host:port"`` (e.g. ``"localhost:55555"``),
        or ``"<ApertureDB>"`` if the host/port cannot be determined.

    Example:
        desc = connection_description(connector)
        # "localhost:55555"
    """
    try:
        return f"{connector.host}:{connector.port}"
    except AttributeError:
        return "<ApertureDB>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_status_response(response: list) -> dict:
    """Extract the GetStatus body from a connector.query() response.

    ApertureDB returns a list of per-command response dicts. For a GetStatus
    query the first element should have a "GetStatus" key.

    Args:
        response: Raw response list from ``connector.query()``.

    Returns:
        The GetStatus body dict, or an empty dict if parsing fails.
    """
    if not response or not isinstance(response, list):
        return {}
    first = response[0]
    if not isinstance(first, dict):
        return {}
    return first.get("GetStatus", {})
