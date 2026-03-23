"""
aperture-nexus: The Unified KMC (Knowledge, Memory, Context) Engine
for Agentic State.

Quickstart:
    >>> from aperture_nexus import NexusAdmin, Memory, Context, Information
    >>> from aperture_nexus import generate_session_id
    >>> admin = NexusAdmin()
    >>> principal = admin.authenticate(user_id="alice", api_key="...")
    >>> ctx = Context(
    ...     principal=principal, session_name="my-session", purpose="..."
    ... )
    >>> info = Information(context_id=ctx.id)
    >>> info.log(text="Hello, world")
    >>> memory = Memory()
    >>> memory.commit(ctx, info)
"""

import uuid

from aperture_nexus.exceptions import (
    NexusError,
    NexusConfigError,
    NexusValidationError,
    NexusConnectionError,
    NexusPermissionError,
    NexusProcessingError,
    NexusStorageError,
)

__version__ = "0.1.0"

__all__ = [
    # Core API — import these
    "Memory",
    "Context",
    "Information",
    "NexusAdmin",
    "SearchResult",
    "MemoryTask",
    # Utilities
    "generate_session_id",
    # Exceptions
    "NexusError",
    "NexusConfigError",
    "NexusValidationError",
    "NexusConnectionError",
    "NexusPermissionError",
    "NexusProcessingError",
    "NexusStorageError",
]


def generate_session_id(prefix: str = "") -> str:
    """Generate a unique session ID.

    No DB access required. Use as the ``session_id`` argument to
    ``Context`` when you want a stable, portable ID across services.

    Args:
        prefix: Optional string prepended to the ID, separated by ``-``.
            Useful for debugging (e.g. ``"support"`` → ``"support-a3f7c..."``).

    Returns:
        A unique string session ID.

    Example:
        sid = generate_session_id()
        sid = generate_session_id(prefix="support")
        ctx = Context(principal=principal, session_id=sid)
    """
    uid = uuid.uuid4().hex
    return f"{prefix}-{uid}" if prefix else uid


# Deferred imports to avoid loading heavy dependencies at import time.
def __getattr__(name: str):
    if name == "Memory":
        from aperture_nexus.memory import Memory
        return Memory
    if name == "SearchResult":
        from aperture_nexus.memory import SearchResult
        return SearchResult
    if name == "Context":
        from aperture_nexus.context import Context
        return Context
    if name == "Information":
        from aperture_nexus.information import Information
        return Information
    if name == "NexusAdmin":
        from aperture_nexus.admin import NexusAdmin
        return NexusAdmin
    if name == "MemoryTask":
        from aperture_nexus.tasks import MemoryTask
        return MemoryTask
    raise AttributeError(f"module 'aperture_nexus' has no attribute {name!r}")
