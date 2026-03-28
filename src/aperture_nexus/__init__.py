"""
aperture-nexus: The Cognition Engine for Enterprise AI.

Enables enterprise AI agents to establish relations for context,
capture knowledge across text, images, video, and documents, and
commit it to memory for search and retrieval when needed.

Quickstart:
    >>> import os
    >>> from aperture_nexus import Memory, Context, Information
    >>> from aperture_nexus import generate_session_id
    >>> memory = Memory()
    >>> principal = memory.authenticate(
    ...     user_id="alice", api_key=os.environ["NEXUS_API_KEY"]
    ... )
    >>> ctx = Context(
    ...     principal=principal, session_name="my-session", purpose="..."
    ... )
    >>> info = Information(context_id=ctx.id)
    >>> info.log(text="Hello, world")
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
    "MemoryEntry",
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
    if name == "MemoryEntry":
        from aperture_nexus.memory import MemoryEntry
        return MemoryEntry
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
