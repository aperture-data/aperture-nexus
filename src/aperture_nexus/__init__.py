"""
aperture-nexus: The Unified KMC (Knowledge, Memory, Context) Engine for Agentic State.

Quickstart:
    >>> from aperture_nexus import Memory, Context, Information
    >>> memory = Memory()
    >>> principal = memory.authenticate(user_id="alice", api_key="...")
    >>> ctx = Context(principal=principal, session_name="my-session", purpose="...")
    >>> info = Information(context_id=ctx.id)
    >>> info.log(text="Hello, world")
    >>> memory.commit(ctx, info)
"""

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
    # Exceptions
    "NexusError",
    "NexusConfigError",
    "NexusValidationError",
    "NexusConnectionError",
    "NexusPermissionError",
    "NexusProcessingError",
    "NexusStorageError",
]

# Deferred imports to avoid loading heavy dependencies at import time.
# Memory, Context, and Information are imported on first use.
def __getattr__(name: str):
    if name == "Memory":
        from aperture_nexus.memory import Memory
        return Memory
    if name == "Context":
        from aperture_nexus.context import Context
        return Context
    if name == "Information":
        from aperture_nexus.information import Information
        return Information
    raise AttributeError(f"module 'aperture_nexus' has no attribute {name!r}")
