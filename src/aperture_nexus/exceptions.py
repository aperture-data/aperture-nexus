"""
aperture-nexus exception hierarchy.

All exceptions raised by aperture-nexus are subclasses of NexusError,
allowing callers to catch broadly or specifically:

    try:
        memory.commit(ctx, info)
    except NexusPermissionError:
        # handle permission violation
    except NexusConnectionError:
        # handle DB unreachable
    except NexusError:
        # catch anything else from aperture-nexus

Every exception includes:
- A clear message describing what went wrong
- Why it went wrong (where known)
- What the caller can do to fix it

Original exceptions are always chained (raise ... from e) so the full
traceback is preserved for debugging.
"""


class NexusError(Exception):
    """
    Base exception for all aperture-nexus errors.

    Also raised directly for unexpected errors where a more specific
    subclass does not apply. Always chains the original exception so
    the full traceback is preserved.

    Example:
        >>> try:
        ...     memory.commit(ctx, info)
        ... except NexusError as e:
        ...     print(f"aperture-nexus error: {e}")
    """


class NexusConfigError(NexusError):
    """
    Raised when aperture-nexus is misconfigured.

    Common causes:
    - aperture_nexus.json not found and no defaults available
    - Required model not configured when process_and_commit() is called
    - Optional dependency not installed for an enabled feature
    - UI hosted without an api_key set

    Example:
        >>> # No models configured, but process_and_commit() called
        >>> memory.process_and_commit(ctx, info)
        NexusConfigError: No embedding model resolved for image input.
        Provide embedding_model='your-model' in log(), or add a "models"
        section to aperture_nexus.json. Run 'adb-nexus init' to regenerate
        your config.
    """


class NexusValidationError(NexusError):
    """
    Raised when input data fails validation.

    Validation happens eagerly at Information.log() time — before any
    DB interaction — so errors surface as close to the problem as possible.

    Common causes:
    - Unsupported input type passed to log()
    - numpy array with wrong shape or dtype for an image/embedding
    - File path does not exist or is not readable
    - URL is not reachable
    - Embedding dimensions do not match the configured DescriptorSet

    Example:
        >>> info.log(image="/nonexistent/path.jpg")
        NexusValidationError: Image file not found: /nonexistent/path.jpg.
        Provide a valid file path, a URL, a PIL Image, a numpy array,
        or raw bytes.
    """


class NexusConnectionError(NexusError):
    """
    Raised when aperture-nexus cannot connect to ApertureDB.

    Common causes:
    - ApertureDB is not running at the configured host/port
    - APERTUREDB_KEY is invalid or expired
    - Network unreachable

    Example:
        >>> memory = Memory()
        NexusConnectionError: Could not connect to ApertureDB at
        localhost:55555. Verify ApertureDB is running and your credentials
        are correct. Run 'adb-nexus validate' to test your connection.
    """


class NexusPermissionError(NexusError):
    """
    Raised when an operation is denied due to insufficient permissions.

    Permissions in aperture-nexus are scoped to a Principal (authenticated
    user) and enforced at commit, search, and remove time. Local and global
    restrictions on a Context may also trigger this error.

    Example:
        >>> memory.search(query=image, filters={})
        NexusPermissionError: Principal 'alice' does not have search
        permission in this context. Check the restrictions configured
        on the Context or contact your administrator.
    """


class NexusProcessingError(NexusError):
    """
    Raised when model processing fails during process_and_commit() or
    async_process_and_commit().

    Common causes:
    - LLM/VLM model call failed or timed out
    - Model returned unexpected output format
    - Embedding generation failed for a specific modality

    The MemoryTask.error and MemoryTask.error_message fields provide
    details when this occurs in an async context.

    Example:
        >>> memory.process_and_commit(ctx, info)
        NexusProcessingError: Embedding generation failed for image input
        using model 'clip-vit-base'. Check model availability and input
        format. Original error: <original exception>
    """


class NexusStorageError(NexusError):
    """
    Raised when ApertureDB rejects a write operation.

    Common causes:
    - ApertureDB schema constraint violation
    - Duplicate entity with unique constraint
    - Write quota exceeded
    - ApertureDB returned an error status for a query

    Example:
        >>> memory.commit(ctx, info)
        NexusStorageError: ApertureDB rejected the write operation.
        Status: -1. Details: <ApertureDB error message>.
        Check your schema and retry. If the problem persists, run
        'adb-nexus validate' to verify your connection and schema.
    """
