"""
MemoryTask — async processing task for Memory.async_process_and_commit().

Tasks are created by async_process_and_commit() and run in the background.
Their state is tracked in-memory only; it does not survive process restarts.
Callers poll via is_ready(), block via await task.wait(), or resubmit via
task.retry() after failure.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

from aperture_nexus.exceptions import NexusValidationError

if TYPE_CHECKING:
    pass  # avoid circular imports; Memory is referenced by string only

logger = logging.getLogger(__name__)

# Valid task status values.
_PENDING = "pending"
_PROCESSING = "processing"
_COMPLETE = "complete"
_FAILED = "failed"
_TERMINAL = {_COMPLETE, _FAILED}


@dataclass
class MemoryTask:
    """An async memory processing task.

    Returned by ``Memory.async_process_and_commit()``. The task runs in the
    background; its status is persisted in ApertureDB and survives process
    restarts.

    Do not construct directly — use ``Memory.async_process_and_commit()``.

    Attributes:
        task_id: Unique identifier for this task (UUID).
        status: Current state: ``"pending"`` | ``"processing"``
            | ``"complete"`` | ``"failed"``. Held in memory only —
            not persisted to ApertureDB and not visible across processes
            or restarts.
        context_id: The committed context ID. Available only when
            ``status == "complete"``.
        completed_at: Completion timestamp. Available only when
            ``status == "complete"``.
        error: The exception that caused failure. Available only when
            ``status == "failed"``.
        error_message: Human-readable failure description. Available
            only when ``status == "failed"``.
        failed_at: Failure timestamp. Available only when
            ``status == "failed"``.

    Example:
        task = await memory.async_process_and_commit(ctx, info)
        await task.wait()

        if task.status == "complete":
            print(f"context_id: {task.context_id}")
        elif task.status == "failed":
            print(f"failed: {task.error_message}")
            task.retry()
    """

    task_id: str
    status: str = _PENDING
    context_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    failed_at: Optional[datetime] = None

    # Internal: callable that resubmits this task. Injected by Memory.
    _retry_fn: Optional[Callable[[], Coroutine[Any, Any, None]]] = field(
        default=None, repr=False, compare=False
    )

    def is_ready(self) -> bool:
        """Return True if the task has reached a terminal state.

        Returns:
            True when status is ``"complete"`` or ``"failed"``.

        Example:
            while not task.is_ready():
                await asyncio.sleep(0.1)
        """
        return self.status in _TERMINAL

    async def wait(self) -> None:
        """Block asynchronously until the task reaches a terminal state.

        Polls every 500 ms. Returns immediately if already complete or
        failed.

        Example:
            await task.wait()
            assert task.status in ("complete", "failed")
        """
        while not self.is_ready():
            await asyncio.sleep(0.5)

    async def retry(self) -> None:
        """Resubmit a failed task for processing.

        Only valid when ``status == "failed"``. Resets the task to
        ``"pending"`` and requeues it in the background.

        Raises:
            NexusValidationError: If the task is not in ``"failed"``
                state.

        Example:
            for task in memory.failed_commits():
                await task.retry()
        """
        if self.status != _FAILED:
            raise NexusValidationError(
                f"Cannot retry a task that is not failed. "
                f"Current status: {self.status!r}. "
                f"Only tasks with status 'failed' can be retried."
            )
        if self._retry_fn is None:
            raise NexusValidationError(
                "This task has no retry function attached. "
                "It may have been loaded from ApertureDB without "
                "a live Memory reference."
            )
        self.status = _PENDING
        self.error = None
        self.error_message = None
        self.failed_at = None
        logger.debug("Retrying task %r", self.task_id)
        await self._retry_fn()

    # ------------------------------------------------------------------
    # Internal state transitions — called by Memory, not by callers
    # ------------------------------------------------------------------

    def _mark_processing(self) -> None:
        self.status = _PROCESSING
        logger.debug("Task %r: processing", self.task_id)

    def _mark_complete(self, context_id: str) -> None:
        self.status = _COMPLETE
        self.context_id = context_id
        self.completed_at = datetime.utcnow()
        logger.debug(
            "Task %r: complete (context_id=%r)", self.task_id, context_id
        )

    def _mark_failed(self, exc: Exception) -> None:
        self.status = _FAILED
        self.error = exc
        self.error_message = str(exc)
        self.failed_at = datetime.utcnow()
        logger.debug(
            "Task %r: failed (%s)", self.task_id, self.error_message
        )

    def __repr__(self) -> str:
        return (
            f"MemoryTask(task_id={self.task_id!r},"
            f" status={self.status!r})"
        )
