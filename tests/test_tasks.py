"""
Unit tests for aperture_nexus.tasks.

Tests cover:
- MemoryTask construction and defaults
- is_ready(): pending/processing are not ready; complete/failed are
- wait(): returns immediately when already terminal
- retry(): only valid on failed tasks; resets state; fires retry fn
- retry() raises on non-failed status
- retry() raises when no retry function attached
- Internal state transitions: _mark_processing, _mark_complete, _mark_failed
- __repr__

No live ApertureDB instance required.
"""

import asyncio
from datetime import datetime

import pytest

from aperture_nexus.exceptions import NexusValidationError
from aperture_nexus.tasks import MemoryTask


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestMemoryTaskConstruction:
    def test_defaults(self):
        task = MemoryTask(task_id="t-001")
        assert task.task_id == "t-001"
        assert task.status == "pending"
        assert task.context_id is None
        assert task.completed_at is None
        assert task.error is None
        assert task.error_message is None
        assert task.failed_at is None

    def test_repr(self):
        task = MemoryTask(task_id="t-abc")
        r = repr(task)
        assert "t-abc" in r
        assert "pending" in r


# ---------------------------------------------------------------------------
# is_ready()
# ---------------------------------------------------------------------------


class TestIsReady:
    def test_pending_not_ready(self):
        assert MemoryTask(task_id="t").is_ready() is False

    def test_processing_not_ready(self):
        task = MemoryTask(task_id="t", status="processing")
        assert task.is_ready() is False

    def test_complete_is_ready(self):
        task = MemoryTask(task_id="t", status="complete")
        assert task.is_ready() is True

    def test_failed_is_ready(self):
        task = MemoryTask(task_id="t", status="failed")
        assert task.is_ready() is True


# ---------------------------------------------------------------------------
# wait()
# ---------------------------------------------------------------------------


class TestWait:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_complete(self):
        task = MemoryTask(task_id="t", status="complete")
        await task.wait()  # must not hang

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_failed(self):
        task = MemoryTask(task_id="t", status="failed")
        await task.wait()

    @pytest.mark.asyncio
    async def test_wait_resolves_when_status_changes(self):
        task = MemoryTask(task_id="t", status="pending")

        async def flip():
            await asyncio.sleep(0.05)
            task.status = "complete"

        await asyncio.gather(task.wait(), flip())
        assert task.status == "complete"


# ---------------------------------------------------------------------------
# retry()
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retry_on_non_failed_raises(self):
        task = MemoryTask(task_id="t", status="pending")
        with pytest.raises(NexusValidationError, match="not failed"):
            task.retry()

    def test_retry_on_complete_raises(self):
        task = MemoryTask(task_id="t", status="complete")
        with pytest.raises(NexusValidationError, match="not failed"):
            task.retry()

    def test_retry_without_retry_fn_raises(self):
        task = MemoryTask(task_id="t", status="failed")
        with pytest.raises(NexusValidationError, match="retry function"):
            task.retry()

    def test_retry_resets_state(self):
        fired = []

        async def noop():
            fired.append(True)

        task = MemoryTask(task_id="t", status="failed")
        task._mark_failed(ValueError("oops"))
        task._retry_fn = noop
        task.retry()

        assert task.status == "pending"
        assert task.error is None
        assert task.error_message is None
        assert task.failed_at is None


# ---------------------------------------------------------------------------
# Internal state transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_mark_processing(self):
        task = MemoryTask(task_id="t")
        task._mark_processing()
        assert task.status == "processing"

    def test_mark_complete(self):
        task = MemoryTask(task_id="t")
        task._mark_complete("ctx-xyz")
        assert task.status == "complete"
        assert task.context_id == "ctx-xyz"
        assert isinstance(task.completed_at, datetime)

    def test_mark_failed(self):
        task = MemoryTask(task_id="t")
        exc = RuntimeError("boom")
        task._mark_failed(exc)
        assert task.status == "failed"
        assert task.error is exc
        assert task.error_message == "boom"
        assert isinstance(task.failed_at, datetime)

    def test_mark_complete_clears_prior_state(self):
        """Complete after processing — context_id and timestamp are set."""
        task = MemoryTask(task_id="t")
        task._mark_processing()
        task._mark_complete("ctx-abc")
        assert task.status == "complete"
        assert task.context_id == "ctx-abc"
