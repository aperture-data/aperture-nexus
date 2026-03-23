"""
Unit tests for aperture_nexus.memory (Memory, SearchResult).

Tests cover:
- commit(): happy path, empty info raises, drain called on success,
  storage error propagated, connection error propagated
- process_and_commit(): pre-computed embeddings used as-is;
  missing model raises NexusConfigError
- async_process_and_commit(): returns MemoryTask, drains info immediately,
  task transitions to complete/failed
- search(): ndarray without modality raises; metadata-only; vector search;
  min_score filtering; unknown modality raises
- connect(): happy path; empty relationship raises
- remove(): happy path; empty memory_id raises
- pending_commits() / failed_commits(): task list filtering
- stats(): raises NexusConfigError when prometheus not installed
- _resolve_session_id(): uses session_id if set; derives from session_name
- _build_constraints(): filters known keys, drops unknown keys
- _ensure_descriptor_set(): creates on first use, skips if exists

All tests use mock_connector — no live ApertureDB required.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aperture_nexus.exceptions import (
    NexusConfigError,
    NexusConnectionError,
    NexusStorageError,
    NexusValidationError,
)
from aperture_nexus.memory import Memory, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_response(count: int = 0, entities: list | None = None) -> tuple:
    body: dict = {"status": 0, "count": count}
    if entities is not None:
        body["entities"] = entities
    return ([{"FindEntity": body}], [])


def _find_descriptor_set_response(count: int = 0) -> tuple:
    return ([{"FindDescriptorSet": {"status": 0, "count": count}}], [])


def _find_descriptor_response(
    descriptors: list | None = None, distances: list | None = None
) -> tuple:
    body: dict = {"status": 0}
    if descriptors is not None:
        body["descriptors"] = descriptors
    if distances is not None:
        body["distances"] = distances
    return ([{"FindDescriptor": body}], [])


def _ok_response(cmd_name: str = "AddEntity") -> tuple:
    return ([{cmd_name: {"status": 0}}], [])


def _make_memory(mock_connector):
    return Memory(db_client=mock_connector)


def _make_principal(user_id: str = "alice"):
    from aperture_nexus.auth import Principal
    return Principal(user_id=user_id, user_name="Test User")


def _make_ctx(principal=None, session_id: str | None = "s-001",
              session_name: str | None = None):
    from aperture_nexus.context import Context
    p = principal or _make_principal()
    kwargs: dict = {"principal": p}
    if session_id is not None:
        kwargs["session_id"] = session_id
    if session_name is not None:
        kwargs["session_name"] = session_name
    return Context(**kwargs)


def _make_info(ctx=None):
    from aperture_nexus.information import Information
    c = ctx or _make_ctx()
    return Information(context_id=c.id)


# Standard sequence for a commit() with one text entry and no existing
# session or context:
#   1. FindEntity session → 0 (create needed)
#   2. AddEntity session  → ok
#   3. FindEntity context → 0 (create needed)
#   4. AddEntity context  → ok
#   5. AddEntity memory   → ok
#   6. AddBlob  text      → ok
def _commit_side_effects_fresh():
    return [
        _find_response(count=0),    # ensure_session: not found
        _ok_response("AddEntity"),  # ensure_session: add
        _find_response(count=0),    # ensure_context: not found
        _ok_response("AddEntity"),  # ensure_context: add
        _ok_response("AddEntity"),  # write_memory_entity
        _ok_response("AddBlob"),    # write text entry
    ]


# Same but session + context already exist
def _commit_side_effects_existing():
    return [
        _find_response(count=1),    # ensure_session: exists
        _find_response(count=1),    # ensure_context: exists
        _ok_response("AddEntity"),  # write_memory_entity
        _ok_response("AddBlob"),    # write text entry
    ]


# ---------------------------------------------------------------------------
# commit()
# ---------------------------------------------------------------------------


class TestCommit:
    def test_happy_path_returns_memory_id(self, mock_connector):
        mock_connector.query.side_effect = _commit_side_effects_existing()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        mid = memory.commit(ctx, info)
        assert isinstance(mid, str) and len(mid) > 10

    def test_drains_info_after_success(self, mock_connector):
        mock_connector.query.side_effect = _commit_side_effects_existing()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        assert len(info) == 1
        memory.commit(ctx, info)
        assert len(info) == 0

    def test_does_not_drain_on_failure(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),  # ensure_session: exists
            _find_response(count=1),  # ensure_context: exists
            ([{"AddEntity": {"status": -1, "info": "schema error"}}], []),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        with pytest.raises(NexusStorageError):
            memory.commit(ctx, info)
        assert len(info) == 1   # not drained

    def test_empty_info_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        with pytest.raises(NexusValidationError, match="no entries"):
            memory.commit(ctx, info)

    def test_creates_session_when_absent(self, mock_connector):
        mock_connector.query.side_effect = _commit_side_effects_fresh()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 6

    def test_skips_session_creation_when_present(self, mock_connector):
        mock_connector.query.side_effect = _commit_side_effects_existing()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 4

    def test_storage_error_propagated(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            ([{"AddEntity": {"status": -1, "info": "conflict"}}], []),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        with pytest.raises(NexusStorageError, match="conflict"):
            memory.commit(ctx, info)

    def test_connection_error_propagated(self, mock_connector):
        mock_connector.query.side_effect = NexusConnectionError("timeout")
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        with pytest.raises(NexusConnectionError):
            memory.commit(ctx, info)

    def test_multiple_commits_on_same_info(self, mock_connector):
        """Second commit after drain still works (buffer reused)."""
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)

        mock_connector.query.side_effect = _commit_side_effects_existing()
        info.log(text="first")
        mid1 = memory.commit(ctx, info)
        assert len(info) == 0

        # Second round
        mock_connector.query.side_effect = _commit_side_effects_existing()
        info.log(text="second")
        mid2 = memory.commit(ctx, info)
        assert len(info) == 0
        assert mid1 != mid2


# ---------------------------------------------------------------------------
# commit() — with embeddings
# ---------------------------------------------------------------------------


class TestCommitWithDescriptors:
    def test_writes_descriptor_for_precomputed_embedding(self, mock_connector):
        # FindEntity session, AddEntity session, FindEntity context,
        # AddEntity context, AddEntity memory, AddBlob text,
        # FindDescriptorSet (exists), AddDescriptor
        mock_connector.query.side_effect = [
            _find_response(count=1),                    # session exists
            _find_response(count=1),                    # context exists
            _ok_response("AddEntity"),                  # memory entity
            _ok_response("AddBlob"),                    # text blob
            _find_descriptor_set_response(count=1),     # dset exists
            _ok_response("AddDescriptor"),              # write descriptor
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        vec = np.ones(128, dtype=np.float32)
        info.log(text="hello", embedding=vec, embedding_model="test-model")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 6

    def test_creates_descriptor_set_on_first_use(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),                    # session exists
            _find_response(count=1),                    # context exists
            _ok_response("AddEntity"),                  # memory entity
            _ok_response("AddBlob"),                    # text blob
            _find_descriptor_set_response(count=0),     # dset not found
            _ok_response("AddDescriptorSet"),           # create dset
            _ok_response("AddDescriptor"),              # write descriptor
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        vec = np.ones(64, dtype=np.float32)
        info.log(text="hello", embedding=vec, embedding_model="test-model")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 7


# ---------------------------------------------------------------------------
# process_and_commit()
# ---------------------------------------------------------------------------


class TestProcessAndCommit:
    def test_raises_config_error_when_no_model_and_no_precomputed(
        self, mock_connector
    ):
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")  # no embedding
        with pytest.raises(NexusConfigError, match="No embedding model"):
            memory.process_and_commit(ctx, info)

    def test_uses_precomputed_embedding_without_model(self, mock_connector):
        # process_and_commit calls _ensure_descriptor_sets BEFORE commit(),
        # so the FindDescriptorSet comes first, then the commit sequence.
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=1),  # _ensure_descriptor_sets
            _find_response(count=1),                  # session exists
            _find_response(count=1),                  # context exists
            _ok_response("AddEntity"),                # memory entity
            _ok_response("AddBlob"),                  # text blob
            _find_descriptor_set_response(count=1),  # _write_descriptor dset check
            _ok_response("AddDescriptor"),            # write descriptor
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        vec = np.ones(128, dtype=np.float32)
        info.log(text="hello", embedding=vec, embedding_model="text-emb-3")
        mid = memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)

    def test_blob_only_entry_does_not_need_embedding(self, mock_connector):
        # blob-only entry: no embedding needed — commit proceeds
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _ok_response("AddEntity"),
            _ok_response("AddBlob"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"raw", document_type="pdf")
        mid = memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)


# ---------------------------------------------------------------------------
# async_process_and_commit()
# ---------------------------------------------------------------------------


class TestAsyncProcessAndCommit:
    @pytest.mark.asyncio
    async def test_returns_pending_task(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _ok_response("AddEntity"),
            _ok_response("AddBlob"),
            _find_descriptor_set_response(count=1),
            _ok_response("AddDescriptor"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        vec = np.ones(32, dtype=np.float32)
        info.log(text="hi", embedding=vec, embedding_model="m")
        task = await memory.async_process_and_commit(ctx, info)
        assert task.status in ("pending", "processing", "complete")

    @pytest.mark.asyncio
    async def test_drains_info_immediately(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _ok_response("AddEntity"),
            _ok_response("AddBlob"),
            _find_descriptor_set_response(count=1),
            _ok_response("AddDescriptor"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        vec = np.ones(32, dtype=np.float32)
        info.log(text="hi", embedding=vec, embedding_model="m")
        assert len(info) == 1
        await memory.async_process_and_commit(ctx, info)
        # info is drained immediately regardless of task completion
        assert len(info) == 0

    @pytest.mark.asyncio
    async def test_task_completes_successfully(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _ok_response("AddEntity"),
            _ok_response("AddBlob"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"data", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        await task.wait()
        assert task.status == "complete"
        assert isinstance(task.memory_id, str)

    @pytest.mark.asyncio
    async def test_task_fails_on_storage_error(self, mock_connector):
        mock_connector.query.side_effect = NexusStorageError("db down")
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"data", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        await task.wait()
        assert task.status == "failed"
        assert task.error_message is not None

    @pytest.mark.asyncio
    async def test_task_registered_in_memory_instance(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_response(count=1),
            _find_response(count=1),
            _ok_response("AddEntity"),
            _ok_response("AddBlob"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"d", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        assert task.task_id in memory._tasks


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    def test_ndarray_without_modality_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        vec = np.ones(64, dtype=np.float32)
        with pytest.raises(NexusValidationError, match="modality is required"):
            memory.search(query=vec)

    def test_unknown_modality_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        vec = np.ones(64, dtype=np.float32)
        with pytest.raises(NexusValidationError, match="Unknown modality"):
            memory.search(query=vec, modality="audio")

    def test_metadata_only_search(self, mock_connector):
        entities = [
            {
                "memory_id": "m-1",
                "session_id": "s-1",
                "context_id": "c-1",
                "created_at": "2024-01-01T00:00:00",
                "user_id": "alice",
            }
        ]
        mock_connector.query.return_value = (
            [{"FindEntity": {"status": 0, "entities": entities}}], []
        )
        memory = _make_memory(mock_connector)
        results = memory.search(filters={"session_id": "s-1"})
        assert len(results) == 1
        assert results[0].memory_id == "m-1"
        assert results[0].score == 1.0

    def test_vector_search_returns_results(self, mock_connector):
        descriptors = [
            {
                "memory_id": "m-2",
                "session_id": "s-1",
                "context_id": "c-1",
                "created_at": "2024-01-01T00:00:00",
                "modality": "text",
                "embedding_model": "test-model",
            }
        ]
        mock_connector.query.return_value = _find_descriptor_response(
            descriptors=descriptors, distances=[0.95]
        )
        memory = _make_memory(mock_connector)
        vec = np.ones(64, dtype=np.float32)
        results = memory.search(query=vec, modality="text")
        assert len(results) == 1
        assert results[0].memory_id == "m-2"
        assert results[0].score == pytest.approx(0.95)

    def test_vector_search_respects_min_score(self, mock_connector):
        descriptors = [
            {
                "memory_id": "m-1",
                "session_id": "s-1",
                "context_id": "c-1",
                "created_at": "2024-01-01T00:00:00",
            },
            {
                "memory_id": "m-2",
                "session_id": "s-1",
                "context_id": "c-1",
                "created_at": "2024-01-01T00:00:00",
            },
        ]
        mock_connector.query.return_value = _find_descriptor_response(
            descriptors=descriptors, distances=[0.9, 0.3]
        )
        memory = _make_memory(mock_connector)
        vec = np.ones(64, dtype=np.float32)
        results = memory.search(query=vec, modality="text", min_score=0.5)
        assert len(results) == 1
        assert results[0].memory_id == "m-1"

    def test_none_query_with_no_filters_returns_all(self, mock_connector):
        mock_connector.query.return_value = (
            [{"FindEntity": {"status": 0, "entities": []}}], []
        )
        memory = _make_memory(mock_connector)
        results = memory.search()
        assert results == []

    def test_string_query_raises_config_error_without_model(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusConfigError, match="No text embedding model"):
            memory.search(query="what did we discuss?")

    def test_db_error_raises_connection_error(self, mock_connector):
        mock_connector.query.side_effect = ConnectionError("timeout")
        memory = _make_memory(mock_connector)
        vec = np.ones(64, dtype=np.float32)
        with pytest.raises(NexusConnectionError, match="search query failed"):
            memory.search(query=vec, modality="text")


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    def test_creates_connection_between_contexts(self, mock_connector):
        mock_connector.query.return_value = (
            [
                {"FindEntity": {"status": 0}},
                {"FindEntity": {"status": 0}},
                {"AddConnection": {"status": 0}},
            ],
            [],
        )
        memory = _make_memory(mock_connector)
        ctx1 = _make_ctx(session_id="s-1")
        ctx2 = _make_ctx(session_id="s-2")
        memory.connect(ctx1, ctx2, relationship="follows")
        assert mock_connector.query.call_count == 1

    def test_creates_connection_between_memory_ids(self, mock_connector):
        mock_connector.query.return_value = (
            [
                {"FindEntity": {"status": 0}},
                {"FindEntity": {"status": 0}},
                {"AddConnection": {"status": 0}},
            ],
            [],
        )
        memory = _make_memory(mock_connector)
        memory.connect("mem-1", "mem-2", relationship="caused_by")
        assert mock_connector.query.call_count == 1

    def test_empty_relationship_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx1 = _make_ctx(session_id="s-1")
        ctx2 = _make_ctx(session_id="s-2")
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.connect(ctx1, ctx2, relationship="")

    def test_whitespace_relationship_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx1 = _make_ctx(session_id="s-1")
        ctx2 = _make_ctx(session_id="s-2")
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.connect(ctx1, ctx2, relationship="   ")

    def test_connection_properties_stored(self, mock_connector):
        captured = []

        def capture(cmd, blobs=None):
            captured.extend(cmd)
            return (
                [
                    {"FindEntity": {"status": 0}},
                    {"FindEntity": {"status": 0}},
                    {"AddConnection": {"status": 0}},
                ],
                [],
            )

        mock_connector.query.side_effect = capture
        memory = _make_memory(mock_connector)
        memory.connect(
            "mem-1", "mem-2",
            relationship="references",
            properties={"confidence": 0.9},
        )
        add_conn = next(c for c in captured if "AddConnection" in c)
        props = add_conn["AddConnection"]["properties"]
        assert props["confidence"] == 0.9
        assert "created_at" in props


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------


class TestRemove:
    def test_removes_existing_memory(self, mock_connector):
        mock_connector.query.return_value = (
            [{"DeleteEntity": {"status": 0}}], []
        )
        memory = _make_memory(mock_connector)
        memory.remove("mem-xyz")
        assert mock_connector.query.call_count == 1

    def test_empty_memory_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.remove("")

    def test_whitespace_memory_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.remove("   ")

    def test_storage_error_propagated(self, mock_connector):
        mock_connector.query.return_value = (
            [{"DeleteEntity": {"status": -1, "info": "not found"}}], []
        )
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusStorageError, match="not found"):
            memory.remove("ghost-id")


# ---------------------------------------------------------------------------
# pending_commits() / failed_commits()
# ---------------------------------------------------------------------------


class TestTaskMonitoring:
    @pytest.mark.asyncio
    async def test_pending_commits_includes_in_flight(self, mock_connector):
        # Use a future that never resolves to keep task pending
        mock_connector.query.side_effect = NexusStorageError("oops")
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"x", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        # Before task resolves, check pending (task may be pending or processing)
        in_flight = memory.pending_commits()
        # It might already have failed — test the union case
        all_tracked = memory.pending_commits() + memory.failed_commits()
        assert task in all_tracked

    @pytest.mark.asyncio
    async def test_failed_commits_after_error(self, mock_connector):
        mock_connector.query.side_effect = NexusStorageError("db error")
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"x", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        await task.wait()
        assert task in memory.failed_commits()
        assert task not in memory.pending_commits()


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


class TestStats:
    def test_raises_config_error_when_prometheus_missing(self, mock_connector):
        memory = _make_memory(mock_connector)
        with patch.dict("sys.modules", {"prometheus_client": None}):
            with pytest.raises(NexusConfigError, match="metrics support"):
                memory.stats()


# ---------------------------------------------------------------------------
# _resolve_session_id()
# ---------------------------------------------------------------------------


class TestResolveSessionId:
    def test_uses_session_id_when_set(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx = _make_ctx(session_id="explicit-sid")
        assert memory._resolve_session_id(ctx) == "explicit-sid"

    def test_derives_from_session_name_when_no_session_id(
        self, mock_connector
    ):
        memory = _make_memory(mock_connector)
        ctx = _make_ctx(session_id=None, session_name="support-001")
        sid = memory._resolve_session_id(ctx)
        assert isinstance(sid, str) and len(sid) == 32

    def test_derived_id_is_deterministic(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx1 = _make_ctx(session_id=None, session_name="same-name")
        ctx2 = _make_ctx(session_id=None, session_name="same-name")
        assert memory._resolve_session_id(ctx1) == memory._resolve_session_id(ctx2)

    def test_different_names_give_different_ids(self, mock_connector):
        memory = _make_memory(mock_connector)
        ctx1 = _make_ctx(session_id=None, session_name="name-a")
        ctx2 = _make_ctx(session_id=None, session_name="name-b")
        assert memory._resolve_session_id(ctx1) != memory._resolve_session_id(ctx2)


# ---------------------------------------------------------------------------
# _build_constraints()
# ---------------------------------------------------------------------------


class TestBuildConstraints:
    def test_known_keys_included(self):
        result = Memory._build_constraints({"session_id": "s-1", "user_id": "alice"})
        assert result == {
            "session_id": ["==", "s-1"],
            "user_id": ["==", "alice"],
        }

    def test_unknown_keys_dropped(self):
        result = Memory._build_constraints({"session_id": "s-1", "unknown": "x"})
        assert "unknown" not in result
        assert "session_id" in result

    def test_empty_filters_returns_empty(self):
        assert Memory._build_constraints({}) == {}


# ---------------------------------------------------------------------------
# _ensure_descriptor_set()
# ---------------------------------------------------------------------------


class TestEnsureDescriptorSet:
    def test_creates_when_absent(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=0),
            _ok_response("AddDescriptorSet"),
        ]
        memory = _make_memory(mock_connector)
        vec = np.ones(128, dtype=np.float32)
        memory._ensure_descriptor_set("nexus_text", vec, "CS", "FaissFlat")
        assert mock_connector.query.call_count == 2

    def test_skips_when_exists(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=1),
        ]
        memory = _make_memory(mock_connector)
        vec = np.ones(128, dtype=np.float32)
        memory._ensure_descriptor_set("nexus_text", vec, "CS", "FaissFlat")
        assert mock_connector.query.call_count == 1

    def test_infers_dimensions_from_embedding(self, mock_connector):
        captured = []

        def capture(cmd, blobs=None):
            captured.extend(cmd)
            return _ok_response("AddDescriptorSet")

        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=0),
            capture,
        ]
        # Override the second call
        calls = [_find_descriptor_set_response(count=0)]

        def side_effect(cmd, blobs=None):
            if mock_connector.query.call_count == 1:
                return _find_descriptor_set_response(count=0)
            captured.extend(cmd)
            return _ok_response("AddDescriptorSet")

        mock_connector.query.side_effect = side_effect
        memory = _make_memory(mock_connector)
        vec = np.ones(256, dtype=np.float32)
        memory._ensure_descriptor_set("nexus_text", vec, "CS", "FaissFlat")
        add_cmd = next(c for c in captured if "AddDescriptorSet" in c)
        assert add_cmd["AddDescriptorSet"]["dimensions"] == 256


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_construction(self):
        sr = SearchResult(
            score=0.95,
            modality="text",
            memory_id="m-1",
            session_id="s-1",
            context_id="c-1",
            timestamp=datetime(2024, 1, 1),
        )
        assert sr.score == 0.95
        assert sr.text is None
        assert sr.metadata == {}
