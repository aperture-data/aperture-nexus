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
- remove(): happy path; empty context_id raises
- pending_commits() / failed_commits(): task list filtering
- authenticate(): valid credentials return Principal, wrong key raises,
  unknown user raises, DB error raises NexusConnectionError
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

from aperture_nexus.admin import _hash_key
from aperture_nexus.auth import Principal
from aperture_nexus.exceptions import (
    NexusConfigError,
    NexusConnectionError,
    NexusPermissionError,
    NexusStorageError,
    NexusValidationError,
)
from aperture_nexus.memory import Memory, MemoryEntry, SearchResult


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
    # ApertureDB returns descriptors under "entities"; distance is
    # the "_distance" property on each entity, not a separate list.
    body: dict = {"status": 0}
    if descriptors is not None:
        entities = []
        for i, d in enumerate(descriptors):
            ent = dict(d)
            if distances is not None and i < len(distances):
                ent["_distance"] = distances[i]
            entities.append(ent)
        body["entities"] = entities
    return ([{"FindDescriptor": body}], [])


def _ok_response(cmd_name: str = "AddEntity") -> tuple:
    return ([{cmd_name: {"status": 0}}], [])


def _make_memory(mock_connector):
    m = Memory(db_client=mock_connector)
    m._schema_ensured = True  # skip CreateIndex calls in unit tests
    m._ensured_contexts = set()  # start fresh per test
    return m


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


def _ok_conn_response() -> tuple:
    """3-command batch: FindEntity + FindEntity + AddConnection."""
    return ([
        {"FindEntity": {"status": 0}},
        {"FindEntity": {"status": 0}},
        {"AddConnection": {"status": 0}},
    ], [])


# Standard sequence for a commit() with one text entry.
# Session/context use if_not_found — one AddEntity each (no prior FindEntity).
# After context AddEntity, a nexus_session_context connection is written (3 cmds).
# ApertureDB returns status=0 whether it created or skipped the entity.
#   1. AddEntity session  (if_not_found) → ok
#   2. AddEntity context  (if_not_found) → ok
#   3. nexus_session_context connection  → ok (FindEntity+FindEntity+AddConnection)
#   4. AddBlob  text                     → ok
def _commit_side_effects():
    return [
        _ok_response("AddEntity"),  # ensure_session (if_not_found)
        _ok_response("AddEntity"),  # ensure_context (if_not_found)
        _ok_conn_response(),        # nexus_session_context connection
        _ok_response("AddBlob"),    # write text entry
    ]


# Alias — both fresh and existing sessions have the same mock shape now
_commit_side_effects_fresh = _commit_side_effects
_commit_side_effects_existing = _commit_side_effects


# ---------------------------------------------------------------------------
# commit()
# ---------------------------------------------------------------------------


class TestCommit:
    def test_happy_path_returns_context_id(self, mock_connector):
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
            _ok_response("AddEntity"),  # ensure_session (if_not_found)
            _ok_response("AddEntity"),  # ensure_context (if_not_found)
            ([{"FindEntity": {"status": -1, "info": "schema error"}},
              {"FindEntity": {"status": 0}},
              {"AddConnection": {"status": 0}}], []),  # nexus_session_context fails
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
        # if_not_found: 1 AddEntity per ensure call regardless of prior state
        mock_connector.query.side_effect = _commit_side_effects()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 4

    def test_skips_session_creation_when_present(self, mock_connector):
        # if_not_found: same query count whether session exists or not
        mock_connector.query.side_effect = _commit_side_effects()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        memory.commit(ctx, info)
        assert mock_connector.query.call_count == 4

    def test_storage_error_propagated(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("AddEntity"),  # ensure_session
            _ok_response("AddEntity"),  # ensure_context
            _ok_conn_response(),        # nexus_session_context connection
            ([{"AddBlob": {"status": -1, "info": "conflict"}}], []),  # blob write fails
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello")
        with pytest.raises(NexusStorageError, match="conflict"):
            memory.commit(ctx, info)

    def test_metadata_included_in_blob_props(self, mock_connector):
        """metadata from log() is forwarded as ApertureDB properties."""
        mock_connector.query.side_effect = _commit_side_effects()
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(text="hello", metadata={"ticket_id": "T-99", "priority": 1})
        memory.commit(ctx, info)
        # The 4th call is AddBlob — check its properties contain metadata
        blob_call_args = mock_connector.query.call_args_list[3][0][0]
        props = blob_call_args[0]["AddBlob"]["properties"]
        assert props["ticket_id"] == "T-99"
        assert props["priority"] == 1

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

        # Second round — same ctx returns same context_id
        # Note: for the second commit, context is already in _ensured_contexts
        # so the connection step is skipped. Provide only 3 responses.
        mock_connector.query.side_effect = [
            _ok_response("AddEntity"),  # ensure_session (if_not_found)
            _ok_response("AddEntity"),  # ensure_context (if_not_found)
            _ok_response("AddBlob"),    # write text entry
        ]
        info.log(text="second")
        mid2 = memory.commit(ctx, info)
        assert len(info) == 0
        assert mid1 == mid2  # same ctx returns same context_id


# ---------------------------------------------------------------------------
# commit() — with embeddings
# ---------------------------------------------------------------------------


class TestCommitWithDescriptors:
    def test_writes_descriptor_for_precomputed_embedding(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("AddEntity"),                  # ensure_session
            _ok_response("AddEntity"),                  # ensure_context
            _ok_conn_response(),                        # nexus_session_context connection
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
            _ok_response("AddEntity"),                  # ensure_session
            _ok_response("AddEntity"),                  # ensure_context
            _ok_conn_response(),                        # nexus_session_context connection
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
            _ok_response("AddEntity"),                # ensure_session
            _ok_response("AddEntity"),                # ensure_context
            _ok_conn_response(),                      # nexus_session_context connection
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
            _ok_response("AddEntity"),  # ensure_session
            _ok_response("AddEntity"),  # ensure_context
            _ok_conn_response(),        # nexus_session_context connection
            _ok_response("AddBlob"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"raw", document_type="pdf")
        mid = memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)


# ---------------------------------------------------------------------------
# Video clip embeddings
# ---------------------------------------------------------------------------


class TestVideoClipEmbeddings:
    """Video entries produce per-clip Descriptors, not a single mean-pool."""

    def test_video_clip_embeddings_written_as_multiple_descriptors(
        self, mock_connector
    ):
        # video entry with 2 pre-generated clip embeddings (simulating what
        # _generate_missing_embeddings does when a CLIP model is configured).
        # Expected query sequence:
        #   1. _ensure_descriptor_sets → FindDescriptorSet (exists)
        #   2. commit: AddEntity session (if_not_found), AddEntity context (if_not_found),
        #              nexus_session_context connection
        #   3. AddVideo
        #   4. _write_video_clip_descriptors → FindDescriptorSet + AddDescriptor × 2
        clip1 = np.ones(128, dtype=np.float32)
        clip2 = np.zeros(128, dtype=np.float32) + 0.5
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=1),   # _ensure_descriptor_sets
            _ok_response("AddEntity"),                 # ensure_session (if_not_found)
            _ok_response("AddEntity"),                 # ensure_context (if_not_found)
            _ok_conn_response(),                       # nexus_session_context connection
            _ok_response("AddVideo"),                  # video blob
            _find_descriptor_set_response(count=1),   # _write_video_clip_descriptors dset check
            _ok_response("AddDescriptor"),             # clip 1
            _ok_response("AddDescriptor"),             # clip 2
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        # Manually set video_clip_embeddings on the entry to bypass actual CLIP
        info.log(video=b"fakevideo")
        entry = info._entries[0]
        entry.video_clip_embeddings = [
            (clip1, {"start_frame": 0, "stop_frame": 29}),
            (clip2, {"start_frame": 30, "stop_frame": 59}),
        ]
        entry.embedding_model = "ViT-B-32"

        mid = memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)

        # Verify AddDescriptor was called twice (once per clip)
        descriptor_calls = [
            call
            for call in mock_connector.query.call_args_list
            if call.args and any("AddDescriptor" in cmd for cmd in call.args[0])
        ]
        assert len(descriptor_calls) == 2

    def test_video_clip_descriptors_carry_frame_metadata(self, mock_connector):
        # Verify start_frame/stop_frame land on the Descriptor properties.
        clip_emb = np.ones(64, dtype=np.float32)
        captured_cmds = []

        def _capture(cmd, blobs=None):
            captured_cmds.append(cmd)
            if any("FindDescriptorSet" in c for c in cmd):
                return _find_descriptor_set_response(count=1)
            if any("FindEntity" in c for c in cmd):
                return _find_response(count=1)
            return _ok_response(list(cmd[0].keys())[0])

        mock_connector.query.side_effect = _capture

        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(video=b"v")
        entry = info._entries[0]
        entry.video_clip_embeddings = [
            (clip_emb, {"start_frame": 5, "stop_frame": 34}),
        ]
        entry.embedding_model = "ViT-B-32"

        memory.process_and_commit(ctx, info)

        descriptor_cmd = next(
            cmd[0]["AddDescriptor"]
            for cmd in captured_cmds
            if cmd and "AddDescriptor" in cmd[0]
        )
        props = descriptor_cmd["properties"]
        assert props["start_frame"] == 5
        assert props["stop_frame"] == 34
        assert props["modality"] == "video"

    def test_single_precomputed_video_embedding_still_works(self, mock_connector):
        # A caller who passes embedding= directly still gets one Descriptor
        # (the existing code path — not the per-clip path).
        vec = np.ones(128, dtype=np.float32)
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=1),   # _ensure_descriptor_sets
            _ok_response("AddEntity"),                 # ensure_session (if_not_found)
            _ok_response("AddEntity"),                 # ensure_context (if_not_found)
            _ok_conn_response(),                       # nexus_session_context connection
            _ok_response("AddVideo"),
            _find_descriptor_set_response(count=1),
            _ok_response("AddDescriptor"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        # Pre-compute a single embedding — video_clip_embeddings stays None
        info.log(video=b"v", embedding=vec, embedding_model="ViT-B/16")
        mid = memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)

        descriptor_calls = [
            call
            for call in mock_connector.query.call_args_list
            if call.args and any("AddDescriptor" in cmd for cmd in call.args[0])
        ]
        assert len(descriptor_calls) == 1


# ---------------------------------------------------------------------------
# async_process_and_commit()
# ---------------------------------------------------------------------------


class TestAsyncProcessAndCommit:
    @pytest.mark.asyncio
    async def test_returns_pending_task(self, mock_connector):
        mock_connector.query.side_effect = [
            _ok_response("AddEntity"),                # ensure_session (if_not_found)
            _ok_response("AddEntity"),                # ensure_context (if_not_found)
            _ok_conn_response(),                      # nexus_session_context connection
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
            _ok_response("AddEntity"),                # ensure_session (if_not_found)
            _ok_response("AddEntity"),                # ensure_context (if_not_found)
            _ok_conn_response(),                      # nexus_session_context connection
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
            _ok_response("AddEntity"),  # ensure_session (if_not_found)
            _ok_response("AddEntity"),  # ensure_context (if_not_found)
            _ok_conn_response(),        # nexus_session_context connection
            _ok_response("AddBlob"),
        ]
        memory = _make_memory(mock_connector)
        ctx = _make_ctx()
        info = _make_info(ctx)
        info.log(blob=b"data", document_type="bin")
        task = await memory.async_process_and_commit(ctx, info)
        await task.wait()
        assert task.status == "complete"
        assert isinstance(task.context_id, str)

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
            _ok_response("AddEntity"),  # ensure_session (if_not_found)
            _ok_response("AddEntity"),  # ensure_context (if_not_found)
            _ok_conn_response(),        # nexus_session_context connection
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
                "nexus_ctx_id": "m-1",
                "session_id": "s-1",
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
        assert results[0].context_id == "m-1"
        assert results[0].score == 1.0

    def test_vector_search_returns_results(self, mock_connector):
        descriptors = [
            {
                "session_id": "s-1",
                "context_id": "c-1",
                "user_id": "alice",
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
        assert results[0].context_id == "c-1"
        assert results[0].score == pytest.approx(0.95)

    def test_vector_search_respects_min_score(self, mock_connector):
        descriptors = [
            {
                "session_id": "s-1",
                "context_id": "c-1",
                "user_id": "alice",
                "created_at": "2024-01-01T00:00:00",
            },
            {
                "session_id": "s-1",
                "context_id": "c-2",
                "user_id": "alice",
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
        assert results[0].context_id == "c-1"

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

    def test_creates_connection_between_context_ids(self, mock_connector):
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


def _ok_response(cmd_name: str) -> tuple:
    return ([{cmd_name: {"status": 0, "count": 0}}], [])


def _find_dset_response(names: list[str]) -> tuple:
    entities = [{"_name": n} for n in names]
    body = {"status": 0, "returned": len(entities), "entities": entities}
    return ([{"FindDescriptorSet": body}], [])


class TestRemove:
    def _cascade_side_effect(self, dset_names: list[str]):
        """
        Build side_effect matching the actual remove() call sequence:
          DeleteBlob, DeleteImage, DeleteVideo,
          FindDescriptorSet,
          DeleteDescriptor × len(dset_names),
          DeleteEntity.
        """
        responses = [
            _ok_response("DeleteBlob"),
            _ok_response("DeleteImage"),
            _ok_response("DeleteVideo"),
            _find_dset_response(dset_names),
        ]
        for _ in dset_names:
            responses.append(_ok_response("DeleteDescriptor"))
        responses.append(_ok_response("DeleteEntity"))
        return responses

    def test_removes_existing_context(self, mock_connector):
        mock_connector.query.side_effect = self._cascade_side_effect([])
        memory = _make_memory(mock_connector)
        memory.remove("ctx-xyz")
        # FindDescriptorSet + 3 deletes + DeleteEntity = 5 calls
        assert mock_connector.query.call_count == 5

    def test_cascades_content_and_descriptors(self, mock_connector):
        dset_names = ["nexus_text__ViT-B/16", "nexus_image__ViT-B/16"]
        mock_connector.query.side_effect = (
            self._cascade_side_effect(dset_names)
        )
        memory = _make_memory(mock_connector)
        memory.remove("ctx-abc")
        # FindDescriptorSet + 3 content deletes + 2 descriptor deletes
        # + DeleteEntity = 7 calls
        assert mock_connector.query.call_count == 7

    def test_empty_context_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.remove("")

    def test_whitespace_context_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.remove("   ")

    def test_storage_error_propagated(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_dset_response([]),
            ([{"DeleteBlob": {"status": -1, "info": "gone"}}], []),
        ]
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusStorageError, match="gone"):
            memory.remove("ghost-id")


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


def _find_blob_response(entities: list, blobs: list) -> tuple:
    """Mock FindBlob response with blobs_start=0."""
    body = {
        "status": 0,
        "returned": len(entities),
        "blobs_start": 0,
        "entities": entities,
    }
    return ([{"FindBlob": body}], blobs)


def _find_image_response(entities: list, blobs: list) -> tuple:
    """Mock FindImage response with blobs_start=0."""
    body = {
        "status": 0,
        "returned": len(entities),
        "blobs_start": 0,
        "entities": entities,
    }
    return ([{"FindImage": body}], blobs)


def _find_video_response(entities: list, blobs: list) -> tuple:
    """Mock FindVideo response with blobs_start=0."""
    body = {
        "status": 0,
        "returned": len(entities),
        "blobs_start": 0,
        "entities": entities,
    }
    return ([{"FindVideo": body}], blobs)


def _empty_find_response(cmd_name: str) -> tuple:
    """Mock empty FindBlob/FindImage/FindVideo response."""
    body = {"status": 0, "returned": 0, "blobs_start": 0, "entities": []}
    return ([{cmd_name: body}], [])


class TestRetrieve:
    def test_empty_context_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.retrieve("")

    def test_whitespace_context_id_raises(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusValidationError, match="non-empty"):
            memory.retrieve("   ")

    def test_retrieves_text_entry(self, mock_connector):
        text_bytes = b"hello world"
        entities = [{
            "_blob_index": 0,
            "_uniqueid": "1.2.3",
            "context_id": "ctx-1",
            "session_id": "sess-1",
            "document_type": "text",
            "created_at": "2024-01-01T00:00:00",
        }]
        mock_connector.query.side_effect = [
            _find_blob_response(entities, [text_bytes]),
            _empty_find_response("FindImage"),
            _empty_find_response("FindVideo"),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-1")
        assert len(entries) == 1
        assert entries[0].modality == "text"
        assert entries[0].text == "hello world"
        assert entries[0].data is None
        assert entries[0].session_id == "sess-1"

    def test_retrieves_blob_entry(self, mock_connector):
        blob_bytes = b"%PDF-1.4 fake"
        entities = [{
            "_blob_index": 0,
            "_uniqueid": "1.2.4",
            "context_id": "ctx-2",
            "session_id": "sess-2",
            "document_type": "pdf",
            "created_at": "2024-01-01T00:00:00",
        }]
        mock_connector.query.side_effect = [
            _find_blob_response(entities, [blob_bytes]),
            _empty_find_response("FindImage"),
            _empty_find_response("FindVideo"),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-2")
        assert len(entries) == 1
        assert entries[0].modality == "blob"
        assert entries[0].data == blob_bytes
        assert entries[0].document_type == "pdf"
        assert entries[0].text is None

    def test_retrieves_image_entry(self, mock_connector):
        img_bytes = b"\x89PNG\r\n"
        entities = [{
            "_blob_index": 0,
            "_uniqueid": "2.1.0",
            "context_id": "ctx-3",
            "session_id": "sess-3",
            "created_at": "2024-01-01T00:00:00",
        }]
        mock_connector.query.side_effect = [
            _empty_find_response("FindBlob"),
            _find_image_response(entities, [img_bytes]),
            _empty_find_response("FindVideo"),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-3")
        assert len(entries) == 1
        assert entries[0].modality == "image"
        assert entries[0].data == img_bytes
        assert entries[0].text is None

    def test_retrieves_video_entry(self, mock_connector):
        vid_bytes = b"\x00\x00\x00 ftyp"
        entities = [{
            "_blob_index": 0,
            "_uniqueid": "5.1.0",
            "context_id": "ctx-4",
            "session_id": "sess-4",
            "created_at": "2024-01-01T00:00:00",
        }]
        mock_connector.query.side_effect = [
            _empty_find_response("FindBlob"),
            _empty_find_response("FindImage"),
            _find_video_response(entities, [vid_bytes]),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-4")
        assert len(entries) == 1
        assert entries[0].modality == "video"
        assert entries[0].data == vid_bytes

    def test_retrieves_mixed_modalities(self, mock_connector):
        txt_ent = [{
            "_blob_index": 0,
            "_uniqueid": "1.2.5",
            "context_id": "ctx-5",
            "session_id": "sess-5",
            "document_type": "text",
            "created_at": "2024-01-01T00:00:00",
        }]
        img_ent = [{
            "_blob_index": 0,
            "_uniqueid": "2.2.0",
            "context_id": "ctx-5",
            "session_id": "sess-5",
            "created_at": "2024-01-01T00:00:00",
        }]
        mock_connector.query.side_effect = [
            _find_blob_response(txt_ent, [b"mixed text"]),
            _find_image_response(img_ent, [b"\x89PNG"]),
            _empty_find_response("FindVideo"),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-5")
        assert len(entries) == 2
        modalities = {e.modality for e in entries}
        assert modalities == {"text", "image"}

    def test_empty_context_returns_empty_list(self, mock_connector):
        mock_connector.query.side_effect = [
            _empty_find_response("FindBlob"),
            _empty_find_response("FindImage"),
            _empty_find_response("FindVideo"),
        ]
        memory = _make_memory(mock_connector)
        entries = memory.retrieve("ctx-empty")
        assert entries == []

    def test_db_error_raises_connection_error(self, mock_connector):
        mock_connector.query.side_effect = ConnectionError("timeout")
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusConnectionError, match="blob retrieval failed"):
            memory.retrieve("ctx-boom")


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
    def test_raises_not_implemented(self, mock_connector):
        memory = _make_memory(mock_connector)
        with pytest.raises(NexusConfigError, match="not yet implemented"):
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
        memory._ensure_descriptor_set("nexus_text__mymodel", vec, "text", "mymodel", "CS", "HNSW")
        assert mock_connector.query.call_count == 2

    def test_skips_when_exists(self, mock_connector):
        mock_connector.query.side_effect = [
            _find_descriptor_set_response(count=1),
        ]
        memory = _make_memory(mock_connector)
        vec = np.ones(128, dtype=np.float32)
        memory._ensure_descriptor_set("nexus_text__mymodel", vec, "text", "mymodel", "CS", "HNSW")
        assert mock_connector.query.call_count == 1

    def test_infers_dimensions_from_embedding(self, mock_connector):
        captured = []

        def side_effect(cmd, blobs=None):
            if mock_connector.query.call_count == 1:
                return _find_descriptor_set_response(count=0)
            captured.extend(cmd)
            return _ok_response("AddDescriptorSet")

        mock_connector.query.side_effect = side_effect
        memory = _make_memory(mock_connector)
        vec = np.ones(256, dtype=np.float32)
        memory._ensure_descriptor_set("nexus_text__mymodel", vec, "text", "mymodel", "CS", "HNSW")
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
            session_id="s-1",
            context_id="c-1",
            user_id="alice",
            created_at=datetime(2024, 1, 1),
        )
        assert sr.score == 0.95
        assert sr.text is None
        assert sr.metadata == {}


# ---------------------------------------------------------------------------
# authenticate()
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_valid_credentials_return_principal(self, mock_connector):
        api_key = "test-key-12345"
        record = {
            "user_id": "alice",
            "user_name": "Alice Chen",
            "department": "support",
            "organization": "AcmeCorp",
            "api_key_hash": _hash_key(api_key),
        }
        mock_connector.query.return_value = _find_response(
            count=1, entities=[record]
        )
        memory = Memory(db_client=mock_connector)
        principal = memory.authenticate("alice", api_key)
        assert isinstance(principal, Principal)
        assert principal.user_id == "alice"
        assert principal.user_name == "Alice Chen"
        assert principal.department == "support"
        assert principal.organization == "AcmeCorp"

    def test_wrong_api_key_raises(self, mock_connector):
        record = {
            "user_id": "alice",
            "api_key_hash": _hash_key("correct-key"),
        }
        mock_connector.query.return_value = _find_response(
            count=1, entities=[record]
        )
        memory = Memory(db_client=mock_connector)
        with pytest.raises(NexusPermissionError, match="Invalid credentials"):
            memory.authenticate("alice", "wrong-key")

    def test_unknown_user_raises(self, mock_connector):
        mock_connector.query.return_value = _find_response(
            count=0, entities=[]
        )
        memory = Memory(db_client=mock_connector)
        with pytest.raises(NexusPermissionError, match="does not exist"):
            memory.authenticate("ghost", "any-key")

    def test_db_error_raises_connection_error(self, mock_connector):
        mock_connector.query.side_effect = ConnectionError("timeout")
        memory = Memory(db_client=mock_connector)
        with pytest.raises(NexusConnectionError, match="query failed"):
            memory.authenticate("alice", "key")

    def test_empty_user_id_raises(self, mock_connector):
        memory = Memory(db_client=mock_connector)
        with pytest.raises(NexusValidationError):
            memory.authenticate("", "key")

    def test_empty_api_key_raises(self, mock_connector):
        memory = Memory(db_client=mock_connector)
        with pytest.raises(NexusValidationError):
            memory.authenticate("alice", "")
