"""
Integration tests for Memory against a real ApertureDB instance.

Spins up ApertureDB community edition via Docker (see conftest.py).
Each test is isolated by using unique session_ids and cleans up
the memories it creates.

Run these tests with:
    pytest -m integration
    make test-integration
"""

import uuid
import numpy as np
import pytest

from aperture_nexus.context import Context
from aperture_nexus.information import Information
from aperture_nexus.exceptions import NexusValidationError, NexusStorageError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_sid():
    return f"integ-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Basic commit and retrieve
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCommitAndRetrieve:
    def test_commit_text_returns_memory_id(self, memory_engine, test_principal):
        ctx = Context(
            principal=test_principal,
            session_id=_unique_sid(),
            purpose="integration test",
        )
        info = Information(context_id=ctx.id)
        info.log(text="ApertureDB is a multimodal vector database.")
        mid = memory_engine.commit(ctx, info)
        assert isinstance(mid, str) and len(mid) > 10

    def test_commit_drains_info(self, memory_engine, test_principal):
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="drain test")
        assert len(info) == 1
        memory_engine.commit(ctx, info)
        assert len(info) == 0

    def test_commit_image_bytes(self, memory_engine, test_principal):
        import io
        from PIL import Image as PILImage
        # Create a tiny red 4x4 PNG
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(image=image_bytes)
        mid = memory_engine.commit(ctx, info)
        assert isinstance(mid, str)

    def test_multiple_commits_same_context(self, memory_engine, test_principal):
        """Multiple commits to the same context all return the same context_id."""
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="first message")
        mid1 = memory_engine.commit(ctx, info)
        info.log(text="second message")
        mid2 = memory_engine.commit(ctx, info)
        # commit() returns context_id — same context, same ID both times
        assert mid1 == mid2 == ctx.id

    def test_empty_info_raises(self, memory_engine, test_principal):
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        with pytest.raises(NexusValidationError, match="no entries"):
            memory_engine.commit(ctx, info)


# ---------------------------------------------------------------------------
# Metadata search
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMetadataSearch:
    def test_search_by_session_id(self, memory_engine, test_principal):
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(text="searchable memory")
        memory_engine.commit(ctx, info)

        results = memory_engine.search(filters={"session_id": sid})
        assert len(results) >= 1
        assert all(r.session_id == sid for r in results)

    def test_search_unknown_session_returns_empty(self, memory_engine, test_principal):
        results = memory_engine.search(filters={"session_id": "no-such-session-xyz"})
        assert results == []


# ---------------------------------------------------------------------------
# Vector search with pre-computed embeddings
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVectorSearch:
    def test_search_with_precomputed_embedding(self, memory_engine, test_principal):
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        vec = np.random.rand(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        info.log(text="vector search test", embedding=vec, embedding_model="test-model")
        memory_engine.commit(ctx, info)

        # Search with the same vector — must specify the same embedding_model
        # used at commit time so search looks in the right DescriptorSet.
        results = memory_engine.search(
            query=vec, modality="text", k=5, embedding_model="test-model"
        )
        assert len(results) >= 1
        # The memory we just committed should appear
        assert any(r.session_id == sid for r in results)

    def test_vector_search_without_modality_raises(self, memory_engine):
        vec = np.random.rand(512).astype(np.float32)
        with pytest.raises(NexusValidationError, match="modality is required"):
            memory_engine.search(query=vec)


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConnect:
    def test_connect_two_contexts(self, memory_engine, test_principal):
        ctx1 = Context(principal=test_principal, session_id=_unique_sid())
        ctx2 = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx1.id)
        info.log(text="source context")
        memory_engine.commit(ctx1, info)
        info.log(text="target context")
        memory_engine.commit(ctx2, info)
        # Should not raise
        memory_engine.connect(ctx1, ctx2, relationship="follows")


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRemove:
    def test_remove_committed_memory(self, memory_engine, test_principal):
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="to be removed")
        mid = memory_engine.commit(ctx, info)
        # Remove — should not raise
        memory_engine.remove(mid)

    def test_remove_empty_id_raises(self, memory_engine):
        with pytest.raises(NexusValidationError):
            memory_engine.remove("")


# ---------------------------------------------------------------------------
# NexusAdmin pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAdminPipeline:
    def test_full_auth_pipeline(self, nexus_admin, memory_engine):
        """create_principal → Memory.authenticate() → principal has correct fields."""
        uid = f"auth-test-{uuid.uuid4().hex[:8]}"
        api_key = nexus_admin.create_principal(
            user_id=uid, user_name="Auth Test User"
        )
        assert isinstance(api_key, str) and len(api_key) > 20
        principal = memory_engine.authenticate(user_id=uid, api_key=api_key)
        assert principal.user_id == uid
        assert principal.user_name == "Auth Test User"
        # Cleanup
        nexus_admin.delete_principal(uid)

    def test_rotate_key_invalidates_old_key(self, nexus_admin, memory_engine):
        from aperture_nexus.exceptions import NexusPermissionError
        uid = f"rotate-test-{uuid.uuid4().hex[:8]}"
        old_key = nexus_admin.create_principal(user_id=uid)
        new_key = nexus_admin.rotate_key(user_id=uid)
        # old key no longer works
        with pytest.raises(NexusPermissionError):
            memory_engine.authenticate(user_id=uid, api_key=old_key)
        # new key works
        principal = memory_engine.authenticate(user_id=uid, api_key=new_key)
        assert principal.user_id == uid
        nexus_admin.delete_principal(uid)

    def test_delete_principal_prevents_auth(self, nexus_admin, memory_engine):
        from aperture_nexus.exceptions import NexusPermissionError
        uid = f"del-test-{uuid.uuid4().hex[:8]}"
        api_key = nexus_admin.create_principal(user_id=uid)
        nexus_admin.delete_principal(uid)
        with pytest.raises(NexusPermissionError):
            memory_engine.authenticate(user_id=uid, api_key=api_key)

# ---------------------------------------------------------------------------
# CLIP process_and_commit — text search, image commit, metadata search
# ---------------------------------------------------------------------------

_CLIP_MODEL = "ViT-B-32"


@pytest.fixture()
def clip_memory(adb_config_name):
    """Memory instance configured with a CLIP model for all three modalities."""
    from aperture_nexus.memory import Memory
    m = Memory()
    m._cfg.models.text_embedding = _CLIP_MODEL
    m._cfg.models.image_embedding = _CLIP_MODEL
    m._cfg.models.video_embedding = _CLIP_MODEL
    return m


@pytest.mark.integration
class TestClipTextSearch:
    """process_and_commit with text entries — CLIP text embeddings."""

    def test_process_and_commit_text_returns_context_id(
        self, clip_memory, test_principal
    ):
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="ApertureDB is a multimodal vector database.")
        mid = clip_memory.process_and_commit(ctx, info)
        assert isinstance(mid, str) and len(mid) > 10

    def test_text_search_finds_committed_entry(self, clip_memory, test_principal):
        """A text committed via CLIP is retrievable by vector search."""
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(text="The quick brown fox jumps over the lazy dog.")
        clip_memory.process_and_commit(ctx, info)

        results = clip_memory.search(
            query="quick fox jumping", modality="text",
            embedding_model=_CLIP_MODEL, k=5,
        )
        assert any(r.session_id == sid for r in results), (
            f"Committed session {sid!r} not found in search results: "
            + str([r.session_id for r in results])
        )

    def test_text_search_result_has_inline_text(self, clip_memory, test_principal):
        """Short text is stored inline on the Descriptor and returned in results."""
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(text="inline text entry for CLIP")
        clip_memory.process_and_commit(ctx, info)

        results = clip_memory.search(
            query="inline text", modality="text",
            embedding_model=_CLIP_MODEL, k=5,
        )
        matching = [r for r in results if r.session_id == sid]
        assert matching, "Committed entry not found"
        assert matching[0].text == "inline text entry for CLIP"


@pytest.mark.integration
class TestClipImageSearch:
    """process_and_commit with image entries — CLIP image embeddings."""

    @staticmethod
    def _make_png(color: tuple = (255, 0, 0)) -> bytes:
        import io
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (16, 16), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_image_commit_succeeds(self, clip_memory, test_principal):
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(image=self._make_png())
        mid = clip_memory.process_and_commit(ctx, info)
        assert isinstance(mid, str)

    def test_image_search_finds_committed_image(self, clip_memory, test_principal):
        """An image committed via CLIP is retrievable by image vector search."""
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        img_bytes = self._make_png(color=(0, 128, 255))
        info.log(image=img_bytes)
        clip_memory.process_and_commit(ctx, info)

        # Search with the same image bytes — should find itself
        from aperture_nexus._embeddings import get_clip_embedder
        embedder = get_clip_embedder(_CLIP_MODEL)
        query_vec = embedder.embed_image(img_bytes)

        results = clip_memory.search(
            query=query_vec, modality="image",
            embedding_model=_CLIP_MODEL, k=5,
        )
        assert any(r.session_id == sid for r in results), (
            f"Committed image session {sid!r} not in results: "
            + str([r.session_id for r in results])
        )

    def test_image_result_has_image_modality(self, clip_memory, test_principal):
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(image=self._make_png())
        clip_memory.process_and_commit(ctx, info)

        from aperture_nexus._embeddings import get_clip_embedder
        embedder = get_clip_embedder(_CLIP_MODEL)
        query_vec = embedder.embed_image(self._make_png())

        results = clip_memory.search(
            query=query_vec, modality="image",
            embedding_model=_CLIP_MODEL, k=5,
        )
        matching = [r for r in results if r.session_id == sid]
        assert matching
        assert matching[0].modality == "image"


@pytest.mark.integration
class TestMetadataSearchContentEntities:
    """_search_by_metadata returns Blob/Image/Video entities, not NexusContext."""

    def test_metadata_search_finds_text_blob(self, clip_memory, test_principal):
        """A committed text entry is found by session_id metadata filter."""
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(text="metadata search test entry")
        clip_memory.process_and_commit(ctx, info)

        results = clip_memory.search(filters={"session_id": sid})
        assert len(results) >= 1
        assert all(r.session_id == sid for r in results)

    def test_metadata_search_finds_image(self, clip_memory, test_principal):
        """A committed image entry is found by session_id metadata filter."""
        import io
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (8, 8), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(image=buf.getvalue())
        clip_memory.process_and_commit(ctx, info)

        results = clip_memory.search(filters={"session_id": sid})
        assert len(results) >= 1
        matching = [r for r in results if r.session_id == sid]
        assert any(r.modality == "image" for r in matching)

    def test_metadata_search_unknown_session_returns_empty(self, clip_memory):
        results = clip_memory.search(filters={"session_id": "no-such-session-xyz"})
        assert results == []


# ---------------------------------------------------------------------------
# Information.remove_all() and Information.remove() — buffer cleanup before commit
#
# These tests verify that cleanup on the local buffer (before commit) is
# invisible to ApertureDB — only entries still in the buffer at commit()
# time are stored. This is the expected pattern for abandoning stale
# content, e.g. after an upstream error, or for removing a specific entry
# before it reaches the database.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInformationRemoveAllBeforeCommit:
    def test_remove_all_then_log_only_fresh_entry_committed(
        self, memory_engine, test_principal
    ):
        """Entries logged before remove_all() are not committed; only
        post-remove_all entries reach ApertureDB."""
        sid = _unique_sid()
        ctx = Context(
            principal=test_principal,
            session_id=sid,
            purpose="remove_all before commit test",
        )
        info = Information(context_id=ctx.id)

        # Log something, decide to start over, then log the real entry
        info.log(text="preliminary draft — should not be committed")
        info.remove_all()
        info.log(text="final entry — this one should be committed")
        memory_engine.commit(ctx, info)

        results = memory_engine.search(filters={"session_id": sid})
        assert len(results) == 1
        assert results[0].text == "final entry — this one should be committed"

    def test_remove_all_entries_then_commit_raises(
        self, memory_engine, test_principal
    ):
        """Committing an empty buffer raises NexusValidationError — there is
        nothing to store."""
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="will be removed")
        info.remove_all()
        with pytest.raises(NexusValidationError, match="no entries"):
            memory_engine.commit(ctx, info)


@pytest.mark.integration
class TestInformationRemoveBeforeCommit:
    def test_remove_first_entry_only_second_committed(
        self, memory_engine, test_principal
    ):
        """Removing the first entry (by reference) before commit stores only the second."""
        sid = _unique_sid()
        ctx = Context(
            principal=test_principal,
            session_id=sid,
            purpose="remove before commit test",
        )
        info = Information(context_id=ctx.id)
        entry_a = info.log(text="entry A — remove this")
        info.log(text="entry B — keep this")
        info.remove(entry_a)
        memory_engine.commit(ctx, info)

        results = memory_engine.search(filters={"session_id": sid})
        assert len(results) == 1
        assert results[0].text == "entry B — keep this"

    def test_remove_last_entry_only_first_committed(
        self, memory_engine, test_principal
    ):
        """Removing the last entry (by reference) before commit stores only the first."""
        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        info.log(text="keep this")
        discard = info.log(text="discard this")
        info.remove(discard)
        memory_engine.commit(ctx, info)

        results = memory_engine.search(filters={"session_id": sid})
        assert len(results) == 1
        assert results[0].text == "keep this"


@pytest.mark.integration
class TestBlobFilePath:
    def test_blob_from_file_path_stores_bytes(
        self, memory_engine, test_principal, tmp_path
    ):
        """Passing blob as a file path — content is read and stored at
        commit() time, not at log() time."""
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        sid = _unique_sid()
        ctx = Context(principal=test_principal, session_id=sid)
        info = Information(context_id=ctx.id)
        # Pass the path as a string — no open() needed
        info.log(blob=str(pdf_path), document_type="pdf")
        memory_engine.commit(ctx, info)

        # Should be committed without error; searchable by session
        results = memory_engine.search(filters={"session_id": sid})
        assert len(results) >= 1
        blob_result = next(
            (r for r in results if r.modality == "blob"), None
        )
        assert blob_result is not None
