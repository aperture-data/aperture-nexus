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
        ctx = Context(principal=test_principal, session_id=_unique_sid())
        info = Information(context_id=ctx.id)
        info.log(text="first message")
        mid1 = memory_engine.commit(ctx, info)
        info.log(text="second message")
        mid2 = memory_engine.commit(ctx, info)
        assert mid1 != mid2

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

        # Search with the same vector — should be top hit
        results = memory_engine.search(query=vec, modality="text", k=5)
        assert len(results) >= 1
        memory_ids = [r.memory_id for r in results]
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
    def test_full_auth_pipeline(self, nexus_admin):
        """create_principal → authenticate → principal has correct fields."""
        uid = f"auth-test-{uuid.uuid4().hex[:8]}"
        api_key = nexus_admin.create_principal(
            user_id=uid, user_name="Auth Test User"
        )
        assert isinstance(api_key, str) and len(api_key) > 20
        principal = nexus_admin.authenticate(user_id=uid, api_key=api_key)
        assert principal.user_id == uid
        assert principal.user_name == "Auth Test User"
        # Cleanup
        nexus_admin.delete_principal(uid)

    def test_delete_principal_prevents_auth(self, nexus_admin):
        from aperture_nexus.exceptions import NexusPermissionError
        uid = f"del-test-{uuid.uuid4().hex[:8]}"
        api_key = nexus_admin.create_principal(user_id=uid)
        nexus_admin.delete_principal(uid)
        with pytest.raises(NexusPermissionError):
            nexus_admin.authenticate(user_id=uid, api_key=api_key)
