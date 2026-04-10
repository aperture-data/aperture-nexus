"""
Unit tests for aperture_nexus.information.

Tests cover:
- Construction (valid, invalid context_id)
- log(): happy paths for all modalities (text, image, video, blob)
- log(): returns InformationEntry; tag and logged_at fields set
- log(): image input variants (path, URL, bytes, PIL, numpy)
- log(): blob input variants (bytes, file path, URL)
- log(): validation errors (missing file, permission denied, wrong dtype,
  missing document_type, empty text, no modality, etc.)
- log(): embedding with and without embedding_model
- log(): combined modalities in one call
- remove_all(): empties buffer without committing
- remove(entry): removes a specific entry by identity
- remove_tagged(tag): removes all entries with the given tag
- remove_before(timestamp): removes entries logged before a timestamp
- __len__ and __repr__

No live ApertureDB instance required.
"""

import stat
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image as PILImage

from aperture_nexus.exceptions import NexusValidationError
from aperture_nexus.information import Information, InformationEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_array(h: int = 4, w: int = 4) -> np.ndarray:
    """Return a valid HWC uint8 RGB numpy array."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _pil_image() -> PILImage.Image:
    return PILImage.new("RGB", (8, 8), color=(128, 0, 0))


def _embedding(dim: int = 512) -> np.ndarray:
    return np.ones(dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInformationConstruction:
    def test_valid_context_id(self):
        info = Information(context_id="ctx-abc")
        assert info.context_id == "ctx-abc"
        assert len(info) == 0

    def test_empty_context_id_raises(self):
        with pytest.raises(NexusValidationError, match="context_id"):
            Information(context_id="")

    def test_whitespace_context_id_raises(self):
        with pytest.raises(NexusValidationError, match="context_id"):
            Information(context_id="   ")

    def test_non_string_context_id_raises(self):
        with pytest.raises(NexusValidationError, match="context_id"):
            Information(context_id=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# log() — text
# ---------------------------------------------------------------------------


class TestLogText:
    def test_plain_text(self):
        info = Information(context_id="c")
        entry = info.log(text="Hello world")
        assert isinstance(entry, InformationEntry)
        assert len(info) == 1
        assert info._entries[0].text == "Hello world"

    def test_log_returns_entry_with_logged_at(self):
        before = datetime.now(timezone.utc)
        info = Information(context_id="c")
        entry = info.log(text="timestamped")
        after = datetime.now(timezone.utc)
        assert entry.logged_at.tzinfo is not None
        assert before <= entry.logged_at <= after

    def test_log_with_tag(self):
        info = Information(context_id="c")
        entry = info.log(text="tagged entry", tag="order-1")
        assert entry.tag == "order-1"
        assert info._entries[0].tag == "order-1"

    def test_log_tag_default_is_none(self):
        info = Information(context_id="c")
        entry = info.log(text="no tag")
        assert entry.tag is None

    def test_log_empty_tag_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="empty"):
            info.log(text="x", tag="")

    def test_empty_text_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="empty"):
            info.log(text="")

    def test_whitespace_text_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="empty"):
            info.log(text="   ")

    def test_non_string_text_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="string"):
            info.log(text=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# log() — image variants
# ---------------------------------------------------------------------------


class TestLogImage:
    def test_image_url_accepted(self):
        info = Information(context_id="c")
        info.log(image="https://example.com/photo.jpg")
        assert info._entries[0].image == "https://example.com/photo.jpg"

    def test_image_http_url_accepted(self):
        info = Information(context_id="c")
        info.log(image="http://example.com/photo.jpg")
        assert info._entries[0].image == "http://example.com/photo.jpg"

    def test_image_file_path(self, tmp_path):
        img_path = tmp_path / "test.png"
        _pil_image().save(img_path)
        info = Information(context_id="c")
        info.log(image=str(img_path))
        assert info._entries[0].image == str(img_path)

    def test_image_missing_file_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="not found"):
            info.log(image="/nonexistent/path/image.jpg")

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not reliable on Windows")
    def test_image_permission_denied_raises(self, tmp_path):
        img_path = tmp_path / "noperm.png"
        _pil_image().save(img_path)
        img_path.chmod(0o000)
        try:
            info = Information(context_id="c")
            with pytest.raises(NexusValidationError, match="permission denied"):
                info.log(image=str(img_path))
        finally:
            img_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_image_pil(self):
        info = Information(context_id="c")
        img = _pil_image()
        info.log(image=img)
        assert isinstance(info._entries[0].image, PILImage.Image)

    def test_image_bytes(self):
        info = Information(context_id="c")
        raw = b"\x89PNG\r\n\x1a\n"
        info.log(image=raw)
        assert info._entries[0].image == raw

    def test_image_numpy_hwc_uint8(self):
        info = Information(context_id="c")
        arr = _rgb_array()
        info.log(image=arr)
        assert info._entries[0].image is arr

    def test_image_numpy_hwc_float32(self):
        info = Information(context_id="c")
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        info.log(image=arr)
        assert info._entries[0].image is arr

    def test_image_numpy_hw_grayscale(self):
        info = Information(context_id="c")
        arr = np.zeros((8, 8), dtype=np.uint8)
        info.log(image=arr)
        assert info._entries[0].image is arr

    def test_image_numpy_hwc_single_channel(self):
        info = Information(context_id="c")
        arr = np.zeros((4, 4, 1), dtype=np.uint8)
        info.log(image=arr)
        assert info._entries[0].image is arr

    def test_image_numpy_wrong_dtype_raises(self):
        info = Information(context_id="c")
        arr = np.zeros((4, 4, 3), dtype=np.int32)
        with pytest.raises(NexusValidationError, match="dtype"):
            info.log(image=arr)

    def test_image_numpy_wrong_shape_raises(self):
        info = Information(context_id="c")
        arr = np.zeros((4,), dtype=np.uint8)  # 1D — not an image
        with pytest.raises(NexusValidationError, match="shape"):
            info.log(image=arr)

    def test_image_numpy_bad_channel_count_raises(self):
        info = Information(context_id="c")
        arr = np.zeros((4, 4, 5), dtype=np.uint8)  # 5 channels
        with pytest.raises(NexusValidationError, match="channel"):
            info.log(image=arr)

    def test_image_unsupported_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="Unsupported"):
            info.log(image={"not": "an image"})  # type: ignore


# ---------------------------------------------------------------------------
# log() — video
# ---------------------------------------------------------------------------


class TestLogVideo:
    def test_video_url(self):
        info = Information(context_id="c")
        info.log(video="https://example.com/clip.mp4")
        assert info._entries[0].video == "https://example.com/clip.mp4"

    def test_video_file_path(self, tmp_path):
        vid_path = tmp_path / "test.mp4"
        vid_path.write_bytes(b"fake-mp4-content")
        info = Information(context_id="c")
        info.log(video=str(vid_path))
        assert info._entries[0].video == str(vid_path)

    def test_video_missing_file_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="not found"):
            info.log(video="/no/such/video.mp4")

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not reliable on Windows")
    def test_video_permission_denied_raises(self, tmp_path):
        vid_path = tmp_path / "noperm.mp4"
        vid_path.write_bytes(b"fake-mp4")
        vid_path.chmod(0o000)
        try:
            info = Information(context_id="c")
            with pytest.raises(NexusValidationError, match="permission denied"):
                info.log(video=str(vid_path))
        finally:
            vid_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_video_bytes(self):
        info = Information(context_id="c")
        raw = b"fake-video-bytes"
        info.log(video=raw)
        assert info._entries[0].video == raw

    def test_video_unsupported_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="Unsupported"):
            info.log(video=12345)  # type: ignore


# ---------------------------------------------------------------------------
# log() — blob
# ---------------------------------------------------------------------------


class TestLogBlob:
    def test_blob_with_document_type(self):
        info = Information(context_id="c")
        info.log(blob=b"%PDF-1.4 content", document_type="pdf")
        assert info._entries[0].blob == b"%PDF-1.4 content"
        assert info._entries[0].document_type == "pdf"

    def test_document_type_lowercased(self):
        info = Information(context_id="c")
        info.log(blob=b"data", document_type="PDF")
        assert info._entries[0].document_type == "pdf"

    def test_blob_without_document_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="document_type"):
            info.log(blob=b"data")

    def test_document_type_without_blob_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="blob is None"):
            info.log(text="hello", document_type="pdf")

    def test_blob_file_path(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        info = Information(context_id="c")
        info.log(blob=str(pdf_path), document_type="pdf")
        assert info._entries[0].blob == str(pdf_path)
        assert info._entries[0].document_type == "pdf"

    def test_blob_url(self):
        info = Information(context_id="c")
        info.log(
            blob="https://example.com/report.pdf",
            document_type="pdf",
        )
        assert info._entries[0].blob == "https://example.com/report.pdf"

    def test_blob_http_url(self):
        info = Information(context_id="c")
        info.log(
            blob="http://example.com/audio.mp3",
            document_type="mp3",
        )
        assert info._entries[0].blob == "http://example.com/audio.mp3"

    def test_blob_missing_file_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="not found"):
            info.log(blob="/nonexistent/file.pdf", document_type="pdf")

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not reliable on Windows")
    def test_blob_permission_denied_raises(self, tmp_path):
        pdf_path = tmp_path / "noperm.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        pdf_path.chmod(0o000)
        try:
            info = Information(context_id="c")
            with pytest.raises(NexusValidationError, match="permission denied"):
                info.log(blob=str(pdf_path), document_type="pdf")
        finally:
            pdf_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_blob_unsupported_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="file path, URL, or bytes"):
            info.log(blob=12345, document_type="txt")  # type: ignore

    def test_empty_document_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="document_type"):
            info.log(blob=b"data", document_type="")


# ---------------------------------------------------------------------------
# log() — embedding
# ---------------------------------------------------------------------------


class TestLogEmbedding:
    def test_embedding_with_model(self):
        info = Information(context_id="c")
        emb = _embedding()
        info.log(image=_rgb_array(), embedding=emb,
                 embedding_model="clip-vit-base-patch32")
        assert info._entries[0].embedding is emb
        assert info._entries[0].embedding_model == "clip-vit-base-patch32"

    def test_embedding_without_model_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="embedding_model"):
            info.log(image=_rgb_array(), embedding=_embedding())

    def test_embedding_non_array_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="numpy array"):
            info.log(
                image=_rgb_array(),
                embedding=[0.1, 0.2, 0.3],  # type: ignore
                embedding_model="model",
            )

    def test_embedding_2d_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="1D"):
            info.log(
                image=_rgb_array(),
                embedding=np.ones((4, 128), dtype=np.float32),
                embedding_model="model",
            )

    def test_embedding_integer_dtype_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="float"):
            info.log(
                image=_rgb_array(),
                embedding=np.ones(512, dtype=np.int32),
                embedding_model="model",
            )

    def test_embedding_float64_accepted(self):
        info = Information(context_id="c")
        emb = np.ones(256, dtype=np.float64)
        info.log(image=_rgb_array(), embedding=emb,
                 embedding_model="my-model")
        assert info._entries[0].embedding is emb

    def test_embedding_model_slash_accepted(self):
        """Standard OpenCLIP names like ViT-B/16 must pass through unchanged."""
        info = Information(context_id="c")
        emb = _embedding()
        info.log(image=_rgb_array(), embedding=emb, embedding_model="ViT-B/16")
        assert info._entries[0].embedding_model == "ViT-B/16"

    def test_embedding_model_double_quote_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="not valid"):
            info.log(image=_rgb_array(), embedding=_embedding(),
                     embedding_model='bad"name')

    def test_embedding_model_backslash_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="not valid"):
            info.log(image=_rgb_array(), embedding=_embedding(),
                     embedding_model="bad\\name")

    def test_embedding_model_control_char_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="control character"):
            info.log(image=_rgb_array(), embedding=_embedding(),
                     embedding_model="bad\x00name")


# ---------------------------------------------------------------------------
# log() — combined modalities and general
# ---------------------------------------------------------------------------


class TestLogCombined:
    def test_no_modality_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="At least one"):
            info.log()

    def test_text_and_blob_combined(self):
        info = Information(context_id="c")
        info.log(text="See attached", blob=b"pdf-data", document_type="pdf")
        entry = info._entries[0]
        assert entry.text == "See attached"
        assert entry.blob == b"pdf-data"
        assert entry.document_type == "pdf"

    def test_image_and_text_combined(self):
        info = Information(context_id="c")
        info.log(text="Caption here", image=_rgb_array())
        entry = info._entries[0]
        assert entry.text == "Caption here"
        assert entry.image is not None

    def test_multiple_log_calls_accumulate(self):
        info = Information(context_id="c")
        info.log(text="first")
        info.log(text="second")
        info.log(image=_rgb_array())
        assert len(info) == 3


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class TestLogMetadata:
    def test_metadata_stored_on_entry(self):
        info = Information(context_id="c")
        info.log(text="hello", metadata={"ticket_id": "T-99", "priority": 1})
        assert info._entries[0].metadata == {"ticket_id": "T-99", "priority": 1}

    def test_metadata_none_by_default(self):
        info = Information(context_id="c")
        info.log(text="hello")
        assert info._entries[0].metadata is None

    def test_metadata_all_value_types(self):
        info = Information(context_id="c")
        info.log(text="x", metadata={
            "s": "string", "i": 42, "f": 3.14, "b": True
        })
        assert info._entries[0].metadata["s"] == "string"
        assert info._entries[0].metadata["i"] == 42
        assert info._entries[0].metadata["f"] == 3.14
        assert info._entries[0].metadata["b"] is True

    def test_metadata_not_a_dict_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="metadata must be a dict"):
            info.log(text="x", metadata=["not", "a", "dict"])

    def test_metadata_non_string_key_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="keys must be strings"):
            info.log(text="x", metadata={1: "value"})

    def test_metadata_reserved_key_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="reserved"):
            info.log(text="x", metadata={"context_id": "override"})

    def test_metadata_all_reserved_keys_rejected(self):
        reserved = [
            "context_id", "session_id", "user_id", "created_at",
            "document_type", "text_preview", "text",
            "embedding_model", "modality", "start_frame", "stop_frame",
        ]
        info = Information(context_id="c")
        for key in reserved:
            with pytest.raises(NexusValidationError, match="reserved"):
                info.log(text="x", metadata={key: "v"})

    def test_metadata_unsupported_value_type_raises(self):
        info = Information(context_id="c")
        with pytest.raises(NexusValidationError, match="unsupported type"):
            info.log(text="x", metadata={"tags": ["a", "b"]})

    def test_metadata_dict_is_copied(self):
        """Mutating the original dict after log() does not affect the entry."""
        original = {"k": "v"}
        info = Information(context_id="c")
        info.log(text="x", metadata=original)
        original["k"] = "mutated"
        assert info._entries[0].metadata["k"] == "v"


# ---------------------------------------------------------------------------
# __len__ and __repr__
# ---------------------------------------------------------------------------


class TestInformationMeta:
    def test_len_starts_at_zero(self):
        info = Information(context_id="c")
        assert len(info) == 0

    def test_len_increments(self):
        info = Information(context_id="c")
        info.log(text="one")
        info.log(text="two")
        assert len(info) == 2

    def test_repr_contains_context_id(self):
        info = Information(context_id="ctx-xyz")
        assert "ctx-xyz" in repr(info)

    def test_repr_contains_entry_count(self):
        info = Information(context_id="c")
        info.log(text="hello")
        assert "1" in repr(info)


# ---------------------------------------------------------------------------
# _drain() — checkpoint / periodic-flush behaviour
# ---------------------------------------------------------------------------


class TestDrain:
    def test_drain_returns_entries(self):
        info = Information(context_id="c")
        info.log(text="first")
        info.log(text="second")
        entries = info._drain()
        assert len(entries) == 2
        assert entries[0].text == "first"

    def test_drain_resets_buffer(self):
        info = Information(context_id="c")
        info.log(text="entry")
        info._drain()
        assert len(info) == 0

    def test_drain_allows_continued_logging(self):
        """Buffer is reused after drain — same object, multiple commits."""
        info = Information(context_id="c")
        info.log(text="before flush")
        info._drain()
        info.log(text="after flush")
        assert len(info) == 1
        assert info._entries[0].text == "after flush"

    def test_drain_empty_buffer_returns_empty_list(self):
        info = Information(context_id="c")
        entries = info._drain()
        assert entries == []
        assert len(info) == 0

# ---------------------------------------------------------------------------
# remove_all() — discard entire buffer
# ---------------------------------------------------------------------------


class TestRemoveAll:
    def test_remove_all_empties_buffer(self):
        info = Information(context_id="c")
        info.log(text="first")
        info.log(text="second")
        info.remove_all()
        assert len(info) == 0

    def test_remove_all_empty_buffer_is_noop(self):
        info = Information(context_id="c")
        info.remove_all()
        assert len(info) == 0

    def test_remove_all_allows_continued_logging(self):
        info = Information(context_id="c")
        info.log(text="stale")
        info.remove_all()
        info.log(text="fresh")
        assert len(info) == 1
        assert info._entries[0].text == "fresh"

    def test_remove_all_removes_pending_connections(self):
        info = Information(context_id="c")
        info.log(text="entry")
        info.connect(target="other-ctx-id", relationship="follows")
        info.remove_all()
        assert len(info._pending_connections) == 0


# ---------------------------------------------------------------------------
# remove(entry) — discard a specific entry by identity
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_returns_true_when_found(self):
        info = Information(context_id="c")
        entry = info.log(text="target")
        result = info.remove(entry)
        assert result is True
        assert len(info) == 0

    def test_remove_returns_false_when_not_in_buffer(self):
        info = Information(context_id="c")
        info2 = Information(context_id="c2")
        entry = info2.log(text="from other buffer")
        result = info.remove(entry)
        assert result is False

    def test_remove_first_entry_by_reference(self):
        info = Information(context_id="c")
        a = info.log(text="a")
        info.log(text="b")
        info.log(text="c")
        info.remove(a)
        assert len(info) == 2
        assert info._entries[0].text == "b"

    def test_remove_middle_entry_by_reference(self):
        info = Information(context_id="c")
        info.log(text="a")
        b = info.log(text="b")
        info.log(text="c")
        info.remove(b)
        texts = [e.text for e in info._entries]
        assert texts == ["a", "c"]

    def test_remove_last_entry_by_reference(self):
        info = Information(context_id="c")
        info.log(text="a")
        last = info.log(text="b")
        info.remove(last)
        assert len(info) == 1
        assert info._entries[0].text == "a"

    def test_remove_wrong_type_raises(self):
        info = Information(context_id="c")
        info.log(text="entry")
        with pytest.raises(NexusValidationError, match="InformationEntry"):
            info.remove(0)  # type: ignore

    def test_remove_after_drain_returns_false(self):
        """Entry is no longer in the buffer after _drain(); remove() returns False."""
        info = Information(context_id="c")
        entry = info.log(text="committed")
        info._drain()
        assert info.remove(entry) is False

    def test_remove_same_text_uses_identity_not_equality(self):
        """Two entries with identical text are distinct objects — remove only removes one."""
        info = Information(context_id="c")
        e1 = info.log(text="same")
        e2 = info.log(text="same")
        info.remove(e1)
        assert len(info) == 1
        assert info._entries[0] is e2


# ---------------------------------------------------------------------------
# remove_tagged(tag) — discard all entries with a given tag
# ---------------------------------------------------------------------------


class TestRemoveTagged:
    def test_remove_tagged_removes_matching_entries(self):
        info = Information(context_id="c")
        info.log(text="Order placed", tag="order-1")
        info.log(blob=b"receipt", document_type="pdf", tag="order-1")
        info.log(text="unrelated note")
        removed = info.remove_tagged("order-1")
        assert removed == 2
        assert len(info) == 1
        assert info._entries[0].text == "unrelated note"

    def test_remove_tagged_returns_zero_when_no_match(self):
        info = Information(context_id="c")
        info.log(text="no tag")
        removed = info.remove_tagged("nonexistent")
        assert removed == 0
        assert len(info) == 1

    def test_remove_tagged_empty_buffer_returns_zero(self):
        info = Information(context_id="c")
        assert info.remove_tagged("any") == 0

    def test_remove_tagged_removes_all_matching(self):
        info = Information(context_id="c")
        for i in range(5):
            info.log(text=f"item {i}", tag="batch")
        removed = info.remove_tagged("batch")
        assert removed == 5
        assert len(info) == 0

    def test_remove_tagged_leaves_other_tags_intact(self):
        info = Information(context_id="c")
        info.log(text="keep", tag="keep-tag")
        info.log(text="drop", tag="drop-tag")
        info.remove_tagged("drop-tag")
        assert len(info) == 1
        assert info._entries[0].tag == "keep-tag"

    def test_remove_tagged_empty_string_raises(self):
        info = Information(context_id="c")
        info.log(text="x")
        with pytest.raises(NexusValidationError, match="non-empty"):
            info.remove_tagged("")

    def test_remove_tagged_non_string_raises(self):
        info = Information(context_id="c")
        info.log(text="x")
        with pytest.raises(NexusValidationError, match="non-empty"):
            info.remove_tagged(None)  # type: ignore


# ---------------------------------------------------------------------------
# remove_before(timestamp) — discard entries logged before a checkpoint
# ---------------------------------------------------------------------------


class TestRemoveBefore:
    def test_remove_before_removes_earlier_entries(self):
        info = Information(context_id="c")
        info.log(text="before checkpoint")
        checkpoint = datetime.now(timezone.utc)
        info.log(text="after checkpoint")
        removed = info.remove_before(checkpoint)
        assert removed == 1
        assert len(info) == 1
        assert info._entries[0].text == "after checkpoint"

    def test_remove_before_returns_zero_when_all_after(self):
        checkpoint = datetime.now(timezone.utc) - timedelta(hours=1)
        info = Information(context_id="c")
        info.log(text="recent entry")
        removed = info.remove_before(checkpoint)
        assert removed == 0
        assert len(info) == 1

    def test_remove_before_removes_all_when_old_checkpoint(self):
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        removed = info.remove_before(future)
        assert removed == 2
        assert len(info) == 0

    def test_remove_before_empty_buffer_returns_zero(self):
        info = Information(context_id="c")
        checkpoint = datetime.now(timezone.utc)
        assert info.remove_before(checkpoint) == 0

    def test_remove_before_naive_datetime_raises(self):
        info = Information(context_id="c")
        info.log(text="entry")
        naive = datetime.now()  # no tzinfo
        with pytest.raises(NexusValidationError, match="timezone-aware"):
            info.remove_before(naive)

    def test_remove_before_non_datetime_raises(self):
        info = Information(context_id="c")
        info.log(text="entry")
        with pytest.raises(NexusValidationError, match="datetime"):
            info.remove_before("2025-01-01")  # type: ignore

    def test_remove_before_keeps_recent_discards_old(self):
        """remove_before keeps entries at or after the timestamp."""
        info = Information(context_id="c")
        info.log(text="old entry")
        cutoff = datetime.now(timezone.utc)
        info.log(text="recent entry A")
        info.log(text="recent entry B")
        # Discard old entries before cutoff:
        removed = info.remove_before(cutoff)
        assert removed == 1
        assert len(info) == 2
        texts = [e.text for e in info._entries]
        assert "recent entry A" in texts
        assert "recent entry B" in texts


# ---------------------------------------------------------------------------
# remove_since(checkpoint) — rollback: discard entries logged since checkpoint
# ---------------------------------------------------------------------------


class TestRemoveSince:
    def test_rollback_to_checkpoint(self):
        """Classic rollback: stable entries before checkpoint are kept."""
        info = Information(context_id="c")
        info.log(text="stable entry")
        checkpoint = datetime.now(timezone.utc)
        info.log(text="attempt A")
        info.log(text="attempt B")
        removed = info.remove_since(checkpoint)
        assert removed == 2
        assert len(info) == 1
        assert info._entries[0].text == "stable entry"

    def test_remove_since_future_removes_nothing(self):
        info = Information(context_id="c")
        info.log(text="entry")
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        removed = info.remove_since(future)
        assert removed == 0
        assert len(info) == 1

    def test_remove_since_past_removes_all(self):
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        removed = info.remove_since(past)
        assert removed == 2
        assert len(info) == 0

    def test_remove_since_empty_buffer_returns_zero(self):
        info = Information(context_id="c")
        checkpoint = datetime.now(timezone.utc)
        assert info.remove_since(checkpoint) == 0

    def test_remove_since_naive_datetime_raises(self):
        info = Information(context_id="c")
        info.log(text="entry")
        with pytest.raises(NexusValidationError, match="timezone-aware"):
            info.remove_since(datetime.now())

    def test_remove_since_non_datetime_raises(self):
        info = Information(context_id="c")
        info.log(text="entry")
        with pytest.raises(NexusValidationError, match="datetime"):
            info.remove_since("2025-01-01")  # type: ignore
