"""
Unit tests for aperture_nexus.information.

Tests cover:
- Construction (valid, invalid context_id)
- log(): happy paths for all modalities (text, image, video, blob)
- log(): image input variants (path, URL, bytes, PIL, numpy)
- log(): blob input variants (bytes, file path, URL)
- log(): validation errors (missing file, wrong dtype, missing
  document_type, empty text, no modality, etc.)
- log(): embedding with and without embedding_model
- log(): combined modalities in one call
- clear(): empties buffer without committing
- remove(): removes a single entry by index
- __len__ and __repr__

No live ApertureDB instance required.
"""

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from aperture_nexus.exceptions import NexusValidationError
from aperture_nexus.information import Information


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
        info.log(text="Hello world")
        assert len(info) == 1
        assert info._entries[0].text == "Hello world"

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
# clear() — discard entire buffer
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_buffer(self):
        info = Information(context_id="c")
        info.log(text="first")
        info.log(text="second")
        info.clear()
        assert len(info) == 0

    def test_clear_empty_buffer_is_noop(self):
        info = Information(context_id="c")
        info.clear()
        assert len(info) == 0

    def test_clear_allows_continued_logging(self):
        info = Information(context_id="c")
        info.log(text="stale")
        info.clear()
        info.log(text="fresh")
        assert len(info) == 1
        assert info._entries[0].text == "fresh"

    def test_clear_removes_pending_connections(self):
        info = Information(context_id="c")
        info.log(text="entry")
        info.connect(target="other-ctx-id", relationship="follows")
        info.clear()
        assert len(info._pending_connections) == 0


# ---------------------------------------------------------------------------
# remove() — discard a single entry by index
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_first_entry(self):
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        info.log(text="c")
        info.remove(0)
        assert len(info) == 2
        assert info._entries[0].text == "b"

    def test_remove_last_entry(self):
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        info.remove(1)
        assert len(info) == 1
        assert info._entries[0].text == "a"

    def test_remove_middle_entry(self):
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        info.log(text="c")
        info.remove(1)
        texts = [e.text for e in info._entries]
        assert texts == ["a", "c"]

    def test_remove_negative_index(self):
        """Python-style negative indexing is supported."""
        info = Information(context_id="c")
        info.log(text="a")
        info.log(text="b")
        info.remove(-1)
        assert len(info) == 1
        assert info._entries[0].text == "a"

    def test_remove_out_of_range_raises(self):
        info = Information(context_id="c")
        info.log(text="only entry")
        with pytest.raises(IndexError, match="out of range"):
            info.remove(5)

    def test_remove_empty_buffer_raises(self):
        info = Information(context_id="c")
        with pytest.raises(IndexError):
            info.remove(0)

    def test_remove_non_integer_raises(self):
        from aperture_nexus.exceptions import NexusValidationError
        info = Information(context_id="c")
        info.log(text="entry")
        with pytest.raises(NexusValidationError, match="integer"):
            info.remove("0")  # type: ignore

    def test_remove_only_entry_empties_buffer(self):
        info = Information(context_id="c")
        info.log(text="solo")
        info.remove(0)
        assert len(info) == 0
