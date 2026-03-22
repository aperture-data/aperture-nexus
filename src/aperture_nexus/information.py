"""
Information — local buffer for multimodal inputs.

Nothing is written to ApertureDB until Memory.commit() is called.
Validation happens eagerly at log() time to surface errors as close
to the problem as possible — before any DB or model interaction.

Example:
    info = Information(context_id=ctx.id)
    info.log(text="Customer says order #4821 never arrived")
    info.log(image="screenshot.png")
    info.log(video="recording.mp4")
    info.log(blob=open("contract.pdf", "rb").read(), document_type="pdf")
    memory.commit(ctx, info)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage

from aperture_nexus.exceptions import NexusValidationError

logger = logging.getLogger(__name__)

# Accepted numpy dtypes for image arrays
_IMAGE_DTYPES = {np.dtype("uint8"), np.dtype("float32")}

# Accepted number of channels for image arrays
_IMAGE_CHANNELS = {1, 3, 4}


@dataclass
class _LogEntry:
    """Internal: one item added via Information.log() or .query()."""

    text: Optional[str] = None
    image: Optional[Any] = None   # PIL.Image, np.ndarray, bytes, or str
    video: Optional[Any] = None   # str (path/url) or bytes
    blob: Optional[bytes] = None
    document_type: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = None
    is_query: bool = False


class Information:
    """Local buffer for multimodal inputs.

    Nothing is written to ApertureDB until Memory.commit() is called.
    Validation happens eagerly at log() time — bad inputs raise
    NexusValidationError immediately, before any DB or model call.

    Args:
        context_id: ID of the Context this information belongs to.
            Use ``ctx.id``.

    Example:
        info = Information(context_id=ctx.id)

        # Text
        info.log(text="Customer says order #4821 never arrived")

        # Image — any form
        info.log(image="screenshot.png")
        info.log(image="https://example.com/photo.jpg")
        info.log(image=pil_image)
        info.log(image=numpy_array)

        # Video
        info.log(video="recording.mp4")

        # Blob — document_type required
        info.log(blob=pdf_bytes, document_type="pdf")

        # Combined — one log entry, multiple modalities
        info.log(text="See attached", blob=pdf_bytes, document_type="pdf")

        # Pre-computed embedding — skips model call at commit time
        info.log(
            image=img,
            embedding=my_vector,
            embedding_model="clip-vit-base-patch32",
        )
    """

    def __init__(self, context_id: str) -> None:
        if not isinstance(context_id, str) or not context_id.strip():
            raise NexusValidationError(
                "context_id must be a non-empty string. "
                "Use ctx.id from a Context instance."
            )
        self.context_id = context_id
        self._entries: list[_LogEntry] = []
        logger.debug(
            "Information buffer created: context_id=%r", context_id
        )

    def log(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        blob: Optional[bytes] = None,
        document_type: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Add one entry to the buffer.

        Multiple modalities can be combined in a single call (e.g.
        text + blob for "see attached PDF"). Validation happens
        immediately — bad inputs raise NexusValidationError before
        anything is stored.

        Args:
            text: Plain text. Long text is chunked automatically at
                commit time.
            image: Image in any common form: file path, URL, bytes,
                PIL Image, or numpy array (HW or HWC uint8/float32).
            video: Video file path, URL, or raw bytes.
            blob: Raw bytes for any binary format. Requires
                ``document_type``.
            document_type: File extension for blobs: ``"pdf"``,
                ``"mp3"``, ``"docx"``, ``"csv"``, etc.
            embedding: Pre-computed embedding vector (1D float array).
                Skips model call at commit time. Requires
                ``embedding_model``.
            embedding_model: Name of the model that produced the
                embedding. Required when ``embedding`` is provided.

        Raises:
            NexusValidationError: If input is invalid (missing file,
                wrong numpy shape, missing document_type for blob,
                embedding provided without embedding_model, etc.)
        """
        if all(v is None for v in [text, image, video, blob]):
            raise NexusValidationError(
                "At least one of text, image, video, or blob "
                "must be provided."
            )

        validated_text = _validate_text(text)
        validated_image = _validate_image(image)
        validated_video = _validate_video(video)
        validated_blob, validated_doc_type = _validate_blob(
            blob, document_type
        )
        validated_emb, validated_emb_model = _validate_embedding(
            embedding, embedding_model
        )

        entry = _LogEntry(
            text=validated_text,
            image=validated_image,
            video=validated_video,
            blob=validated_blob,
            document_type=validated_doc_type,
            embedding=validated_emb,
            embedding_model=validated_emb_model,
        )
        self._entries.append(entry)
        logger.debug(
            "Logged entry %d for context_id=%r",
            len(self._entries),
            self.context_id,
        )

    def query(self, text: str) -> None:
        """Log a retrieval intent.

        Records what the user or agent was looking for. Stored as
        metadata and used to improve future search relevance.

        Args:
            text: Description of what is being searched for.

        Raises:
            NexusValidationError: If text is empty or not a string.

        Example:
            info.query("what did we discuss last quarter?")
        """
        if not isinstance(text, str) or not text.strip():
            raise NexusValidationError(
                "Query text must be a non-empty string."
            )
        entry = _LogEntry(text=text.strip(), is_query=True)
        self._entries.append(entry)
        logger.debug(
            "Logged query entry for context_id=%r", self.context_id
        )

    def __len__(self) -> int:
        """Return the number of logged entries."""
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"Information(context_id={self.context_id!r},"
            f" entries={len(self._entries)})"
        )


# ---------------------------------------------------------------------------
# Validation helpers — module-private
# ---------------------------------------------------------------------------


def _validate_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        raise NexusValidationError(
            f"text must be a string. Got {type(text).__name__!r}."
        )
    if not text.strip():
        raise NexusValidationError("text must not be empty.")
    return text


def _validate_image(image: Optional[Any]) -> Optional[Any]:
    if image is None:
        return None

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            return image  # URL — reachability checked at commit time
        path = Path(image)
        if not path.is_file():
            raise NexusValidationError(
                f"Image file not found: {image}. "
                "Provide a valid file path, URL, PIL Image, "
                "numpy array, or raw bytes."
            )
        return image

    if isinstance(image, bytes):
        return image

    if isinstance(image, PILImage.Image):
        return image

    if isinstance(image, np.ndarray):
        _validate_image_array(image)
        return image

    raise NexusValidationError(
        f"Unsupported image type: {type(image).__name__!r}. "
        "Provide a file path, URL, bytes, PIL Image, or numpy array."
    )


def _validate_image_array(arr: np.ndarray) -> None:
    if arr.ndim not in {2, 3}:
        raise NexusValidationError(
            "Image numpy array must be 2D (H, W) or 3D (H, W, C). "
            f"Got shape {arr.shape}."
        )
    if arr.ndim == 3 and arr.shape[2] not in _IMAGE_CHANNELS:
        raise NexusValidationError(
            "Image array channel count must be 1, 3, or 4. "
            f"Got shape {arr.shape}."
        )
    if arr.dtype not in _IMAGE_DTYPES:
        raise NexusValidationError(
            f"Image array dtype must be uint8 or float32. "
            f"Got {arr.dtype}."
        )


def _validate_video(video: Optional[Any]) -> Optional[Any]:
    if video is None:
        return None

    if isinstance(video, str):
        if video.startswith(("http://", "https://")):
            return video  # URL — reachability checked at commit time
        path = Path(video)
        if not path.is_file():
            raise NexusValidationError(
                f"Video file not found: {video}. "
                "Provide a valid file path, URL, or raw bytes."
            )
        return video

    if isinstance(video, bytes):
        return video

    raise NexusValidationError(
        f"Unsupported video type: {type(video).__name__!r}. "
        "Provide a file path, URL, or bytes."
    )


def _validate_blob(
    blob: Optional[bytes],
    document_type: Optional[str],
) -> tuple[Optional[bytes], Optional[str]]:
    if blob is None:
        if document_type is not None:
            raise NexusValidationError(
                "document_type provided but blob is None. "
                "Provide blob=<bytes> alongside document_type."
            )
        return None, None

    if not isinstance(blob, bytes):
        raise NexusValidationError(
            f"blob must be bytes. Got {type(blob).__name__!r}."
        )

    if not document_type:
        raise NexusValidationError(
            "document_type is required when blob is provided. "
            'Examples: "pdf", "mp3", "docx", "csv".'
        )

    if not isinstance(document_type, str) or not document_type.strip():
        raise NexusValidationError(
            "document_type must be a non-empty string. "
            'Examples: "pdf", "mp3", "docx".'
        )

    return blob, document_type.strip().lower()


def _validate_embedding(
    embedding: Optional[np.ndarray],
    embedding_model: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if embedding is None:
        return None, None

    if not isinstance(embedding, np.ndarray):
        raise NexusValidationError(
            "embedding must be a numpy array. "
            f"Got {type(embedding).__name__!r}."
        )

    if embedding.ndim != 1:
        raise NexusValidationError(
            "embedding must be a 1D array. "
            f"Got shape {embedding.shape}."
        )

    if not np.issubdtype(embedding.dtype, np.floating):
        raise NexusValidationError(
            "embedding dtype must be a float type. "
            f"Got {embedding.dtype}."
        )

    if not embedding_model:
        raise NexusValidationError(
            "embedding_model is required when embedding is provided. "
            "Provide the name of the model that produced the embedding "
            "(e.g. 'clip-vit-base-patch32')."
        )

    return embedding, embedding_model
