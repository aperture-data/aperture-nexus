"""
CLIP-based embedding backend for aperture-nexus.

Why CLIP is the default
-----------------------
CLIP (Contrastive Language-Image Pre-training) handles text, images, and
video frames in a single model with a shared embedding space. This means:
- One model, one index, three modalities.
- Cross-modal search works out of the box: a text query can find images,
  an image query can find related text snippets.

CLIP text search limitations
-----------------------------
CLIP was trained on image-text pairs, not document retrieval tasks.
For workloads that are primarily text-based, a purpose-built text
embedding model will give significantly better recall:
- BGE-M3 (bge-m3): strong multilingual, long-document retrieval
- text-embedding-3-small (OpenAI API): strong general-purpose text
- Instructor-XL: task-aware embeddings

CLIP's text encoder also truncates input at 77 tokens (roughly 300
characters). Longer passages must be chunked before embedding. Dedicated
text models handle up to 8192 tokens.

Use CLIP when you need cross-modal search (text ↔ images ↔ video) or
when all your data is visual. Use a purpose-built text model when your
workload is primarily text retrieval.

Install:
    pip install aperture-nexus[clip]
"""

from __future__ import annotations

import io
import logging

import numpy as np

from aperture_nexus.exceptions import NexusConfigError, NexusValidationError

logger = logging.getLogger(__name__)

CLIP_DEFAULT_MODEL = "ViT-B/16"
CLIP_DEFAULT_PRETRAINED = "openai"  # OpenAI pretrained weights, not the API

# Known CLIP model names users can specify
_CLIP_KNOWN = {
    "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14",
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14",
    "RN50", "RN101",
}


def is_clip_model(name: str) -> bool:
    """Return True if name refers to a known CLIP architecture."""
    return name in _CLIP_KNOWN or name.startswith(("ViT-", "RN", "convnext_"))


def _normalize_clip_name(name: str) -> str:
    """Normalize slash variants to hyphen (ViT-B/32 → ViT-B-32 for open_clip).

    open_clip uses slash names internally but also accepts hyphens.
    """
    # open_clip accepts both; pass through
    return name


class ClipEmbedder:
    """CLIP-based embedder for text, images, and video frames.

    Lazy-loads the model on first use. Thread-safe once loaded.
    Model is cached on the instance — reuse the same ClipEmbedder
    for all calls within a Memory instance.

    Args:
        model_name: CLIP model name. Default: "ViT-B/16".
        pretrained: Pretrained weights name. Default: "openai".

    Raises:
        NexusConfigError: If open-clip-torch is not installed.
    """

    def __init__(
        self,
        model_name: str = CLIP_DEFAULT_MODEL,
        pretrained: str = CLIP_DEFAULT_PRETRAINED,
    ):
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load CLIP model on first use."""
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError:
            raise NexusConfigError(
                "CLIP embedding requires open-clip-torch. "
                "Install it with: pip install aperture-nexus[clip]"
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        # import torch for later use
        self._torch = torch

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of text strings.

        Note: CLIP truncates text at 77 tokens (approximately 300 characters).
        For longer text, chunk before calling this method.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float32 numpy arrays, one per input string.

        Raises:
            NexusConfigError: If open-clip-torch is not installed.
        """
        self._load()
        tokens = self._tokenizer(texts)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return [features[i].cpu().numpy().astype(np.float32) for i in range(len(texts))]

    def embed_image(self, image) -> np.ndarray:
        """Embed one image (PIL Image, np.ndarray, bytes, or file path str).

        Args:
            image: Input image as PIL Image, numpy array, bytes, or file path.

        Returns:
            Float32 numpy array of shape (D,).

        Raises:
            NexusConfigError: If open-clip-torch is not installed.
            NexusValidationError: If the image type is not supported.
        """
        self._load()
        import PIL.Image as PILImage

        if isinstance(image, bytes):
            pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            pil = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil = PILImage.fromarray(
                image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
            ).convert("RGB")
        elif isinstance(image, PILImage.Image):
            pil = image.convert("RGB")
        else:
            raise NexusValidationError(
                f"Cannot embed image type {type(image).__name__!r}."
            )

        tensor = self._preprocess(pil).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().astype(np.float32)

    def embed_video(
        self,
        video,
        frame_interval: int = 30,
        frames_per_clip: int = 10,
    ) -> list[tuple[np.ndarray, dict]]:
        """Embed a video as a list of per-clip embeddings.

        Each clip segment gets one embedding (mean of frame embeddings within
        that segment). This preserves temporal structure for search: a text
        query can find the specific part of a video that matches, not just
        whether the video matches at all.

        Requires opencv-python for frame extraction.

        Args:
            video: File path (str) or bytes.
            frame_interval: Sample one frame every N video frames. Default 30
                means one frame per second at 30 fps.
            frames_per_clip: Number of sampled frames to group into one clip
                segment. Default 10 means 10 sampled frames per clip.
                At frame_interval=30 and frames_per_clip=10, each clip covers
                ~300 original frames (~10 seconds at 30 fps).

        Returns:
            List of ``(embedding, metadata)`` tuples, one per clip segment.
            ``embedding`` is float32 of shape (D,).
            ``metadata`` keys: ``start_frame`` (int), ``stop_frame`` (int) —
            frame numbers in the original video's frame index.

        Raises:
            NexusConfigError: If open-clip-torch or opencv-python is not installed.
            NexusValidationError: If no frames could be extracted.
        """
        self._load()
        try:
            import cv2
        except ImportError:
            raise NexusConfigError(
                "Video embedding requires opencv-python. "
                "Install it with: pip install opencv-python"
            )

        import os
        import tempfile
        import PIL.Image as PILImage

        # Write bytes to a temp file if needed
        tmp = None
        if isinstance(video, bytes):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(video)
            tmp.flush()
            path = tmp.name
        else:
            path = video

        try:
            cap = cv2.VideoCapture(path)
            # List of (original_frame_number, PIL image)
            sampled: list[tuple[int, any]] = []
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sampled.append((frame_number, PILImage.fromarray(frame_rgb)))
                frame_number += 1
            cap.release()
        finally:
            if tmp is not None:
                os.unlink(path)

        if not sampled:
            raise NexusValidationError(
                "No frames could be extracted from the video."
            )

        # Group sampled frames into clip segments
        results: list[tuple[np.ndarray, dict]] = []
        for clip_idx in range(0, len(sampled), frames_per_clip):
            clip_frames = sampled[clip_idx: clip_idx + frames_per_clip]
            start_frame = clip_frames[0][0]
            stop_frame = clip_frames[-1][0]

            frame_embeddings = [self.embed_image(pil) for _, pil in clip_frames]
            clip_emb = np.mean(frame_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(clip_emb)
            if norm > 0:
                clip_emb = clip_emb / norm

            results.append((clip_emb, {"start_frame": start_frame, "stop_frame": stop_frame}))

        return results


# Module-level embedder cache keyed by (model_name, pretrained)
_embedder_cache: dict[tuple, ClipEmbedder] = {}


def get_clip_embedder(
    model_name: str = CLIP_DEFAULT_MODEL,
    pretrained: str = CLIP_DEFAULT_PRETRAINED,
) -> ClipEmbedder:
    """Return a cached ClipEmbedder for the given model.

    Args:
        model_name: CLIP model name. Default: "ViT-B/16".
        pretrained: Pretrained weights name. Default: "openai".

    Returns:
        A ClipEmbedder instance (lazily loads the model on first use).
    """
    key = (model_name, pretrained)
    if key not in _embedder_cache:
        _embedder_cache[key] = ClipEmbedder(model_name, pretrained)
    return _embedder_cache[key]
