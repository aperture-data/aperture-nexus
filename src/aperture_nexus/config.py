"""
Configuration loading and validation for aperture-nexus.

Config is read from aperture_nexus.json, discovered in this order:

    1. Explicit path passed to load_config(path=...)
    2. APERTURE_NEXUS_CONFIG environment variable
    3. ./aperture_nexus.json  (current working directory)
    4. ~/.aperture_nexus/config.json  (user home — no root required)
    5. Built-in defaults (no models configured)

Certain values can be overridden per-process via environment variables
without editing the file — see ENV_OVERRIDES below.

Generate a config file with:
    adb-nexus init

Validate your current config with:
    adb-nexus validate

Example:
    >>> from aperture_nexus.config import load_config
    >>> cfg = load_config()
    >>> cfg.processing.num_threads
    4
    >>> cfg.logging.level
    'ERROR'
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from aperture_nexus.exceptions import NexusConfigError

logger = logging.getLogger(__name__)

# Environment variable that points to a config file path.
ENV_CONFIG_PATH = "APERTURE_NEXUS_CONFIG"

# Per-value environment variable overrides.
# These take precedence over anything in the config file.
ENV_LOG_LEVEL = "APERTURE_NEXUS_LOG_LEVEL"
ENV_UI_API_KEY = "APERTURE_NEXUS_UI_API_KEY"

def _default_search_paths() -> list[Path]:
    """
    Return the default config file search paths in priority order.

    Called at discovery time (not import time) so that Path.cwd() and
    Path.home() reflect the process state at the moment of the call.
    All paths are user-accessible — no root required.
    """
    return [
        Path.cwd() / "aperture_nexus.json",
        Path.home() / ".aperture_nexus" / "config.json",
    ]


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class ModelsConfig(BaseModel):
    """
    AI models used for embedding and processing.

    Required only when process_and_commit() or async_process_and_commit()
    is called. If you only use commit() (raw storage), omit this section.

    Example (aperture_nexus.json):
        {
            "models": {
                "llm": "gpt-4o",
                "vlm": "clip-vit-base-patch32"
            }
        }
    """

    llm: Optional[str] = Field(
        default=None,
        description=(
            "Language model for text processing and summarization. "
            "Example: 'gpt-4o', 'gemini-2.5-pro'."
        ),
    )
    vlm: Optional[str] = Field(
        default=None,
        description=(
            "Vision-language model for image and video embedding. "
            "Example: 'clip-vit-base-patch32', 'gemini-embedding-001'."
        ),
    )


class ProcessingConfig(BaseModel):
    """
    Controls how aperture-nexus processes and stores multimodal data.

    All values have production-safe defaults. Most users never need to
    change these. Tune num_threads and batch_size for high-throughput
    workloads, and video_clip_duration to match your embedding model's
    expected input length.

    Example (aperture_nexus.json):
        {
            "processing": {
                "num_threads": 8,
                "video_clip_duration": 5.0
            }
        }
    """

    # General
    num_threads: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Parallel threads for ApertureDB writes.",
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Items per batch sent to ApertureDB.",
    )
    embedding_batch: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Items per model inference call for embedding generation.",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts for failed DB writes. Set to 0 to disable.",
    )
    retry_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Seconds between retry attempts.",
    )

    # Text chunking
    text_chunk_size: int = Field(
        default=2000,
        ge=64,
        description=(
            "Size of each text chunk before embedding. "
            "Unit is set by text_chunk_unit."
        ),
    )
    text_chunk_overlap: int = Field(
        default=200,
        ge=0,
        description=(
            "Overlap between consecutive chunks. "
            "Must be less than text_chunk_size."
        ),
    )
    text_chunk_unit: Literal["characters", "tokens"] = Field(
        default="characters",
        description=(
            "'characters' requires no extra dependencies. "
            "'tokens' requires pip install aperture-nexus[tokens]."
        ),
    )

    # Video processing
    video_clip_duration: float = Field(
        default=10.0,
        ge=1.0,
        description=(
            "Duration of each video clip in seconds. "
            "Adjust to match your VLM's expected input length."
        ),
    )
    video_clip_overlap: float = Field(
        default=0.5,
        ge=0.0,
        description="Overlap between consecutive video clips in seconds.",
    )
    video_frame_interval: int = Field(
        default=30,
        ge=1,
        description=(
            "Extract one frame every N frames when scene detection "
            "is disabled. At 30fps this extracts ~1 frame per second."
        ),
    )
    video_scene_detection: bool = Field(
        default=False,
        description=(
            "Extract frames at scene boundaries instead of fixed intervals. "
            "Requires pip install aperture-nexus[video]."
        ),
    )
    video_max_frames: int = Field(
        default=100,
        ge=1,
        description=(
            "Hard cap on frames extracted per video, "
            "regardless of method."
        ),
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "ProcessingConfig":
        """Chunk overlap must be strictly less than chunk size."""
        if self.text_chunk_overlap >= self.text_chunk_size:
            raise ValueError(
                f"text_chunk_overlap ({self.text_chunk_overlap}) "
                f"must be less than text_chunk_size "
                f"({self.text_chunk_size})."
            )
        return self

    @model_validator(mode="after")
    def validate_video_clip_overlap(self) -> "ProcessingConfig":
        """Clip overlap must be strictly less than clip duration."""
        if self.video_clip_overlap >= self.video_clip_duration:
            raise ValueError(
                f"video_clip_overlap ({self.video_clip_overlap}) "
                f"must be less than video_clip_duration "
                f"({self.video_clip_duration})."
            )
        return self


class LoggingConfig(BaseModel):
    """
    Controls aperture-nexus internal logging.

    Does not affect your application's logging configuration. Override
    without editing the file:
        export APERTURE_NEXUS_LOG_LEVEL=DEBUG

    Example (aperture_nexus.json):
        {
            "logging": {
                "level": "INFO"
            }
        }
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="ERROR",
        description=(
            "Minimum log level to emit. "
            "Use DEBUG for troubleshooting, ERROR for production."
        ),
    )


class MetricsConfig(BaseModel):
    """
    Prometheus metrics export configuration.

    Requires pip install aperture-nexus[metrics].
    Uses user-space ports only (>1024) — no root required.

    Example (aperture_nexus.json):
        {
            "metrics": {
                "enabled": true,
                "port": 9090
            }
        }
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable Prometheus metrics endpoint. "
            "Requires pip install aperture-nexus[metrics]."
        ),
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description=(
            "Port for the /metrics endpoint. "
            "Must be > 1024 (no root required)."
        ),
    )
    path: str = Field(
        default="/metrics",
        description="HTTP path for the metrics endpoint.",
    )


class UIConfig(BaseModel):
    """
    Web UI and REST API configuration.

    Requires pip install aperture-nexus[ui].

    Security: when host is not '127.0.0.1', api_key MUST be set.
    Never put the api_key value in aperture_nexus.json — use the
    APERTURE_NEXUS_UI_API_KEY environment variable instead.

    Example (aperture_nexus.json — local only):
        {
            "ui": {
                "enabled": true,
                "port": 8000
            }
        }

    Example (aperture_nexus.json — hosted, api_key via env var):
        {
            "ui": {
                "enabled": true,
                "host": "0.0.0.0",
                "port": 8080,
                "api_key": null
            }
        }

    Then set the key:
        export APERTURE_NEXUS_UI_API_KEY="your-secret-key"
    """

    enabled: bool = Field(
        default=False,
        description="Enable the web UI and REST API.",
    )
    host: str = Field(
        default="127.0.0.1",
        description=(
            "Bind address. '127.0.0.1' = local only (default). "
            "'0.0.0.0' = all interfaces — api_key MUST be set."
        ),
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description=(
            "Port to serve the UI on. "
            "Must be > 1024 (no root required)."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for all UI and API requests. "
            "Required when host is not '127.0.0.1'. "
            "Use APERTURE_NEXUS_UI_API_KEY env var — never hardcode this value."
        ),
    )

    @model_validator(mode="after")
    def validate_api_key_for_non_local(self) -> "UIConfig":
        """Enforce api_key when UI is exposed beyond localhost."""
        if self.enabled and self.host != "127.0.0.1" and not self.api_key:
            raise ValueError(
                "ui.api_key is required when ui.host is not '127.0.0.1'. "
                "Set it via the APERTURE_NEXUS_UI_API_KEY environment "
                "variable. Never put the API key value directly in "
                "aperture_nexus.json."
            )
        return self


class NexusConfig(BaseModel):
    """
    Top-level aperture-nexus configuration.

    Loaded automatically from aperture_nexus.json by load_config().
    Generate a config file with: adb-nexus init

    Attributes:
        models: AI models for embedding and processing. Optional.
        processing: Multimodal processing settings.
        logging: Internal log level.
        metrics: Prometheus metrics export. Disabled by default.
        ui: Web UI and REST API. Disabled by default.
    """

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


# ---------------------------------------------------------------------------
# Config discovery and loading
# ---------------------------------------------------------------------------


def _find_config_file() -> Optional[Path]:
    """
    Search for a config file in the standard discovery order.

    Returns the first path found, or None if no config file exists.
    Does not raise — callers decide how to handle a missing file.
    """
    # Check env var first
    env_path = os.environ.get(ENV_CONFIG_PATH)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        # Env var set but file not found — this is always an error
        raise NexusConfigError(
            f"Config file specified by {ENV_CONFIG_PATH} not found: "
            f"{env_path}. Verify the path is correct, or unset "
            f"{ENV_CONFIG_PATH} to use "
            f"automatic discovery."
        )

    for path in _default_search_paths():
        if path.is_file():
            logger.debug("Found config file at: %s", path)
            return path

    return None


def _apply_env_overrides(config: NexusConfig) -> NexusConfig:
    """
    Apply environment variable overrides to a loaded config.

    Environment variables always take precedence over file values.
    This allows per-process overrides without editing the config file.

    Args:
        config: The config loaded from file (or defaults).

    Returns:
        Config with environment variable overrides applied.
    """
    raw = config.model_dump()

    log_level = os.environ.get(ENV_LOG_LEVEL)
    if log_level:
        upper = log_level.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if upper not in valid:
            raise NexusConfigError(
                f"{ENV_LOG_LEVEL}={log_level!r} is not a valid log level. "
                f"Choose one of: {', '.join(sorted(valid))}."
            )
        raw["logging"]["level"] = upper
        logger.debug("Log level overridden by %s: %s", ENV_LOG_LEVEL, upper)

    ui_api_key = os.environ.get(ENV_UI_API_KEY)
    if ui_api_key:
        raw["ui"]["api_key"] = ui_api_key
        logger.debug("UI API key set from %s", ENV_UI_API_KEY)

    try:
        return NexusConfig.model_validate(raw)
    except Exception as e:
        raise NexusConfigError(
            f"Config became invalid after applying environment variable "
            f"overrides: {e}. Check your environment variables."
        ) from e


def _validate_optional_deps(config: NexusConfig) -> None:
    """
    Check that optional dependencies are installed for any enabled features.

    Raises NexusConfigError immediately with a clear install command
    rather than letting the error surface later as an ImportError.

    Args:
        config: The fully loaded and validated config.
    """
    if config.processing.video_scene_detection:
        try:
            import scenedetect  # noqa: F401
        except ImportError:
            raise NexusConfigError(
                "processing.video_scene_detection is enabled but PySceneDetect "
                "is not installed. "
                "Install it with: pip install aperture-nexus[video]"
            )

    if config.processing.text_chunk_unit == "tokens":
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            raise NexusConfigError(
                "processing.text_chunk_unit is 'tokens' but tiktoken "
                "is not installed. "
                "Install it with: pip install aperture-nexus[tokens]"
            )

    if config.metrics.enabled:
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            raise NexusConfigError(
                "metrics.enabled is true but prometheus-client "
                "is not installed. "
                "Install it with: pip install aperture-nexus[metrics]"
            )

    if config.ui.enabled:
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            raise NexusConfigError(
                "ui.enabled is true but the UI dependencies are not installed. "
                "Install them with: pip install aperture-nexus[ui]"
            )


def load_config(
    path: Optional[str | Path] = None,
    *,
    validate_deps: bool = True,
) -> NexusConfig:
    """
    Load and validate the aperture-nexus configuration.

    Searches for a config file in the standard order unless an explicit
    path is provided. Environment variable overrides are always applied
    after the file is loaded.

    Config discovery order:
        1. ``path`` argument (if provided)
        2. ``APERTURE_NEXUS_CONFIG`` environment variable
        3. ``./aperture_nexus.json``
        4. ``~/.aperture_nexus/config.json``
        5. Built-in defaults (no models configured)

    Args:
        path: Explicit path to a config file. Overrides all discovery.
        validate_deps: If True (default), check that optional dependencies
            are installed for any enabled features. Set to False in tests
            to skip dependency checks.

    Returns:
        Validated NexusConfig instance with environment overrides applied.

    Raises:
        NexusConfigError: If the config file exists but is invalid JSON,
            fails validation, specifies missing optional dependencies, or
            sets an unsafe UI configuration.

    Example:
        >>> from aperture_nexus.config import load_config
        >>> cfg = load_config()
        >>> cfg.processing.num_threads
        4

        >>> # Explicit path
        >>> cfg = load_config(path="/etc/myapp/aperture_nexus.json")

        >>> # The config file is optional — defaults are used if not found
        >>> cfg = load_config()   # works even with no config file
    """
    config_path: Optional[Path] = None

    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise NexusConfigError(
                f"Config file not found: {config_path}. "
                f"Run 'adb-nexus init' to generate one."
            )
    else:
        config_path = _find_config_file()

    if config_path is None:
        logger.debug(
            "No config file found — using built-in defaults. "
            "Run 'adb-nexus init' to generate a config file."
        )
        raw: dict = {}
    else:
        logger.debug("Loading config from: %s", config_path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise NexusConfigError(
                f"Config file at {config_path} is not valid JSON: {e}. "
                f"Run 'adb-nexus validate' to check your config, or "
                f"'adb-nexus init' to regenerate it."
            ) from e
        except OSError as e:
            raise NexusConfigError(
                f"Could not read config file at {config_path}: {e}."
            ) from e

    try:
        config = NexusConfig.model_validate(raw)
    except Exception as e:
        source = str(config_path) if config_path else "defaults"
        raise NexusConfigError(
            f"Config validation failed ({source}): {e}. "
            f"Check {config_path or 'your config'} against the schema, or "
            f"run 'adb-nexus init' to regenerate it."
        ) from e

    config = _apply_env_overrides(config)

    if validate_deps:
        _validate_optional_deps(config)

    # Apply the configured log level to the aperture_nexus logger hierarchy
    logging.getLogger("aperture_nexus").setLevel(config.logging.level)

    logger.debug("Config loaded successfully: %r", config)
    return config
