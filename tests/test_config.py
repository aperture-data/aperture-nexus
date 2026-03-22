"""
Unit tests for aperture_nexus.config.

Tests cover:
- Default config values
- Config file loading (valid, invalid JSON, missing)
- Config discovery order (explicit path, env var, cwd, home, defaults)
- Environment variable overrides (log level, UI API key)
- Validation rules (chunk overlap, video overlap, UI host/api_key)
- Optional dependency checks (mocked imports)

No live ApertureDB instance required.
"""

import json
import os
from pathlib import Path

import pytest

from aperture_nexus.config import (
    ENV_CONFIG_PATH,
    ENV_LOG_LEVEL,
    ENV_UI_API_KEY,
    NexusConfig,
    ProcessingConfig,
    UIConfig,
    load_config,
)
from aperture_nexus.exceptions import NexusConfigError


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_processing_defaults(self):
        cfg = load_config(validate_deps=False)
        assert cfg.processing.num_threads == 4
        assert cfg.processing.batch_size == 50
        assert cfg.processing.embedding_batch == 32
        assert cfg.processing.retry_attempts == 3
        assert cfg.processing.retry_interval == 1.0
        assert cfg.processing.text_chunk_size == 2000
        assert cfg.processing.text_chunk_overlap == 200
        assert cfg.processing.text_chunk_unit == "characters"
        assert cfg.processing.video_clip_duration == 10.0
        assert cfg.processing.video_clip_overlap == 0.5
        assert cfg.processing.video_frame_interval == 30
        assert cfg.processing.video_scene_detection is False
        assert cfg.processing.video_max_frames == 100

    def test_logging_default_is_error(self):
        cfg = load_config(validate_deps=False)
        assert cfg.logging.level == "ERROR"

    def test_metrics_disabled_by_default(self):
        cfg = load_config(validate_deps=False)
        assert cfg.metrics.enabled is False

    def test_ui_disabled_by_default(self):
        cfg = load_config(validate_deps=False)
        assert cfg.ui.enabled is False
        assert cfg.ui.host == "127.0.0.1"

    def test_models_empty_by_default(self):
        cfg = load_config(validate_deps=False)
        assert cfg.models.llm is None
        assert cfg.models.vlm is None


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestConfigFileLoading:
    def test_loads_valid_config_file(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({
            "models": {"llm": "gpt-4o", "vlm": "clip-vit-base-patch32"},
            "processing": {"num_threads": 8},
            "logging": {"level": "DEBUG"},
        }))

        cfg = load_config(path=config_file, validate_deps=False)

        assert cfg.models.llm == "gpt-4o"
        assert cfg.models.vlm == "clip-vit-base-patch32"
        assert cfg.processing.num_threads == 8
        assert cfg.logging.level == "DEBUG"

    def test_unspecified_fields_use_defaults(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({"processing": {"num_threads": 8}}))

        cfg = load_config(path=config_file, validate_deps=False)

        assert cfg.processing.num_threads == 8
        assert cfg.processing.batch_size == 50   # default

    def test_raises_on_missing_explicit_path(self, tmp_path):
        missing = tmp_path / "nonexistent.json"

        with pytest.raises(NexusConfigError, match="not found"):
            load_config(path=missing, validate_deps=False)

    def test_raises_on_invalid_json(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text("{ this is not valid json }")

        with pytest.raises(NexusConfigError, match="not valid JSON"):
            load_config(path=config_file, validate_deps=False)

    def test_raises_on_unreadable_file(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text("{}")
        config_file.chmod(0o000)

        try:
            with pytest.raises(NexusConfigError):
                load_config(path=config_file, validate_deps=False)
        finally:
            config_file.chmod(0o644)   # restore for cleanup

    def test_raises_on_invalid_field_value(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({"processing": {"num_threads": -1}}))

        with pytest.raises(NexusConfigError, match="validation failed"):
            load_config(path=config_file, validate_deps=False)

    def test_uses_defaults_when_no_file_found(self, tmp_path, monkeypatch):
        # Ensure discovery finds nothing
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)

        cfg = load_config(validate_deps=False)

        assert cfg.processing.num_threads == 4   # default


# ---------------------------------------------------------------------------
# Config discovery order
# ---------------------------------------------------------------------------


class TestConfigDiscovery:
    def test_explicit_path_takes_precedence(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({"processing": {"num_threads": 16}}))

        env_file = tmp_path / "env.json"
        env_file.write_text(json.dumps({"processing": {"num_threads": 2}}))
        monkeypatch.setenv(ENV_CONFIG_PATH, str(env_file))

        cfg = load_config(path=explicit, validate_deps=False)
        assert cfg.processing.num_threads == 16

    def test_env_var_path_used_when_no_explicit(self, tmp_path, monkeypatch):
        env_file = tmp_path / "env_config.json"
        env_file.write_text(json.dumps({"processing": {"num_threads": 7}}))
        monkeypatch.setenv(ENV_CONFIG_PATH, str(env_file))
        monkeypatch.chdir(tmp_path)

        cfg = load_config(validate_deps=False)
        assert cfg.processing.num_threads == 7

    def test_env_var_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_CONFIG_PATH, str(tmp_path / "ghost.json"))

        with pytest.raises(NexusConfigError, match=ENV_CONFIG_PATH):
            load_config(validate_deps=False)

    def test_cwd_config_discovered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)

        cwd_file = tmp_path / "aperture_nexus.json"
        cwd_file.write_text(json.dumps({"processing": {"num_threads": 6}}))

        cfg = load_config(validate_deps=False)
        assert cfg.processing.num_threads == 6

    def test_home_config_discovered_when_no_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)

        home_dir = tmp_path / "fakehome" / ".aperture_nexus"
        home_dir.mkdir(parents=True)
        home_file = home_dir / "config.json"
        home_file.write_text(json.dumps({"processing": {"num_threads": 3}}))

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        cfg = load_config(validate_deps=False)
        assert cfg.processing.num_threads == 3


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")
        cfg = load_config(validate_deps=False)
        assert cfg.logging.level == "DEBUG"

    def test_log_level_override_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(ENV_LOG_LEVEL, "info")
        cfg = load_config(validate_deps=False)
        assert cfg.logging.level == "INFO"

    def test_invalid_log_level_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_LOG_LEVEL, "VERBOSE")
        with pytest.raises(NexusConfigError, match="not a valid log level"):
            load_config(validate_deps=False)

    def test_ui_api_key_override(self, monkeypatch):
        monkeypatch.setenv(ENV_UI_API_KEY, "secret-key-123")
        cfg = load_config(validate_deps=False)
        assert cfg.ui.api_key == "secret-key-123"

    def test_env_overrides_file_value(self, tmp_path, monkeypatch):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({"logging": {"level": "INFO"}}))
        monkeypatch.setenv(ENV_LOG_LEVEL, "WARNING")

        cfg = load_config(path=config_file, validate_deps=False)
        assert cfg.logging.level == "WARNING"


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


class TestValidation:
    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="text_chunk_overlap"):
            ProcessingConfig(text_chunk_size=100, text_chunk_overlap=100)

    def test_chunk_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="text_chunk_overlap"):
            ProcessingConfig(text_chunk_size=500, text_chunk_overlap=500)

    def test_valid_chunk_overlap_accepted(self):
        cfg = ProcessingConfig(text_chunk_size=500, text_chunk_overlap=100)
        assert cfg.text_chunk_overlap == 100

    def test_video_clip_overlap_must_be_less_than_duration(self):
        with pytest.raises(ValueError, match="video_clip_overlap"):
            ProcessingConfig(video_clip_duration=5.0, video_clip_overlap=5.0)

    def test_ui_requires_api_key_for_non_local_host(self):
        with pytest.raises(ValueError, match="api_key is required"):
            UIConfig(enabled=True, host="0.0.0.0", api_key=None)

    def test_ui_local_host_does_not_require_api_key(self):
        cfg = UIConfig(enabled=True, host="127.0.0.1", api_key=None)
        assert cfg.enabled is True

    def test_ui_non_local_with_api_key_accepted(self):
        cfg = UIConfig(enabled=True, host="0.0.0.0", api_key="secret")
        assert cfg.api_key == "secret"

    def test_ui_api_key_enforcement_only_when_enabled(self):
        # Disabled UI with non-local host and no key should be fine
        cfg = UIConfig(enabled=False, host="0.0.0.0", api_key=None)
        assert cfg.enabled is False

    def test_num_threads_bounds(self):
        with pytest.raises(Exception):
            ProcessingConfig(num_threads=0)
        with pytest.raises(Exception):
            ProcessingConfig(num_threads=65)

    def test_port_must_be_above_1024(self):
        with pytest.raises(Exception):
            UIConfig(port=80)
        with pytest.raises(Exception):
            UIConfig(port=443)


# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------


class TestOptionalDeps:
    def test_scene_detection_missing_dep_raises(self, tmp_path, monkeypatch):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({
            "processing": {"video_scene_detection": True}
        }))

        monkeypatch.setitem(__builtins__ if isinstance(__builtins__, dict)
                            else vars(__builtins__), "scenedetect", None)

        # Simulate missing import by patching the import inside validate_deps
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"scenedetect": None}):
            with pytest.raises(NexusConfigError, match="pip install aperture-nexus\\[video\\]"):
                load_config(path=config_file, validate_deps=True)

    def test_tokens_missing_dep_raises(self, tmp_path, monkeypatch):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({
            "processing": {"text_chunk_unit": "tokens"}
        }))

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"tiktoken": None}):
            with pytest.raises(NexusConfigError, match="pip install aperture-nexus\\[tokens\\]"):
                load_config(path=config_file, validate_deps=True)

    def test_metrics_missing_dep_raises(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({"metrics": {"enabled": True}}))

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"prometheus_client": None}):
            with pytest.raises(NexusConfigError, match="pip install aperture-nexus\\[metrics\\]"):
                load_config(path=config_file, validate_deps=True)

    def test_ui_missing_dep_raises(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({"ui": {"enabled": True}}))

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"fastapi": None}):
            with pytest.raises(NexusConfigError, match="pip install aperture-nexus\\[ui\\]"):
                load_config(path=config_file, validate_deps=True)

    def test_validate_deps_false_skips_checks(self, tmp_path):
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text(json.dumps({
            "processing": {"video_scene_detection": True}
        }))

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"scenedetect": None}):
            # Should not raise — dependency check skipped
            cfg = load_config(path=config_file, validate_deps=False)
            assert cfg.processing.video_scene_detection is True
