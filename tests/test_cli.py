"""
Tests for aperture_nexus.cli internal helpers.

Full CLI command tests (init, validate, ui) require a live ApertureDB
instance and belong in tests/integration/. This file covers pure-logic
helpers that can be tested without any DB access.
"""

import pytest
from pathlib import Path

from aperture_nexus.cli import _write_env_key


# ---------------------------------------------------------------------------
# _write_env_key()
# ---------------------------------------------------------------------------


class TestWriteEnvKey:
    KEY = "NEXUS_API_KEY"
    VAL = "newkey123"

    def test_creates_file_when_absent(self, tmp_path):
        env = tmp_path / ".env"
        _write_env_key(env, self.KEY, self.VAL)
        assert env.read_text() == f"{self.KEY}={self.VAL}\n"

    def test_appends_key_when_absent_from_existing_file(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=1\nBAR=2\n")
        _write_env_key(env, self.KEY, self.VAL)
        assert env.read_text() == f"FOO=1\nBAR=2\n{self.KEY}={self.VAL}\n"

    def test_updates_key_in_place(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(f"FOO=1\n{self.KEY}=oldvalue\nBAR=2\n")
        _write_env_key(env, self.KEY, self.VAL)
        assert env.read_text() == f"FOO=1\n{self.KEY}={self.VAL}\nBAR=2\n"

    def test_preserves_surrounding_variables(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(f"APERTUREDB_JSON=...\n{self.KEY}=old\nLOG_LEVEL=ERROR\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        assert lines[0] == "APERTUREDB_JSON=..."
        assert lines[1] == f"{self.KEY}={self.VAL}"
        assert lines[2] == "LOG_LEVEL=ERROR"

    def test_preserves_blank_lines(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("FOO=1\n\nBAR=2\n")
        _write_env_key(env, self.KEY, self.VAL)
        assert "\n\n" in env.read_text()

    def test_preserves_comments(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(f"# my comment\n{self.KEY}=old\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        assert lines[0] == "# my comment"
        assert lines[1] == f"{self.KEY}={self.VAL}"

    def test_commented_out_key_not_matched(self, tmp_path):
        """A commented-out key must not be activated."""
        env = tmp_path / ".env"
        env.write_text(f"# {self.KEY}=commented\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        # Comment line untouched, new line appended
        assert lines[0] == f"# {self.KEY}=commented"
        assert lines[1] == f"{self.KEY}={self.VAL}"

    def test_similar_key_name_not_matched(self, tmp_path):
        """NEXUS_API_KEY_EXTRA must not be overwritten when writing NEXUS_API_KEY."""
        env = tmp_path / ".env"
        env.write_text(f"{self.KEY}_EXTRA=other\nFOO=1\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        assert lines[0] == f"{self.KEY}_EXTRA=other"
        assert f"{self.KEY}={self.VAL}" in lines

    def test_key_with_space_before_equals_updated(self, tmp_path):
        """KEY =value (with space) is recognised and normalised."""
        env = tmp_path / ".env"
        env.write_text(f"{self.KEY} =oldvalue\nBAR=2\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        assert lines[0] == f"{self.KEY}={self.VAL}"
        assert lines[1] == "BAR=2"

    def test_only_first_occurrence_updated(self, tmp_path):
        """Duplicate keys: only the first occurrence is updated."""
        env = tmp_path / ".env"
        env.write_text(f"{self.KEY}=first\n{self.KEY}=second\n")
        _write_env_key(env, self.KEY, self.VAL)
        lines = env.read_text().splitlines()
        assert lines[0] == f"{self.KEY}={self.VAL}"
        assert lines[1] == f"{self.KEY}=second"

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        """Temp file must not remain after a successful write."""
        env = tmp_path / ".env"
        _write_env_key(env, self.KEY, self.VAL)
        assert not (tmp_path / ".env.tmp").exists()
