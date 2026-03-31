"""
Integration tests for the adb-nexus CLI against a real ApertureDB instance.

Spins up ApertureDB community edition via Docker (see conftest.py).
Each test runs in a tmp_path sandbox so no files leak between tests.

Run with:
    pytest -m integration
    make test-integration
"""

import os
import shutil
import subprocess
import sys
import uuid

import pytest
from typer.testing import CliRunner

from aperture_nexus.cli import app
from aperture_nexus.config import ENV_NEXUS_API_KEY


runner = CliRunner()


def _uid() -> str:
    return f"cli-test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke_init(tmp_path, user_id: str, extra_args: list | None = None):
    """Run `adb-nexus init --defaults` in tmp_path with a custom env file."""
    env_file = tmp_path / ".env"
    config_file = tmp_path / "aperture_nexus.json"
    args = [
        "init",
        "--defaults",
        "--config", str(config_file),
        "--env-file", str(env_file),
    ] + (extra_args or [])
    result = runner.invoke(app, args, env={"USER": user_id}, catch_exceptions=False)
    return result, config_file, env_file


# ---------------------------------------------------------------------------
# init: happy path
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInitHappyPath:
    def test_exits_zero(self, adb_config_name, tmp_path):
        """init returns exit code 0 on success."""
        result, _, _ = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output

    def test_creates_config_file(self, adb_config_name, tmp_path):
        """init writes aperture_nexus.json when it does not exist."""
        result, config_file, _ = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output
        assert config_file.exists()

    def test_config_file_is_valid_json(self, adb_config_name, tmp_path):
        """The written config file is valid JSON."""
        import json
        result, config_file, _ = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output
        content = config_file.read_text()
        json.loads(content)  # raises if invalid

    def test_writes_api_key_to_env_file(self, adb_config_name, tmp_path):
        """init writes NEXUS_API_KEY=<value> to the .env file."""
        result, _, env_file = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output
        assert env_file.exists()
        content = env_file.read_text()
        assert f"{ENV_NEXUS_API_KEY}=" in content

    def test_api_key_is_non_empty(self, adb_config_name, tmp_path):
        """The written API key is a non-empty string."""
        result, _, env_file = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{ENV_NEXUS_API_KEY}="):
                key = line.split("=", 1)[1].strip()
                assert key, "NEXUS_API_KEY must not be empty"
                return
        pytest.fail(f"{ENV_NEXUS_API_KEY} not found in .env")

    def test_output_confirms_principal_created(self, adb_config_name, tmp_path):
        """init output confirms the principal was created."""
        uid = _uid()
        result, _, _ = _invoke_init(tmp_path, uid)
        assert result.exit_code == 0, result.output
        assert "created" in result.output.lower()
        assert uid in result.output

    def test_output_shows_authenticate_snippet(self, adb_config_name, tmp_path):
        """init output includes a working code snippet for authenticate()."""
        uid = _uid()
        result, _, _ = _invoke_init(tmp_path, uid)
        assert result.exit_code == 0, result.output
        assert "memory.authenticate" in result.output
        assert uid in result.output

    def test_uses_existing_config_file(self, adb_config_name, tmp_path):
        """init reuses an existing config file and does not overwrite it."""
        config_file = tmp_path / "aperture_nexus.json"
        config_file.write_text('{"existing": true}\n')
        result, _, _ = _invoke_init(tmp_path, _uid())
        assert result.exit_code == 0, result.output
        assert "Using existing config" in result.output
        # Original content is preserved
        import json
        assert json.loads(config_file.read_text()).get("existing") is True


# ---------------------------------------------------------------------------
# init: the created principal works with Memory.authenticate()
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInitPrincipalIsUsable:
    def test_api_key_authenticates(self, adb_config_name, tmp_path):
        """The NEXUS_API_KEY written by init authenticates successfully."""
        uid = _uid()
        result, config_file, env_file = _invoke_init(tmp_path, uid)
        assert result.exit_code == 0, result.output

        # Extract the key
        api_key = None
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{ENV_NEXUS_API_KEY}="):
                api_key = line.split("=", 1)[1].strip()
        assert api_key, "No API key found in .env"

        # Authenticate — should succeed without admin credentials
        from aperture_nexus.memory import Memory
        memory = Memory(config=str(config_file))
        principal = memory.authenticate(user_id=uid, api_key=api_key)
        assert principal.user_id == uid

    def test_duplicate_user_id_fails(self, adb_config_name, tmp_path):
        """Running init twice with the same user_id fails on the second run."""
        uid = _uid()
        # First run succeeds
        r1, config_file, env_file = _invoke_init(tmp_path, uid)
        assert r1.exit_code == 0, r1.output

        # Second run with same user_id should fail
        r2, _, _ = _invoke_init(tmp_path, uid)
        assert r2.exit_code != 0
        assert "already exists" in r2.output.lower() or "failed" in r2.output.lower()

    def teardown_method(self, method):
        """Best-effort cleanup: nothing to do — testcontainers tears down the DB."""
        pass


# ---------------------------------------------------------------------------
# init: --config path override
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Installed entry point
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInstalledEntryPoint:
    def test_adb_nexus_binary_is_on_path(self):
        """The adb-nexus binary is discoverable on PATH."""
        assert shutil.which("adb-nexus") is not None, (
            "adb-nexus not found on PATH. "
            "Install with: pip install -e . (or pip install aperture-nexus)"
        )

    def test_adb_nexus_help_exits_zero(self):
        """adb-nexus --help exits 0 via the real shell entry point."""
        result = subprocess.run(
            ["adb-nexus", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "init" in result.stdout
        assert "validate" in result.stdout

    def test_adb_nexus_init_defaults_exits_zero(self, adb_config_name, tmp_path):
        """adb-nexus init --defaults succeeds as a real subprocess."""
        config_file = tmp_path / "aperture_nexus.json"
        env_file = tmp_path / ".env"
        env = {**os.environ, "USER": _uid()}
        result = subprocess.run(
            [
                "adb-nexus", "init", "--defaults",
                "--config", str(config_file),
                "--env-file", str(env_file),
            ],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert config_file.exists()
        assert env_file.exists()
        assert f"{ENV_NEXUS_API_KEY}=" in env_file.read_text()


# ---------------------------------------------------------------------------
# init: --config path override
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInitConfigPath:
    def test_custom_config_path(self, adb_config_name, tmp_path):
        """--config writes config to the given path, not the default."""
        custom = tmp_path / "subdir" / "my_config.json"
        custom.parent.mkdir()
        env_file = tmp_path / ".env"
        result = runner.invoke(app, [
            "init", "--defaults",
            "--config", str(custom),
            "--env-file", str(env_file),
        ], env={"USER": _uid()}, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert custom.exists()
        assert not (tmp_path / "aperture_nexus.json").exists()
