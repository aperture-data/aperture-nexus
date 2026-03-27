"""
adb-nexus — CLI for aperture-nexus setup and management.

Commands:
    init      Interactive setup: ApertureDB connection, config file,
              principal creation, .env file. The only command that
              requires admin ApertureDB credentials.
    validate  Test the ApertureDB connection and config.
    stats     Show memory usage statistics.
    ui        Launch the web UI (requires aperture-nexus[ui]).

Admin credentials (APERTUREDB_KEY or APERTUREDB_USER/APERTUREDB_PASSWORD)
are required only for 'init'. All other operations — including
Memory.authenticate() at session time — use regular credentials.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print(
        "aperture-nexus CLI requires typer. "
        "Install it with: pip install aperture-nexus[ui]  "
        "or: pip install typer"
    )
    sys.exit(1)

from aperture_nexus.config import ENV_NEXUS_API_KEY, load_config
from aperture_nexus.exceptions import NexusConfigError, NexusConnectionError

app = typer.Typer(
    name="adb-nexus",
    help="aperture-nexus setup and management CLI.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to write aperture_nexus.json. Default: ./aperture_nexus.json",
    ),
    defaults: bool = typer.Option(
        False, "--defaults",
        help="Non-interactive: accept all defaults without prompting.",
    ),
    env_file: str = typer.Option(
        ".env", "--env-file",
        help="Path to .env file to write NEXUS_API_KEY into.",
    ),
) -> None:
    """Set up aperture-nexus: config file, principal, and .env.

    Requires admin ApertureDB credentials (APERTUREDB_KEY or
    APERTUREDB_USER/APERTUREDB_PASSWORD). Creates your principal in
    ApertureDB and writes NEXUS_API_KEY to .env. After init, no admin
    credentials are needed at session time.
    """
    typer.echo("aperture-nexus init")
    typer.echo("-" * 40)

    # Resolve config path
    config_path = Path(config) if config else Path("aperture_nexus.json")

    # Collect user_id
    if defaults:
        user_id = os.environ.get("USER") or os.environ.get("USERNAME") or "nexus_user"
        typer.echo(f"  user_id: {user_id} (default)")
    else:
        user_id = typer.prompt("  User ID for this principal", default=os.environ.get("USER", ""))
        if not user_id.strip():
            typer.echo("Error: user_id cannot be empty.", err=True)
            raise typer.Exit(1)

    # Write minimal config if it doesn't exist
    if not config_path.exists():
        config_path.write_text("{}\n")
        typer.echo(f"  Created {config_path}")
    else:
        typer.echo(f"  Using existing config: {config_path}")

    # Create principal via NexusAdmin
    try:
        from aperture_nexus.admin import NexusAdmin
        admin = NexusAdmin(config=str(config_path))
        api_key = admin.create_principal(user_id=user_id.strip())
    except NexusConnectionError as e:
        typer.echo(f"\nCould not connect to ApertureDB: {e}", err=True)
        typer.echo(
            "Ensure APERTUREDB_KEY (or APERTUREDB_USER/APERTUREDB_PASSWORD) "
            "is set with admin credentials.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\nFailed to create principal: {e}", err=True)
        raise typer.Exit(1)

    # Write NEXUS_API_KEY to .env
    env_path = Path(env_file)
    _write_env_key(env_path, ENV_NEXUS_API_KEY, api_key)

    typer.echo(f"\n  Principal {user_id!r} created.")
    typer.echo(f"  {ENV_NEXUS_API_KEY} written to {env_path}")
    typer.echo(
        f"\nSetup complete. Load your .env and authenticate with:\n"
        f"\n    memory = Memory()\n"
        f"    principal = memory.authenticate(\n"
        f"        user_id={user_id!r},\n"
        f"        api_key=os.environ[{ENV_NEXUS_API_KEY!r}],\n"
        f"    )\n"
    )

    # Warn if .env is not gitignored
    _check_gitignore(env_path)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to aperture_nexus.json.",
    ),
) -> None:
    """Test ApertureDB connection and validate config."""
    try:
        cfg = load_config(path=config, validate_deps=True)
    except NexusConfigError as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Config loaded: {cfg}")

    try:
        from aperture_nexus._client import get_connector
        db = get_connector(None)
        response, _ = db.query([{"GetSchema": {}}])
        typer.echo("ApertureDB connection: OK")
    except Exception as e:
        typer.echo(f"ApertureDB connection failed: {e}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    detailed: bool = typer.Option(False, "--detailed"),
) -> None:
    """Show memory usage statistics."""
    try:
        from aperture_nexus.memory import Memory
        memory = Memory(config=config)
        result = memory.stats(scope="global" if detailed else "session")
        typer.echo(result)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the web UI and REST API (requires aperture-nexus[ui])."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        typer.echo(
            "The web UI requires aperture-nexus[ui]. "
            "Install it with: pip install aperture-nexus[ui]",
            err=True,
        )
        raise typer.Exit(1)

    if host != "127.0.0.1" and not os.environ.get("APERTURE_NEXUS_UI_API_KEY"):
        typer.echo(
            "Error: APERTURE_NEXUS_UI_API_KEY must be set when "
            "binding to a non-localhost address.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Starting aperture-nexus UI at http://{host}:{port}")
    import uvicorn
    uvicorn.run(
        "aperture_nexus.ui:app",
        host=host,
        port=port,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_env_key(env_path: Path, key: str, value: str) -> None:
    """Write or update a single key in a .env file."""
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n")


def _check_gitignore(env_path: Path) -> None:
    """Warn if .env is not in .gitignore."""
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return
    content = gitignore.read_text()
    if str(env_path) not in content and ".env" not in content:
        typer.echo(
            f"\nWarning: {env_path} is not in .gitignore. "
            "Add it to prevent accidental credential exposure:\n"
            f"    echo '{env_path}' >> .gitignore",
        )
