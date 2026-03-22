# aperture-nexus Configuration Reference

Generate your config file with:
```bash
adb-nexus init
```

This creates `aperture_nexus.json` in your current directory, along with this reference file. The config file is discovered automatically — see [Config Discovery](#config-discovery) for the full priority order.

---

## Config Discovery

aperture-nexus looks for your config file in this order (first found wins):

1. Path passed explicitly: `Memory(config="path/to/config.json")`
2. `APERTURE_NEXUS_CONFIG` environment variable
3. `./aperture_nexus.json` (current working directory)
4. `~/.aperture_nexus/config.json` (user home — works on local, VM, and container)
5. Built-in defaults (no models — `process_and_commit()` will raise `NexusConfigError`)

No root access is required for any of these paths.

---

## Full Example

```json
{
    "models": {
        "llm": "gpt-4o",
        "vlm": "clip-vit-base-patch32"
    },
    "processing": {
        "num_threads": 4,
        "batch_size": 50,
        "embedding_batch": 32,
        "retry_attempts": 3,
        "retry_interval": 1.0,
        "text_chunk_size": 2000,
        "text_chunk_overlap": 200,
        "text_chunk_unit": "characters",
        "video_clip_duration": 10.0,
        "video_clip_overlap": 0.5,
        "video_frame_interval": 30,
        "video_scene_detection": false,
        "video_max_frames": 100
    },
    "logging": {
        "level": "ERROR"
    },
    "metrics": {
        "enabled": false,
        "port": 8000,
        "path": "/metrics"
    },
    "ui": {
        "enabled": false,
        "host": "127.0.0.1",
        "port": 8000,
        "api_key": null
    }
}
```

---

## `models`

Required only if `process_and_commit()` or `async_process_and_commit()` is used.
If you only use `commit()` (raw storage), you can omit this section entirely.

| Field | Type   | Description |
|-------|--------|-------------|
| `llm` | string | Language model for text processing and summarization. Examples: `"gpt-4o"`, `"gemini-2.5-pro"` |
| `vlm` | string | Vision-language model for image and video embedding. Examples: `"clip-vit-base-patch32"`, `"gemini-embedding-001"` |

---

## `processing`

Controls how aperture-nexus processes and stores multimodal data in ApertureDB.

### General

| Field | Default | Description |
|-------|---------|-------------|
| `num_threads` | `4` | Parallel threads for ApertureDB writes. Increase for high-throughput workloads. |
| `batch_size` | `50` | Items per batch sent to ApertureDB. Tune based on item size. |
| `embedding_batch` | `32` | Items per model inference call for embedding generation. |
| `retry_attempts` | `3` | Retry attempts for failed DB writes before raising `NexusStorageError`. Set to `0` to disable retries. |
| `retry_interval` | `1.0` | Seconds between retry attempts. |

### Text Chunking

Long text is automatically chunked before embedding. Chunk size should be tuned to your embedding model's context window.

| Field | Default | Description |
|-------|---------|-------------|
| `text_chunk_size` | `2000` | Size of each chunk. Unit set by `text_chunk_unit`. |
| `text_chunk_overlap` | `200` | Overlap between consecutive chunks. Must be less than `text_chunk_size`. |
| `text_chunk_unit` | `"characters"` | `"characters"` (no extra deps) or `"tokens"` (requires `pip install aperture-nexus[tokens]`). |

### Video Processing

Videos are stored in ApertureDB as `Video` → `Clip` → `Descriptor` (embedding). Clip duration should match your embedding model's expected input length.

| Field | Default | Description |
|-------|---------|-------------|
| `video_clip_duration` | `10.0` | Duration of each clip in seconds. Adjust to match your VLM's expected input (e.g. shorter for action recognition models). |
| `video_clip_overlap` | `0.5` | Overlap between consecutive clips in seconds. |
| `video_frame_interval` | `30` | Extract one frame every N frames (when `video_scene_detection` is `false`). At 30fps this is ~1 frame/sec. |
| `video_scene_detection` | `false` | Extract frames at scene boundaries. More accurate but requires `pip install aperture-nexus[video]`. |
| `video_max_frames` | `100` | Hard cap on frames extracted per video, regardless of method. |

---

## `logging`

Controls aperture-nexus internal logging only. Does not affect your application's logging setup.

To override without editing the file:
```bash
APERTURE_NEXUS_LOG_LEVEL=DEBUG python my_script.py
```

| Field | Default | Options |
|-------|---------|---------|
| `level` | `"ERROR"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

---

## `metrics`

Exposes a Prometheus-compatible `/metrics` endpoint for integration with Grafana, Datadog, and similar tools.

**Requires:** `pip install aperture-nexus[metrics]`

If `enabled` is `true` but the package is not installed, `NexusConfigError` is raised at startup with install instructions.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable metrics export. |
| `port` | `8000` | Port for the metrics endpoint. Must be > 1024 (no root required). |
| `path` | `"/metrics"` | HTTP path for the metrics endpoint. |

**Available metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `aperture_nexus_commits_total` | Counter | Total commit calls |
| `aperture_nexus_commits_failed_total` | Counter | Failed commit calls |
| `aperture_nexus_processing_items_total` | Counter | Items processed, by `modality` label |
| `aperture_nexus_commit_latency_ms` | Histogram | Commit latency in milliseconds |
| `aperture_nexus_embed_latency_ms` | Histogram | Embedding generation latency |
| `aperture_nexus_tasks_pending` | Gauge | MemoryTasks currently in flight |
| `aperture_nexus_tasks_failed` | Gauge | MemoryTasks in failed state |

---

## `ui`

Web UI and REST API for browsing sessions, searching memory, and monitoring tasks.

**Requires:** `pip install aperture-nexus[ui]`

**Security:** When `host` is not `127.0.0.1`, `api_key` MUST be set. aperture-nexus will raise `NexusConfigError` at startup if this rule is violated. Never put the API key value directly in `aperture_nexus.json` — use the environment variable instead:

```bash
export APERTURE_NEXUS_UI_API_KEY="your-secret-key"
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable the web UI and REST API. |
| `host` | `"127.0.0.1"` | Bind address. `127.0.0.1` = local only. `0.0.0.0` = all interfaces (requires `api_key`). |
| `port` | `8000` | Port to serve on. Must be > 1024 (no root required). |
| `api_key` | `null` | Required for non-local deployments. Use `APERTURE_NEXUS_UI_API_KEY` env var — never hardcode. |

**Available endpoints (all under `/api/v1/`):**

| Path | Description |
|------|-------------|
| `/` | Dashboard |
| `/sessions` | Browse and filter sessions |
| `/sessions/{id}` | Session detail with participants |
| `/memory` | Browse committed memories |
| `/search` | Multimodal search UI |
| `/tasks` | MemoryTask status, retry |
| `/settings` | View config (read-only in UI) |
| `/docs` | Auto-generated API documentation |
| `/metrics` | Metrics summary (if enabled) |

---

## Environment Variable Overrides

Any config value can be overridden with an environment variable without editing the file:

| Variable | Overrides |
|----------|-----------|
| `APERTURE_NEXUS_CONFIG` | Config file path |
| `APERTURE_NEXUS_LOG_LEVEL` | `logging.level` |
| `APERTURE_NEXUS_UI_API_KEY` | `ui.api_key` |
| `APERTUREDB_KEY` | ApertureDB connection (encoded key) |
| `APERTUREDB_HOST` | ApertureDB host |
| `APERTUREDB_PORT` | ApertureDB port |
| `APERTUREDB_USER` | ApertureDB username |
| `APERTUREDB_PASSWORD` | ApertureDB password |

---

## IDE Autocomplete

The companion `aperture_nexus.schema.json` file enables autocomplete and inline validation in VS Code and other editors that support JSON Schema. To activate in VS Code, add to your `.vscode/settings.json`:

```json
{
    "json.schemas": [
        {
            "fileMatch": ["aperture_nexus.json"],
            "url": "./aperture_nexus.schema.json"
        }
    ]
}
```
