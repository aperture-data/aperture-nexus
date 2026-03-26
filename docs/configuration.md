---
title: Configuration Reference
description: Every field in aperture_nexus.json with defaults, constraints, and environment variable overrides
sidebar_position: 3
---

# Configuration Reference

Generate your config file with:

```bash
adb-nexus init
```

This creates `aperture_nexus.json` in your current directory. The config file is optional — aperture-nexus works with built-in defaults if no file is found, but you must supply model names to use `process_and_commit()`.

---

## Config Discovery

aperture-nexus looks for your config file in this order (first found wins):

1. Explicit path: `Memory(config="path/to/config.json")`
2. `APERTURE_NEXUS_CONFIG` environment variable
3. `./aperture_nexus.json` (current working directory)
4. `~/.aperture_nexus/config.json` (user home — works on local, VM, and container)
5. Built-in defaults (no models — `process_and_commit()` raises `NexusConfigError`)

No root access is required for any of these paths.

---

## Full Example

```json
{
    "models": {
        "llm": "gpt-4o",
        "text_embedding": "text-embedding-3-small",
        "image_embedding": "clip-vit-base-patch32",
        "video_embedding": "clip-vit-base-patch32"
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
        "video_frames_per_clip": 10,
        "video_max_clips": 50
    },
    "logging": {
        "level": "ERROR"
    },
    "metrics": {
        "enabled": false,
        "port": 8001,
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

Required only if `process_and_commit()` or `async_process_and_commit()`
is used. If you only use `commit()` (raw storage), omit this section.

Each modality has its own embedding model and its own ApertureDB
DescriptorSet (`nexus_text`, `nexus_image`, `nexus_video`). Search is
per-modality: a text query searches text descriptors only, an image
query searches image descriptors only.

| Field | Type | Description |
|-------|------|-------------|
| `llm` | string | Language model for summarization and text enrichment. Examples: `"gpt-4o"`, `"gemini-2.5-pro"` |
| `text_embedding` | string | Embedding model for text. Examples: `"text-embedding-3-small"`, `"text-embedding-ada-002"` |
| `image_embedding` | string | Embedding model for images. Examples: `"clip-vit-base-patch32"`, `"openclip-vit-l-14"` |
| `video_embedding` | string | Embedding model for video frames. Can be the same as `image_embedding`. |

> **Note:** `text_embedding`, `image_embedding`, and `video_embedding`
> must each match the model used at index time (stored on the
> DescriptorSet). Mismatches are caught at search time with a
> `NexusConfigError`.

---

## `processing`

Controls how aperture-nexus processes and stores multimodal data.

### General

| Field | Default | Description |
|-------|---------|-------------|
| `num_threads` | `4` | Parallel threads for ApertureDB writes. Range: 1–32. |
| `batch_size` | `50` | Items per batch sent to ApertureDB. Tune based on item size. |
| `embedding_batch` | `32` | Items per model inference call for embedding generation. |
| `retry_attempts` | `3` | Retry attempts for failed DB writes before raising `NexusStorageError`. Set to `0` to disable. |
| `retry_interval` | `1.0` | Seconds between retry attempts. |

### Text chunking

Long text is chunked automatically before embedding. Tune chunk size to your embedding model's context window.

| Field | Default | Description |
|-------|---------|-------------|
| `text_chunk_size` | `2000` | Size of each chunk. Unit controlled by `text_chunk_unit`. |
| `text_chunk_overlap` | `200` | Overlap between consecutive chunks. Must be less than `text_chunk_size`. |
| `text_chunk_unit` | `"characters"` | `"characters"` (default, no extra deps) or `"tokens"` (requires `pip install aperture-nexus[tokens]`). |

### Video processing

Videos are split into clip segments. Each segment gets one embedding (mean of CLIP frame embeddings within that segment) stored as one `Descriptor` in ApertureDB with `start_frame`/`stop_frame` metadata. This preserves temporal structure: a text query finds the specific segment of a video that matches, not just whether the video matches at all.

| Field | Default | Description |
|-------|---------|-------------|
| `video_clip_duration` | `10.0` | Target clip duration in seconds. Informational — actual duration is determined by `video_frame_interval × video_frames_per_clip / fps`. |
| `video_clip_overlap` | `0.5` | Overlap between consecutive clips in seconds. |
| `video_frame_interval` | `30` | Sample one frame every N frames. At 30fps: ~1 frame/sec. |
| `video_scene_detection` | `false` | Use scene boundaries instead of fixed intervals. Requires `pip install aperture-nexus[video]`. |
| `video_frames_per_clip` | `10` | Number of sampled frames to group into one clip segment. At `frame_interval=30` and `frames_per_clip=10`, each clip covers ~300 original frames (~10s at 30fps). |
| `video_max_clips` | `50` | Hard cap on the number of clip segments (and Descriptors) produced per video. |

---

## `logging`

Controls aperture-nexus internal logging only — does not affect your application's logging.

| Field | Default | Options |
|-------|---------|---------|
| `level` | `"ERROR"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

Override without editing the file:

```bash
APERTURE_NEXUS_LOG_LEVEL=DEBUG python my_script.py
```

---

## `metrics`

Prometheus-compatible metrics export. Requires `pip install aperture-nexus[metrics]`.

If `enabled` is `true` but the package is not installed, `NexusConfigError` is raised at startup with the install command.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable metrics export. |
| `port` | `8001` | Port for the `/metrics` endpoint. Must be > 1024. |
| `path` | `"/metrics"` | HTTP path for the metrics endpoint. |

**Available metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `aperture_nexus_commits_total` | Counter | Total commit calls |
| `aperture_nexus_commits_failed_total` | Counter | Failed commit calls |
| `aperture_nexus_processing_items_total` | Counter | Items processed, labelled by `modality` |
| `aperture_nexus_commit_latency_ms` | Histogram | Commit latency in milliseconds |
| `aperture_nexus_embed_latency_ms` | Histogram | Embedding generation latency |
| `aperture_nexus_tasks_pending` | Gauge | Async MemoryTasks currently in flight |
| `aperture_nexus_tasks_failed` | Gauge | MemoryTasks in failed state |

---

## `ui`

Web UI and REST API for browsing sessions, searching memory, and monitoring tasks. Requires `pip install aperture-nexus[ui]`.

**Security rule:** When `host` is not `127.0.0.1`, `api_key` MUST be set. aperture-nexus raises `NexusConfigError` at startup if this is violated. Set the key via environment variable — never put it in `aperture_nexus.json`:

```bash
export APERTURE_NEXUS_UI_API_KEY="your-secret-key"
adb-nexus ui --host 0.0.0.0 --port 8080
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable the web UI and REST API. |
| `host` | `"127.0.0.1"` | Bind address. `127.0.0.1` = local only. `0.0.0.0` = all interfaces (requires `api_key`). |
| `port` | `8000` | Port to serve on. Must be > 1024. |
| `api_key` | `null` | Required for non-local deployments. Use `APERTURE_NEXUS_UI_API_KEY` — never hardcode. |

---

## Environment variable overrides

Environment variables always take precedence over the config file.

| Variable | Overrides |
|----------|-----------|
| `APERTURE_NEXUS_CONFIG` | Config file path |
| `APERTURE_NEXUS_LOG_LEVEL` | `logging.level` |
| `APERTURE_NEXUS_UI_API_KEY` | `ui.api_key` |
| `APERTUREDB_KEY` | ApertureDB connection (encoded key — takes priority over all other DB vars) |
| `APERTUREDB_HOST` | ApertureDB host |
| `APERTUREDB_PORT` | ApertureDB port |
| `APERTUREDB_USER` | ApertureDB username |
| `APERTUREDB_PASSWORD` | ApertureDB password |

---

## IDE autocomplete

`adb-nexus init` also generates `aperture_nexus.schema.json`. To activate JSON Schema validation and autocomplete in VS Code, add to `.vscode/settings.json`:

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
