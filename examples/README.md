# Examples

Runnable examples showing how to use aperture-nexus. Each example is self-contained and can be run directly with `python`.

## Prerequisites

```bash
pip install aperture-nexus
adb-nexus init       # or set APERTUREDB_KEY in your environment
docker compose up -d # start ApertureDB
```

Or run `bash setup.sh` from the repo root to do all of the above in one step.

---

## Examples

| File | What it shows |
|------|--------------|
| [`quickstart.py`](quickstart.py) | Text storage and search — the simplest possible end-to-end flow |
| [`text.py`](text.py) | Text in depth: chunking, search, metadata filtering |
| [`images.py`](images.py) | Images: file paths, URLs, PIL, numpy arrays, pre-computed embeddings |
| [`video.py`](video.py) | Video: clip extraction, frame sampling, scene detection |
| [`blobs.py`](blobs.py) | Blobs: PDFs, audio files, arbitrary binary data |
| [`multi_user_session.py`](multi_user_session.py) | Multiple participants sharing a session |
| [`async_processing.py`](async_processing.py) | Non-blocking `async_process_and_commit()` with MemoryTask |

---

## Running an example

```bash
python examples/quickstart.py
python examples/images.py
```

Each example prints what it is doing and what it finds. They create real data in ApertureDB — inspect it in the ApertureDB web UI at `http://localhost:8087`.
