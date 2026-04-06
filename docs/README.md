---
title: aperture-nexus Documentation
description: The cognition engine for enterprise AI — establish context, capture multimodal knowledge, and commit to memory for retrieval.
---

# aperture-nexus Documentation

**The Cognition Engine for Enterprise AI.**

aperture-nexus enables enterprise AI agents to establish relations for
context, capture knowledge across text, images, video, and documents,
and commit it to memory for search and retrieval — powered by
[ApertureDB](https://aperturedata.io)'s vector search and knowledge
graph.

The three building blocks — **Knowledge** (what was captured),
**Memory** (the engine that stores and retrieves it), and **Context**
(who did what, in which session, and why) — together form a complete
cognition layer that scales from a single developer session to a
multi-team enterprise deployment.

---

## Pages

| Page | What it covers |
|------|----------------|
| [Getting Started](getting-started.md) | Step-by-step from zero to your first stored memory — **start here** |
| [Concepts](concepts.md) | The KMC model, how the three core objects relate, ApertureDB storage mapping, and architecture diagrams |
| [API Reference](api-reference.md) | Full reference for `Memory`, `Context`, `Information`, `MemoryTask`, and the exception hierarchy |
| [Configuration](configuration.md) | Every field in `aperture_nexus.json` with defaults, constraints, and environment variable overrides |

For a one-command setup, run `bash setup.sh` from the repo root.
See [`examples/`](https://github.com/aperturedata/aperture-nexus/tree/main/examples) for runnable scripts covering each data modality.
For installation and CLI reference, see the [main README](https://github.com/aperturedata/aperture-nexus#readme).

---

## Quick Orientation

aperture-nexus is built around three objects that work together:

```
Context    — who is doing what, in which session, and why
Information — multimodal inputs (text, images, video, blobs) buffered during a session
Memory     — the engine that commits, processes, connects, and searches
```

The typical call sequence is:

```python
from aperture_nexus import Memory, Context, Information

memory = Memory()
principal = memory.authenticate(user_id="alice", api_key="...")

ctx = Context(principal=principal, session_name="my-session", purpose="...")
info = Information(context_id=ctx.id)
info.log(text="...")
info.log(image="photo.jpg")

memory.commit(ctx, info)          # raw storage — fast
# or
memory.process_and_commit(ctx, info)  # embeddings + summarization

results = memory.search(query="...", filters={"session_id": ctx.session_id})
```

See [Concepts](concepts.md) to understand the full data model, or [API Reference](api-reference.md) for method signatures and parameters.

---

## GitHub Pages

To serve these docs as a website from GitHub Pages, point GitHub Pages at the `/docs` folder on `main`. Add a `.nojekyll` file at the repository root to disable Jekyll processing (plain Markdown and Mermaid diagrams render correctly without it):

```bash
touch .nojekyll
```

No build step is required.
