---
title: Getting Started
description: From zero to your first stored and searchable memory in under ten minutes
sidebar_position: 0
---

# Getting Started

This guide takes you from a fresh machine to storing and searching your first multimodal memory. It takes about ten minutes.

---

## Prerequisites

- Python 3.10 or later
- Docker (for ApertureDB) — [install Docker](https://docs.docker.com/get-docker/)

---

## 1. Install aperture-nexus

```bash
pip install aperture-nexus
```

To also enable video processing, token-based chunking, or the web UI, use extras:

```bash
pip install aperture-nexus[video,ui]
```

---

## 2. Start ApertureDB

The fastest way is Docker Compose. From the aperture-nexus directory:

```bash
docker compose up -d
```

Or use the setup script (handles everything, including waiting for ApertureDB to be ready):

```bash
bash setup.sh
```

ApertureDB is ready when you see it accept connections — `adb-nexus validate` will confirm this in step 4.

> **Mac users:** If port 55555 conflicts with AirPlay or another service, start on a different port:
> ```bash
> APERTUREDB_PORT=15555 docker compose up -d
> ```
> Then add `APERTUREDB_PORT=15555` to your `.env` file.

---

## 3. Configure credentials

The simplest option is an encoded ApertureDB key:

```bash
export APERTUREDB_KEY="adbp_..."
```

Or create a `.env` file (copy from the example and edit):

```bash
cp .env.example .env
# Edit .env with your ApertureDB host, port, user, and password
```

For the local Docker Compose setup, the default credentials work without any changes.

Then run the setup wizard to generate your config file:

```bash
adb-nexus init
```

The wizard walks you through connection settings and optional model configuration. For a quick start, accept all defaults — you can add model configuration later when you want to use `process_and_commit()`.

---

## 4. Verify your setup

```bash
adb-nexus validate
```

If everything is working, you will see a confirmation that ApertureDB is reachable and your config is valid. If something is wrong, the error message will tell you what to fix.

---

## 5. Your first memory

Run the quickstart example:

```bash
python examples/quickstart.py
```

Or write your own:

```python
from aperture_nexus import Memory, Context, Information

# Connect and authenticate
memory = Memory()
principal = memory.authenticate(user_id="alice", api_key="your-api-key")

# Describe the session
ctx = Context(
    principal=principal,
    session_name="my-first-session",
    purpose="Testing aperture-nexus",
)

# Log some information
info = Information(context_id=ctx.id)
info.log(text="aperture-nexus stores text, images, video, and more")
info.log(text="Each piece of information is linked to a context and session")

# Commit to ApertureDB (raw storage — fast, no model calls)
memory.commit(ctx, info)
print("Memory committed.")

# Search
results = memory.search(
    query="aperture-nexus",
    filters={"session_id": ctx.session_id},
)
for r in results:
    print(r)
```

---

## 6. Add an image

`Information.log()` accepts images in many forms — file path, URL, PIL Image, or numpy array:

```python
info.log(image="photo.jpg")
info.log(image="https://example.com/photo.jpg")
```

To generate embeddings and enable image similarity search, call `process_and_commit()` instead of `commit()`. This requires a vision-language model configured in `aperture_nexus.json` (set `models.vlm`).

---

## 7. Multi-participant sessions

Multiple users or agents can participate in the same session. Each gets their own `Context`:

```python
sid = memory.generate_session_id()

ctx_user  = Context(principal=user_principal,  session_id=sid, purpose="Question")
ctx_agent = Context(principal=agent_principal, session_id=sid, purpose="Response")

info_user = Information(context_id=ctx_user.id)
info_user.log(text="What happened to order #4821?")

info_agent = Information(context_id=ctx_agent.id)
info_agent.log(text="Order #4821 was shipped on Monday, arriving Thursday.")

memory.commit(ctx_user, info_user)
memory.commit(ctx_agent, info_agent)

# Search returns contributions from all participants
results = memory.search(query="order 4821", filters={"session_id": sid})
```

---

## What's next

- See [Concepts](concepts.md) for the full KMC model and ApertureDB storage mapping
- See [API Reference](api-reference.md) for all parameters and return types
- See [Configuration](configuration.md) to add models and tune processing
- Browse `examples/` for runnable scripts covering each data modality
