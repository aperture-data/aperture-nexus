---
title: Getting Started
description: From zero to your first stored and searchable memory
sidebar_position: 0
---

# Getting Started

This guide takes you from a fresh machine to storing and searching
your first multimodal memory. It takes about ten minutes.

Want to see it in action first? Run the interactive demo — no setup
needed:

```bash
git clone https://github.com/aperturedata/aperture-nexus
cd aperture-nexus
docker compose --profile demo run --rm nexus-demo
```

---

## Prerequisites

- Python 3.10 or later
- Docker — [install Docker](https://docs.docker.com/get-docker/)

---

## 1. Install aperture-nexus

aperture-nexus is not yet on PyPI. Clone the repository and install
directly:

```bash
git clone https://github.com/aperturedata/aperture-nexus
cd aperture-nexus
pip install .
```

To also enable video processing or token-based chunking, use extras:

```bash
pip install ".[video]"
pip install ".[tokens]"
pip install ".[all]"      # everything
```

---

## 2. Start ApertureDB

aperture-nexus stores everything in
[ApertureDB](https://aperturedata.io). Choose the option that fits
your situation:

**Local — Docker Compose (quickest):**

```bash
docker compose up -d
```

This starts ApertureDB, the Lenz gateway, and the web UI at
`http://localhost:8087`.

**Existing or cloud ApertureDB instance:**

Set your connection via environment variable and skip the Docker step:

```bash
# Encoded key (recommended for production — from `adb config`)
export APERTUREDB_KEY="adbp_..."

# Or JSON config for a remote instance
export APERTUREDB_JSON='{"host":"your-host","port":55556,"username":"...","password":"...","use_ssl":true}'
```

See the [ApertureDB documentation](https://docs.aperturedata.io) for
connection options.

---

## 3. Create a Principal

aperture-nexus uses a two-tier identity model. An **admin** creates
**principals** — one per user or agent. Principals authenticate with
an `api_key` that the admin issues and can rotate without disrupting
stored data.

Run the setup wizard once to create your principal and write the key
to `.env`:

```bash
adb-nexus init
```

For non-interactive setup:

```bash
adb-nexus init --defaults
```

In production, principals are created programmatically via
`NexusAdmin`:

```python
from aperture_nexus import NexusAdmin

admin = NexusAdmin()  # requires admin ApertureDB credentials
api_key = admin.create_principal(
    user_id="alice",
    user_name="Alice Chen",
    organization="AcmeCorp",
    department="support",
)
# Deliver api_key to alice; store it securely
```

---

## 4. Verify Your Setup

```bash
adb-nexus validate
```

If everything is working you will see `Config: OK` and
`ApertureDB connection: OK`.

---

## 5. Your First Memory

```python
import os
from dotenv import load_dotenv
from aperture_nexus import Memory, Context, Information

load_dotenv()  # loads NEXUS_API_KEY written by adb-nexus init

memory = Memory()
principal = memory.authenticate(
    user_id="alice",
    api_key=os.environ["NEXUS_API_KEY"],
)

ctx = Context(
    principal=principal,
    session_name="my-first-session",
    purpose="Testing aperture-nexus",
)

info = Information(context_id=ctx.id)
info.log(text="aperture-nexus stores text, images, audio, video, and more")
info.log(text="Each entry is linked to a context and session")

# Commit to ApertureDB (raw storage — fast, no model calls)
ctx_id = memory.commit(ctx, info)
print("Memory committed.")

# Search by metadata filter — no embedding model needed.
# Filter by session_name (human-readable) or session_id (precise).
# k controls the maximum number of results returned (default: 10).
results = memory.search(filters={"session_name": "my-first-session"}, k=10)
for r in results:
    print(r.text)
```

Or run the included quickstart script:

```bash
python examples/quickstart.py
```

---

## 6. Log Multimodal Content

`Information.log()` accepts any combination of modalities in a single
entry:

```python
# Text
info.log(text="Customer reported a timeout on the /export endpoint")

# Image — file path, URL, PIL Image, or numpy array
info.log(image="screenshot.png")
info.log(image="https://example.com/diagram.png")

# Audio or other binary content — pass as blob with a document_type
info.log(blob=audio_bytes, document_type="mp3")
info.log(blob=pdf_bytes,   document_type="pdf")

# Video
info.log(video="recording.mp4")

# Mixed — one entry, multiple modalities
info.log(text="See attached screenshot", image="screenshot.png")
```

To generate embeddings and enable semantic similarity search, use
`process_and_commit()` instead of `commit()`. This requires a model
configured in `aperture_nexus.json` — see
[Configuration](configuration.md).

---

## 7. Multi-Participant Sessions

Multiple agents, workflows, or humans can contribute to the same
session. Each gets their own `Context`:

```python
from aperture_nexus import generate_session_id

sid = generate_session_id()

ctx_user  = Context(
    principal=user_principal, session_id=sid, purpose="Question"
)
ctx_agent = Context(
    principal=agent_principal, session_id=sid, purpose="Response"
)

info_user = Information(context_id=ctx_user.id)
info_user.log(text="What happened to order #4821?")

info_agent = Information(context_id=ctx_agent.id)
info_agent.log(text="Order #4821 was shipped on Monday, arriving Thursday.")

memory.commit(ctx_user,  info_user)
memory.commit(ctx_agent, info_agent)

# Search returns contributions from all participants
results = memory.search(filters={"session_id": sid})
```

---

## What's Next

- [Concepts](concepts.md) — the full KMC model and storage mapping
- [API Reference](api-reference.md) — all parameters and return types
- [Configuration](configuration.md) — add models and tune processing
- [`examples/`](https://github.com/aperturedata/aperture-nexus/tree/main/examples)
  — runnable scripts for each modality
