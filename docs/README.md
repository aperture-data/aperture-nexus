---
title: aperture-nexus Documentation
description: The cognition engine for enterprise AI.
---

# aperture-nexus

**The Cognition Engine for Enterprise AI.**

aperture-nexus enables AI workflows, agents, and the humans working
alongside them to establish context, capture knowledge across text,
images, audio, video, and more — and commit it to memory for search
and retrieval, powered by [ApertureDB](https://aperturedata.io)'s
vector search and knowledge graph.

![aperture-nexus hello world](https://raw.githubusercontent.com/aperturedata/aperture-nexus/main/demo/demo.gif)

---

## How It Works

The KMC model mirrors how human memory works and, in turn, how
enterprises can design AI systems with cognition. **Knowledge** is
semantic: the general facts and relationships that don't change
moment to moment, held as a shared baseline in ApertureDB. **Memory**
is episodic: the specific trace of what happened in a particular
interaction, built up over time from new `Information` committed via
Nexus. **Context** — who, what, when, why, and how — is stamped on
every memory so retrieval is meaningful rather than merely a lookup.

```mermaid
flowchart LR
    I["Information\nlocal Nexus buffer\ntext · image · video · blob"]
    C["Context (C)\nwho · what · when · why · how"]
    M["Memory (M)\nepisodic — accumulated\nmemories with connections"]
    K["Knowledge (K)\nsemantic — shared baseline"]

    C -->|"stamps every memory"| M
    I -->|"commit()"| M
    M <-.->|"searched together"| K
    M -.->|"consolidated / discarded over time"| K
```

| KMC | Concept | In code / storage |
|-----|---------|-------------------|
| **K** | Semantic memory: general facts and relationships that don't change moment to moment | ApertureDB baseline; read via `Memory` |
| **M** | Episodic memory: the trace of what happened in a particular interaction | `Memory` (the class is the interface to the store); `Information` is the staging buffer |
| **C** | Who, what, when, why, and how — the retrieval frame | `Context` |

The same model works for a single developer session, a multi-agent
pipeline, or a human+AI team sharing context across an enterprise.

Together this enables cognition, with hooks to surface, reason,
update, and discard as understanding evolves.

---

## Quick Start

Try the interactive walkthrough in one command — no setup needed:

```bash
git clone https://github.com/aperturedata/aperture-nexus
cd aperture-nexus
docker compose --profile demo run --rm nexus-demo
```

Or jump straight to [Getting Started](getting-started.md) to build
your own integration.

---

## Pages

| Page | What it covers |
|------|----------------|
| [Concepts](concepts.md) | KMC model, core objects, sessions, storage mapping |
| [Getting Started](getting-started.md) | Step-by-step to your first stored memory |
| [API Reference](api-reference.md) | `Memory`, `Context`, `Information`, `MemoryTask` |
| [Configuration](configuration.md) | Every field in `aperture_nexus.json` |
| [Customer Support Agent](customer-support-agent.md) | Multi-agent pipeline with multimodal memory and semantic image search |

See [`examples/`](https://github.com/aperturedata/aperture-nexus/tree/main/examples)
for runnable scripts covering each data modality.
