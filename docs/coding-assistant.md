---
title: Coding Assistant with Continuity
description: Text-only worked example. One principal, three sessions, continuity across days and tools.
sidebar_position: 5
---

# Coding Assistant with Continuity

Every coding agent has the same problem: it forgets. Every new session
starts cold. Every switch to a different tool starts cold. Every
colleague joining the project starts cold. Even yesterday's you had to
be re-explained.

This example walks through what a single developer's project looks
like across three sessions, on different days, in different Python
processes, under one shared Nexus principal. Everything is text; the
same pattern extends to different tools sharing that principal, and to
different content types too (see
[Customer Support Agent](customer-support-agent.md) for the same shape
when your agents also handle images and video).

---

## What This Example Shows

- **Continuity across sessions.** Design decisions from Monday surface
  in Tuesday's implementation session, even from a fresh Python
  process with no in-memory state.
- **Context is what makes retrieval meaningful.** Session name and
  purpose stamped at commit time. Retrieval combines vector similarity
  with that Context frame, so results are scoped to the task at hand
  and not just "closest in embedding space." Same idea as human
  episodic memory: you recall the specific interaction that fits the
  situation, not every superficially-similar thought you've ever had.
- **Aspirational but runnable today.** Three scripts, same principal,
  same Nexus. Copy them, run them in order, watch the graph fill in.
  No MCP server or cross-tool wrapper needed to prove the story; the
  Python API is enough.

---

## Setup

Three prerequisites:

1. A reachable ApertureDB (a local `docker compose up -d` stack works;
   set `APERTUREDB_KEY` or `APERTUREDB_JSON` for anything else).
2. CLIP installed so `process_and_commit` can embed text:
   `pip install "aperture-nexus[clip]"`.
3. An `aperture_nexus.json` config with an embedding model. Either run
   `adb-nexus init` once (interactive) or create the file inline as
   shown below.

The same principal owns all three sessions. Create it once via
`NexusAdmin`, save the API key, and each session authenticates with
it. The snippet below is idempotent so you can rerun it during
development.

```python
import json
import os
from aperture_nexus import NexusAdmin

# Local Docker Compose defaults so the example runs zero-config.
if not os.environ.get("APERTUREDB_KEY") and not os.environ.get("APERTUREDB_JSON"):
    os.environ["APERTUREDB_JSON"] = (
        '{"host":"localhost","port":55556,'
        '"username":"admin","password":"admin","use_ssl":false}'
    )

# Minimal aperture_nexus.json with CLIP so process_and_commit can embed.
# Skip this block if you already ran `adb-nexus init`.
CONFIG_PATH = "aperture_nexus.json"
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "models": {
                "text_embedding": "ViT-B/16",
                "image_embedding": "ViT-B/16",
                "video_embedding": "ViT-B/16",
            },
        }, f)

# In real deployments, the admin creates principals out of band and
# delivers keys through your normal credential channel; agents never
# see NexusAdmin. Idempotent recreate so this snippet is safe to rerun.
admin = NexusAdmin(config=CONFIG_PATH)
try:
    admin.delete_principal(user_id="alice")
except Exception:
    pass

api_key = admin.create_principal(
    user_id="alice",
    user_name="Alice",
    organization="acme",
    department="platform",
)
print(f"Principal 'alice' created. NEXUS_API_KEY={api_key}")
# Save api_key to a .env or credential store; every session below
# authenticates with it as `alice`.
```

Each session below assumes `NEXUS_API_KEY` is set in the environment
for `alice` and the `aperture_nexus.json` config from Setup is on
disk.

---

## Session 1: Designing a Rate Limiter (Monday)

Alice is thinking through a new feature. She writes down the design
decisions as she makes them, so future her (and future teammates) can
trace the reasoning.

```python
# session1_design.py
import os
from aperture_nexus import Context, Information, Memory

memory = Memory()
alice = memory.authenticate(user_id="alice", api_key=os.environ["NEXUS_API_KEY"])

ctx = Context(
    principal=alice,
    session_name="rate-limiter-design",
    purpose="Architect a per-user rate limiter for the API",
    organization="acme",
)

info = Information(context_id=ctx.id)
info.log(text=(
    "Chose token bucket over sliding window: refill semantics fit our "
    "peak-100-rps, avg-10-rps traffic. Sliding window over-throttles bursts."
))
info.log(text=(
    "Backing store: Redis. Keyed by user_id, TTL matches the rate window. "
    "Redis is already provisioned and horizontally scaled; no new infra."
))
info.log(text=(
    "Rate is per-user, NOT per-IP. Auth is user-based; we own the identity. "
    "Per-IP would penalize legit users behind NATs and CGNAT."
))

commit_id = memory.process_and_commit(ctx, info)
print(f"Session 1: 3 design decisions committed under 'rate-limiter-design' "
      f"({commit_id[:8]}...).")
```

Three decisions. Each carries `session_name="rate-limiter-design"` and
`purpose="Architect a per-user rate limiter for the API"`. That is the
Context frame future sessions will use to find these back.

---

## Session 2: Implementation, Next Day, Different Terminal

Tuesday morning. Alice opens a fresh terminal, a fresh Python process.
No in-memory state, no prior conversation loaded. She just needs to
remember what she decided yesterday.

```python
# session2_impl.py
import os
from aperture_nexus import Context, Information, Memory

memory = Memory()
alice = memory.authenticate(user_id="alice", api_key=os.environ["NEXUS_API_KEY"])

# What did we decide yesterday?
# In v1, semantic search filters by user_id, session_id, context_id.
# Broader filters (purpose, session_name, organization, department) work
# on metadata-only search today and are on the roadmap for semantic
# search once descriptor stamping catches up.
prior = memory.search(
    query="rate limiter design decisions",
    modality="text",
    filters={"user_id": "alice"},
    k=5,
)
print("Yesterday's decisions:")
for r in prior:
    print(f"  - {r.text}")

# Continue the work
ctx = Context(
    principal=alice,
    session_name="rate-limiter-impl",
    purpose="Implement the rate limiter from yesterday's design",
    organization="acme",
)

info = Information(context_id=ctx.id)
info.log(text=(
    "Implemented TokenBucket class. Atomic refill via Redis LUA script; "
    "avoids the race under parallel token acquisitions."
))
info.log(text=(
    "Tests: burst-below-capacity passes, burst-above-capacity throttles "
    "correctly, concurrent bursts across users are properly isolated."
))
info.log(text=(
    "Open issue: connection pool exhausts under 500 parallel token "
    "acquisitions. Adding backpressure via bounded semaphore around Redis."
))

commit_id = memory.process_and_commit(ctx, info)
print(f"Session 2: implementation notes committed under 'rate-limiter-impl' "
      f"({commit_id[:8]}...).")
```

The `filters` on the search scope retrieval to Alice's own memories.
Semantic similarity on top narrows the results to the specific rate
limiter thread. In real deployments you would combine this with
metadata attached at log time via `info.log(text=..., metadata={...})`;
descriptor-side stamping of Context.purpose and session_name to enable
those as first-class filters is a v1-to-v2 gap tracked separately.

---

## Session 3: Follow-Up Feature, Wednesday

A day later. New feature request: proper `Retry-After` header on
throttled responses. Alice needs the full picture, both the design and
the implementation notes.

```python
# session3_headers.py
import os
from aperture_nexus import Context, Information, Memory

memory = Memory()
alice = memory.authenticate(user_id="alice", api_key=os.environ["NEXUS_API_KEY"])

# Retrieve both prior threads via semantic search, scoped to Alice.
# Semantic similarity separates design-context matches from
# implementation-context matches.
design = memory.search(
    query="rate limiter architectural decisions",
    modality="text",
    filters={"user_id": "alice"},
    k=5,
)
impl = memory.search(
    query="rate limiter implementation and open issues",
    modality="text",
    filters={"user_id": "alice"},
    k=5,
)
print("Full context from prior work:")
for r in design + impl:
    print(f"  [{r.session_id[:8]}] {r.text}")

ctx = Context(
    principal=alice,
    session_name="rate-limiter-headers",
    purpose="Add Retry-After header derived from token bucket refill time",
    organization="acme",
)

info = Information(context_id=ctx.id)
info.log(text=(
    "Retry-After computed as (empty_tokens_needed / refill_rate). "
    "Derives directly from the token bucket state we already track."
))
info.log(text=(
    "Applied only on 429 responses. Header value is seconds, per RFC 7231. "
    "Clients get a precise, honest backoff hint instead of guessing."
))

commit_id = memory.process_and_commit(ctx, info)
print(f"Session 3: header logic committed under 'rate-limiter-headers' "
      f"({commit_id[:8]}...).")
```

Two searches, one scoped to the design session, the other to the
implementation session. Both return their respective decisions in the
same call. The agent now has architecture, implementation state, and
the new work all in mind.

---

## The Knowledge Graph Underneath

The reason the sessions above find the right memories is that Nexus
stores every commit in the same knowledge graph as the descriptors
used for vector search. Every committed entry carries `context_id`,
`session_id`, `user_id`, `commit_id`, `entry_id`, `created_at`, and
any `metadata` you attach at log time. Search combines vector
similarity with those properties as constraints in a single ApertureDB
query, so retrieval is scoped and semantic together. Broader
Context-side filters (`session_name`, `purpose`, `organization`,
`department`) work on metadata-only search today; extending descriptor
stamping so they also work in semantic search is a follow-up.

Lineage falls out of the same graph for free. Each memory is connected
to the `NexusCommit` that created it and the `NexusContext` that
authored it. You can trace any surfaced fact back to the specific
interaction it came from, which is what lets an audit answer "why did
the agent do X" and what lets deduplication answer "have I seen this
before."

---

## Scaling to a Team

Adding a colleague is one call. Bob joins the platform team; from that
moment his sessions authenticate as `bob` and his memories carry the
same `organization` and `department` scopes as Alice's:

```python
bob_key = admin.create_principal(
    user_id="bob",
    user_name="Bob",
    organization="acme",
    department="platform",
)
```

Bob searches for prior work with `filters={"organization": "acme"}`
and finds everything Alice contributed. Search scoped to `department`
narrows it to the platform team. Search scoped to `session_name` or
`purpose` gets the specific thread. The same principal model powers
per-user scopes when you need those too.

---

## Cleanup

Because sessions ran in different processes, their `Context` objects
are gone; iterate the session IDs from what Nexus retained instead.
`memory.remove(session_id=...)` cascades through content,
`NexusCommit`, `NexusContext`, and `NexusSession` in one call.

```python
# Collect every session Alice touched via a broad search
results = memory.search(filters={"user_id": "alice"}, k=500)
for sid in {r.session_id for r in results if r.session_id}:
    memory.remove(session_id=sid)

admin.delete_principal(user_id="alice")
admin.delete_principal(user_id="bob")
```

---

## Where This Goes Next

- **Different tools, same principal.** Nothing above depended on being
  Python. A Cursor extension, a Copilot integration, or an
  MCP-connected Claude Desktop session, all authenticating as `alice`,
  all writing and reading through the same Nexus, share exactly this
  continuity.
- **Different content, same store.** When Alice's work grows to include
  design diagrams, meeting recordings, or PDFs, the same commits
  handle them. Nothing about the pattern above changes;
  `info.log(image=...)` or `info.log(video=...)` sit alongside the
  text. See [Customer Support Agent](customer-support-agent.md) for
  the shape.
- **Different memory strategies.** Add `memory.connect()` to link
  Alice's rate-limiter contexts to a shared team-wide
  design-decisions context. Add tags at log time to group related
  work. Use `search_contexts()` to find related projects by their
  purposes, not their content.
