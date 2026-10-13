"""
ingest_images.py — load the aperture-nexus architecture diagrams into Nexus.

For each diagram we store two entries under the same tag:
  - a TEXT entry (the caption) → BGE text embedding, searchable via BGE
  - an IMAGE entry (the actual PNG) → CLIP image embedding

Both land in the same session/context so a graph traversal can connect
"caption that matched" ↔ "image that goes with it". When the webapp's
text search surfaces a caption, it also looks up the paired image via
the shared ``figure_id`` metadata and renders the diagram in the result
card.

Usage (from aperture-nexus/):
    python demo/webapp/ingest_images.py              # ingest once
    python demo/webapp/ingest_images.py --reset      # wipe + re-ingest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "APERTUREDB_JSON",
    '{"host":"localhost","port":55556,"username":"admin","password":"admin","use_ssl":false}',
)

from aperture_nexus import Context, Information, Memory  # noqa: E402

LOG = logging.getLogger("nexus-demo.ingest_images")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = Path(__file__).resolve().parent

DEMO_WORKSPACE = "nexus-demo"
DEMO_USER = "nexus-demo-user"
DEMO_SESSION = "aperture-nexus-codebase"
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
IMAGE_MODEL = "ViT-B/16"  # CLIP for image embeddings
API_KEY_FILE = WEBAPP_DIR / ".nexus-demo.key"

# figure_id → (rel path from repo root, caption text).
# Captions are deliberately rich — the better the words, the better BGE
# finds the caption when a question is tangentially related.
DIAGRAMS: list[tuple[str, str, str]] = [
    (
        "nexus_components",
        "docs/nexus_components.png",
        "Architecture diagram: Aperture Nexus Python API components, "
        "building blocks, and the overall architecture at a glance. "
        "Shows the six main components and their roles: "
        "NexusAdmin (identity authority, issues API keys), "
        "Principal (authenticated user identity), "
        "Session (shared thread of work across participants), "
        "Context (per-participant metadata including purpose and tags), "
        "Information (multimodal buffer for text, images, video, documents), "
        "and Memory (the engine that writes to ApertureDB and runs semantic "
        "and metadata search). Each card lists the key methods or fields "
        "for that component. The whole architecture is backed by ApertureDB — "
        "knowledge graph, vector search, multimodal storage.",
    ),
    (
        "nexus_workflow",
        "docs/nexus_workflow.png",
        "Workflow diagram: Nexus write and query data flow architecture. "
        "WRITE path: info.log(...) populates a local buffer, then one of "
        "three commit methods — Memory.commit() for raw storage, "
        "Memory.process_and_commit() for synchronous embed-and-store, "
        "or Memory.async_process_and_commit() for background processing. "
        "All three write to ApertureDB. "
        "QUERY path: an agent or app can retrieve either through "
        "Memory.search() (the Nexus semantic + metadata search layer) "
        "or by querying ApertureDB directly for structural traversal. "
        "Both paths return results to the agent.",
    ),
    (
        "cognition_stack",
        "docs/blog_image_cognition_stack_v2.png",
        "Architecture diagram: Building AI Cognition — the three-layer stack. "
        "Top: Reasoning and Cognition Layer — models, agents, humans — "
        "who interpret context, reason, answer questions, act, and initiate "
        "memory updates. Middle: Memory Infrastructure and Cognition Hooks — "
        "Aperture Nexus, open source and actively growing — turns commits "
        "into memory and keeps context attached to every one. Bottom: "
        "Unified Data Foundation — ApertureDB, production-ready, "
        "battle-tested — knowledge base (knowledge graph), vector search, "
        "multimodal storage. Arrows between the layers: retrieve, reason "
        "over, surface going up; update, enrich, drop going down.",
    ),
    (
        "nexus_schema",
        "demo/nexus_schema.png",
        "Schema diagram: the ApertureDB entity schema for Nexus memory, "
        "showing the storage architecture and relationships. "
        "Shows how Nexus's concepts map to ApertureDB primitives: "
        "NexusUser (one per principal), NexusSession (one per shared "
        "session, with session_id and session_name), NexusContext (per "
        "participant, with purpose and metadata), NexusCommit (one per "
        "commit() call, grouping all entries written together), and the "
        "stored content entities Blob, Image, Video, plus DescriptorSet "
        "and Descriptor for vector search. Edges between them: "
        "nexus_user_context, nexus_session_context, nexus_context_commit, "
        "nexus_commit_entry, nexus_descriptor_entry.",
    ),
]


def _memory_for_demo_user() -> Memory:
    if not API_KEY_FILE.exists():
        LOG.error("Run ingest.py first to provision the demo principal.")
        sys.exit(1)
    key = API_KEY_FILE.read_text().strip()
    m = Memory()
    m._cfg.models.text_embedding = TEXT_MODEL
    m._cfg.models.image_embedding = IMAGE_MODEL
    m.authenticate(user_id=DEMO_USER, api_key=key)
    return m


def _reset(memory: Memory) -> None:
    LOG.info("resetting diagram entries from session %r…", DEMO_SESSION)
    probes = memory.search(
        filters={
            "session_name": DEMO_SESSION,
            "organization": DEMO_WORKSPACE,
            "department": "diagrams",
        },
        k=100,
    )
    if probes:
        memory.remove(results=probes)
        LOG.info("  removed %d diagram entries", len(probes))
    else:
        LOG.info("  nothing to remove.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="wipe diagram entries first")
    args = p.parse_args(argv)

    LOG.info("connecting to Nexus…")
    memory = _memory_for_demo_user()
    principal = memory.authenticate(
        user_id=DEMO_USER, api_key=API_KEY_FILE.read_text().strip()
    )

    if args.reset:
        _reset(memory)

    ctx = Context(
        principal=principal,
        session_name=DEMO_SESSION,
        purpose="aperture-nexus architecture diagrams",
        organization=DEMO_WORKSPACE,
        department="diagrams",
    )

    for figure_id, rel_path, caption in DIAGRAMS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            LOG.warning("missing (skipped): %s", rel_path)
            continue

        info = Information(context_id=ctx.id)
        # 1. Caption as a text entry — searchable via BGE.
        info.log(
            text=caption,
            metadata={
                "kind": "diagram_caption",
                "figure_id": figure_id,
                "image_rel_path": rel_path,
            },
            tag=f"diagram:{figure_id}",
        )
        # 2. Image bytes as an image entry — stored in Nexus, embedded by CLIP.
        info.log(
            image=str(path),
            metadata={
                "kind": "diagram_image",
                "figure_id": figure_id,
                "image_rel_path": rel_path,
            },
            tag=f"diagram:{figure_id}",
        )
        LOG.info("  commit  %-20s  (caption + image)  %s", figure_id, rel_path)
        memory.process_and_commit(ctx, info)

    LOG.info("done — %d diagrams ingested.", len(DIAGRAMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
