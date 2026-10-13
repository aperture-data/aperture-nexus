"""
record_gif.py — automated GIF recording for the ask-the-codebase demo.

Drives Chromium through the three-beat demo script (Information vs
Memory → git attribution → architecture diagram) and writes a GIF to
docs/ask-the-codebase.gif. The webapp must already be running on
http://localhost:9001.

Usage (from aperture-nexus/):
    python demo/webapp/record_gif.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = Path(__file__).resolve().parent
OUT_WEBM_DIR = WEBAPP_DIR / "recordings"
OUT_GIF = REPO_ROOT / "demo" / "ask-the-codebase.gif"

URL = os.environ.get("NEXUS_DEMO_URL", "http://localhost:9001")
VIEWPORT = {"width": 1400, "height": 900}

# Beats to click (chip label-match → wait seconds).
# Each beat allows: sources + answer streaming (~3-4 s) + graph settle
# (~2 s) + reading dwell (~4-5 s) before moving on. Natural speed on
# playback so the viewer can actually read; no post-recording speedup.
BEATS = [
    ("Information vs Memory", 10.0),
    ("Principal fields: when & who", 11.0),
    ("Show architecture", 11.0),
]

# No ffmpeg speedup — uniform fast-forward made the reading phase too
# rushed and the graph's zoom-to-fit animation didn't noticeably
# improve. Natural pacing wins. If you want a shorter GIF, shorten the
# beats above instead of ramping speed.
SPEEDUP = 1.0


async def _click_chip(page, label: str):
    """Click an example chip by its visible label (prefix match)."""
    chip = page.locator(".examples span", has_text=label).first
    await chip.wait_for(state="visible", timeout=5000)
    await chip.click()


async def main() -> int:
    OUT_WEBM_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any old recording files so we pick up the fresh one.
    for p in OUT_WEBM_DIR.glob("*.webm"):
        p.unlink()

    # Warm the server BEFORE Playwright opens so the sentence-transformer
    # model is already loaded. This way the recording's first frame is a
    # settled hero, not a model-loading pause.
    print("→ warming server (loads embedding model)", flush=True)
    subprocess.run([
        "curl", "-s", "-o", "/dev/null", "-X", "POST",
        f"{URL}/search", "-H", "Content-Type: application/json",
        "-d", '{"query":"warmup","k":1}',
    ], check=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUT_WEBM_DIR),
            record_video_size=VIEWPORT,
        )
        page = await context.new_page()

        print(f"→ loading {URL}", flush=True)
        await page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        # Let the schema-graph iframe finish react-force-graph physics
        # before any interaction so the hero frame has visible nodes.
        await page.wait_for_timeout(5000)

        # Beat 0: 1.5 s of settled hero
        print("→ beat 0: hero", flush=True)
        await page.wait_for_timeout(1500)

        for label, wait_s in BEATS:
            print(f"→ beat: {label}", flush=True)
            await _click_chip(page, label)
            await page.wait_for_timeout(int(wait_s * 1000))

        # Hold on the last frame briefly for a soft loop point.
        await page.wait_for_timeout(1500)

        await context.close()
        await browser.close()

    # The video file is named automatically by Playwright.
    webms = sorted(OUT_WEBM_DIR.glob("*.webm"))
    if not webms:
        print("ERROR: no recording produced", file=sys.stderr)
        return 1
    webm = webms[-1]
    print(f"→ recorded {webm}  ({webm.stat().st_size // 1024} KB)")

    # Convert webm → GIF.
    #   -ss skips the opening while the schema graph is still rendering.
    #   mpdecimate drops near-identical consecutive frames (the "nothing
    #      is changing" dead time — graph loading pane, LLM wait before
    #      first token). Text streaming and graph animations stay because
    #      those frames actually differ.
    #   setpts=N/FRAME_RATE/TB re-times the output so playback is smooth
    #      after the drops.
    palette = OUT_WEBM_DIR / "palette.png"
    speed_fragment = f"setpts={1 / SPEEDUP:.4f}*PTS," if SPEEDUP != 1.0 else ""
    # mpdecimate thresholds: hi/lo/frac. Tuned to keep subtle changes
    # (streaming tokens, graph physics settling, diagram strip opening)
    # while cutting pure dead time (blank graph iframe loading, LLM
    # waiting for first token). More permissive than before so each
    # beat remains visually distinct.
    cut_dead = "mpdecimate=hi=1000:lo=400:frac=0.2,setpts=N/FRAME_RATE/TB"
    scale_vf = f"{speed_fragment}{cut_dead},fps=8,scale=900:-1:flags=lanczos"
    skip_s = "4"
    print(f"→ converting to GIF via ffmpeg (skip {skip_s}s, cut dead frames)", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", skip_s, "-i", str(webm),
        "-vf", f"{scale_vf},palettegen=max_colors=64",
        str(palette),
    ], check=True, capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", skip_s, "-i", str(webm), "-i", str(palette),
        "-filter_complex", f"{scale_vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        str(OUT_GIF),
    ], check=True, capture_output=True)

    print(f"→ wrote {OUT_GIF}  ({OUT_GIF.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
