# GIF recording playbook — ask-the-codebase demo

A step-by-step script for recording the hero GIF embedded in
[`docs/ask-the-codebase.md`](../../docs/ask-the-codebase.md).

Goal: a ~30-45 second loop that lands three moments the demo is good at.
Keep it tight — no narration beat should linger more than 7 seconds.

## Setup

1. **Browser window:** 1280 × 800 Chrome in a new incognito window so no
   extensions / badges show. Hide bookmarks bar (`⌘⇧B` on macOS,
   `Ctrl+Shift+B` on Linux).
2. **Zoom:** 100% (`⌘0` / `Ctrl+0`). Verify the input + first chip row
   are fully visible without scrolling.
3. **Server up:**
   ```bash
   ANTHROPIC_API_KEY=sk-ant-... \
     ANTHROPIC_WORKSPACE_ID=wrkspc_... \
     uvicorn demo.webapp.app:app --port 9001 --host 0.0.0.0
   ```
4. **Warm the model** by running one throwaway search (so the
   sentence-transformer isn't loading inside the recording). Reload the
   page after.
5. **Recorder:** QuickTime, OBS, Peek (Linux), Kap (macOS), or
   LICEcap. Target 15 fps, output MP4 or GIF directly.

## Shot list (~35 seconds total)

| Beat | Elapsed | Action | What's on screen |
|---|---|---|---|
| 0. Hero | 0:00–0:02 | Nothing — show the empty state with the schema graph on the right | Clean page, chips row visible |
| 1. Information vs Memory | 0:02–0:10 | Click chip **"Information vs Memory"** | Sources arrive first; answer streams in; graph on the right updates in parallel. Pause once the first ~3 lines of the answer are visible. |
| 2. Git attribution | 0:10–0:20 | Click chip **"Principal fields: when & who"** | Watch the ANSWER card land on the exact commit SHA, date, and author. Pause when the answer mentions "commit 5d6f1ac2" and "2026-03-23". |
| 3. Diagram retrieval | 0:20–0:33 | Click chip **"Show architecture"** | DIAGRAMS strip appears above the graph with the architecture PNG; the answer explains what it shows; the SOURCES panel below has the diagram card with the inline image. |
| 4. Loop frame | 0:33–0:35 | Hold on the diagram view for 2 seconds, then cut back to the Information vs Memory result as a soft loop point. |

## Tips

- Click chips, don't type — chips are reproducible, typing is noisy.
- Don't open DevTools. If you need to debug, do it before recording.
- If the model stream feels slow, that's OK — the GIF will tell the
  truth. If it feels unfairly slow, re-run and keep the recording where
  the LLM was snappier. Don't speed-ramp the recording in post.
- If the browser's address bar is distracting, crop to the content area.
  QuickTime → trim; Peek → crop before export.

## Export

- Final format: GIF, 15 fps, ≤ 5 MB. If the GIF balloons past that,
  shorten beats 1 and 2, or encode as a Twitter-friendly MP4 and
  convert last.
- Target dimensions: 1280 × 800 or scaled to 1000 × 625.
- Save to `demo/ask-the-codebase.gif` so the docs page's embed resolves
  (the page references it via the raw.githubusercontent.com URL, matching
  the convention `demo/demo.gif` and `demo/customer_support_demo.gif`
  already follow).
