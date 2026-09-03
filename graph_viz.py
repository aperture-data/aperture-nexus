"""
graph_viz.py — reusable ApertureDB -> interactive graph helpers.

Built for the Honda demo but written to be reused across other ApertureDB demos:
none of this is specific to this graph's schema except the default color palette,
which auto-generates for any class it hasn't seen before.

Usage in a notebook:

    import graph_viz as gv

    universities = gv.get_entities(client, "University", label_prop="name")
    edges = gv.get_edges(client, "Researcher", "University", "AFFILIATED_WITH",
                          [r["id"] for r in researchers])
    gv.render_graph(nodes, edges, "My Graph", "my_graph.html", client=client)

Renders with react-force-graph-2d (React + graph library loaded from CDN, no pip
install needed). Opens in a new browser tab by default; pass embed=True for an
inline IFrame preview instead (or in addition).
"""

import base64
import json
import os
import webbrowser

from IPython.display import display, Markdown, IFrame

# Auto-assigned to any class not given an explicit color — order-stable so the
# same class always gets the same color across renders within one process.
_DEFAULT_PALETTE = [
    "#1f4e8c", "#c9560c", "#3d7ec9", "#e88b3f", "#8ec1f0",
    "#f5c187", "#b34d9e", "#d6a0c9", "#2ca02c", "#d62728",
]
_auto_colors = {}


def _color_for(cls, node_colors):
    if node_colors and cls in node_colors:
        return node_colors[cls]
    if cls not in _auto_colors:
        _auto_colors[cls] = _DEFAULT_PALETTE[len(_auto_colors) % len(_DEFAULT_PALETTE)]
    return _auto_colors[cls]


def get_entities(client, cls, label_prop="name", id_prop="id"):
    """Fetch all entities of a class with their full properties.

    Returns a list of {"id", "label", "cls", "props"} dicts ready for render_graph.
    """
    resp, _ = client.query([{"FindEntity": {"with_class": cls, "results": {"all_properties": True}}}])
    ents = resp[0]["FindEntity"].get("entities", [])
    out = []
    for e in ents:
        props = {k: v for k, v in e.items() if not k.startswith("_") and k not in (id_prop, label_prop)}
        out.append({"id": e[id_prop], "label": e.get(label_prop, e[id_prop]), "cls": cls, "props": props})
    return out


def get_edges(client, src_cls, dst_cls, conn_cls, src_ids, id_prop="id"):
    """Anchor on each src entity and look up its connected dst entities.

    One query per src id — small N is expected (demo-scale graphs), so this stays
    simple and unambiguous rather than trying to batch and guess at result ordering.
    Returns a list of (src_id, dst_id) tuples.
    """
    pairs = []
    for sid in src_ids:
        q = [
            {"FindEntity": {"with_class": src_cls, "_ref": 1, "constraints": {id_prop: ["==", sid]}}},
            {"FindEntity": {"with_class": dst_cls,
                             "is_connected_to": {"ref": 1, "connection_class": conn_cls},
                             "results": {"list": [id_prop]}}},
        ]
        resp, _ = client.query(q)
        for d in resp[1]["FindEntity"].get("entities", []):
            pairs.append((sid, d[id_prop]))
    return pairs


def get_blobs(client, src_cls, conn_cls, src_ids, id_prop="id", label_prop="name"):
    """Like get_entities + get_edges combined, but for Blob nodes — Blob is a special
    ApertureDB object type (like Image), so it needs FindBlob, not FindEntity.

    Returns (nodes, edges): nodes are {"id", "label", "cls": "Blob", "props"} dicts,
    edges are (src_id, blob_id) tuples — both ready for render_graph.
    """
    nodes, edges = [], []
    seen_ids = set()
    for sid in src_ids:
        q = [
            {"FindEntity": {"with_class": src_cls, "_ref": 1, "constraints": {id_prop: ["==", sid]}}},
            {"FindBlob": {"is_connected_to": {"ref": 1, "connection_class": conn_cls},
                          "results": {"all_properties": True}}},
        ]
        resp, _ = client.query(q)
        for b in resp[1]["FindBlob"].get("entities", []):
            bid = b.get(id_prop)
            if bid is None:
                continue
            edges.append((sid, bid))
            if bid not in seen_ids:
                seen_ids.add(bid)
                props = {k: v for k, v in b.items() if not k.startswith("_") and k not in (id_prop, label_prop)}
                nodes.append({"id": bid, "label": b.get(label_prop, bid), "cls": "Blob", "props": props})
    return nodes, edges


def get_blob_data_uri(client, blob_id, id_prop="id"):
    """Fetch a Blob's bytes by id and return a base64 data: URI. Mime type is read
    from the blob's own 'type' property when present (e.g. 'pdf' -> application/pdf),
    falling back to a generic binary type otherwise."""
    resp, blobs = client.query([{"FindBlob": {"constraints": {id_prop: ["==", blob_id]},
                                               "blobs": True, "results": {"list": ["type"]}}}])
    if not blobs:
        return None
    ents = resp[0]["FindBlob"].get("entities", [])
    btype = (ents[0].get("type") if ents else "") or ""
    mime = "application/pdf" if btype.lower() == "pdf" else "application/octet-stream"
    b64 = base64.b64encode(blobs[0]).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _sniff_mime(blob_bytes):
    """Detect image format from magic bytes rather than trusting a stored property —
    robust regardless of whether the format was recorded at ingest time."""
    if blob_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if blob_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if blob_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def get_image_data_uri(client, image_id, id_prop="id"):
    """Fetch an Image entity's blob by id and return a base64 data: URI, or None
    if no matching image/no blob was found."""
    resp, blobs = client.query([{"FindImage": {"constraints": {id_prop: ["==", image_id]}, "blobs": True}}])
    if not blobs:
        return None
    mime = _sniff_mime(blobs[0])
    b64 = base64.b64encode(blobs[0]).decode("ascii")
    return f"data:{mime};base64,{b64}"


def get_video_data_uri(client, video_id, id_prop="id"):
    """Fetch a Video entity's blob by id and return a base64 data: URI, or None
    if no matching video/no blob was found. Mime type read from the video's
    'container' property when present (e.g. 'mp4' -> video/mp4), falling back
    to video/mp4 otherwise since that's overwhelmingly the common case."""
    resp, blobs = client.query([{"FindVideo": {"constraints": {id_prop: ["==", video_id]},
                                                "blobs": True, "results": {"list": ["container"]}}}])
    if not blobs:
        return None
    ents = resp[0].get("FindVideo", {}).get("entities", [])
    container = (ents[0].get("container") if ents else "") or "mp4"
    mime = f"video/{container.lower()}"
    b64 = base64.b64encode(blobs[0]).decode("ascii")
    return f"data:{mime};base64,{b64}"


GRAPH_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: #0b1020; overflow: hidden; }
  #graph { width: 100vw; height: 100vh; position: relative; }
  #info { position: absolute; top: 14px; left: 14px; color: #eaeaea; font-family: -apple-system, sans-serif;
          font-size: 16px; z-index: 10; background: rgba(0,0,0,0.55); padding: 8px 16px; border-radius: 8px; }
  #titlebox { position: absolute; top: 14px; right: 18px; color: #eaeaea; font-family: -apple-system, sans-serif;
           font-size: 19px; font-weight: 700; z-index: 10; text-shadow: 0 1px 4px rgba(0,0,0,0.6); }
  #panel { position: absolute; bottom: 18px; right: 18px; width: 400px; max-height: 68vh; overflow-y: auto;
           color: #eaeaea; font-family: -apple-system, sans-serif; font-size: 15px; z-index: 10;
           background: rgba(10,14,30,0.96); border: 1px solid rgba(255,255,255,0.18);
           border-radius: 12px; padding: 16px 18px; box-shadow: 0 6px 24px rgba(0,0,0,0.4); }
  #panel .placeholder { color: rgba(234,234,234,0.55); font-style: italic; }
  #panel .panel-title { font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 10px; }
  #panel b { color: #8ec1f0; }
  #panel .row { margin: 5px 0; word-break: break-word; line-height: 1.4; }
  #panel img, #panel video { max-width: 100%; border-radius: 8px; margin-bottom: 10px; display: block; }
  #panel a.openfile { color: #8ec1f0; font-weight: 700; font-size: 15px; text-decoration: none; }
  #panel a.openfile:hover { text-decoration: underline; }
</style>
<script type="importmap">{ "imports": {
  "react": "https://esm.sh/react@18",
  "react-dom": "https://esm.sh/react-dom@18/client"
}}</script>
</head>
<body>
<div id="info">Hover a node to highlight its connections. Click a node for details.</div>
<div id="titlebox">__TITLE__</div>
<div id="panel"><div class="placeholder">Click a node to see its details here.</div></div>
<div id="graph"></div>
<script src="//cdn.jsdelivr.net/npm/@babel/standalone"></script>
<script type="text/jsx" data-type="module">
import ForceGraph2D from 'https://esm.sh/react-force-graph-2d?external=react';
import React from 'react';
import { createRoot } from 'react-dom';

const graphData = __GRAPH_DATA__;
const preHighlight = __HIGHLIGHT_IDS__;

function Graph() {
  const fgRef = React.useRef();
  const [highlightNodes, setHighlightNodes] = React.useState(new Set(preHighlight));
  const [highlightLinks, setHighlightLinks] = React.useState(new Set());
  const [hoverNode, setHoverNode] = React.useState(null);
  const [clickedNode, setClickedNode] = React.useState(null);

  const linkEndId = (end) => (typeof end === 'object' && end !== null) ? end.id : end;

  const handleHover = (node) => {
    const hn = new Set();
    const hl = new Set();
    if (node) {
      hn.add(node.id);
      graphData.links.forEach((l) => {
        const a = linkEndId(l.source);
        const b = linkEndId(l.target);
        if (a === node.id || b === node.id) {
          hl.add(l);
          hn.add(a);
          hn.add(b);
        }
      });
    } else if (preHighlight.length) {
      preHighlight.forEach((id) => hn.add(id));
    }
    setHighlightNodes(hn);
    setHighlightLinks(hl);
    setHoverNode(node || null);
  };

  const handleClick = (node) => {
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 700);
      fgRef.current.zoom(5, 700);
    }
    setClickedNode(node);
  };

  React.useEffect(() => {
    const infoEl = document.getElementById('info');
    if (infoEl) {
      infoEl.textContent = hoverNode
        ? (hoverNode.cls + ': ' + hoverNode.name)
        : (preHighlight.length
            ? preHighlight.length + ' node(s) highlighted from the last search. Click any node for details.'
            : 'Hover a node to highlight its connections. Click a node for details.');
    }
  }, [hoverNode]);

  React.useEffect(() => {
    const el = document.getElementById('panel');
    if (!el) return;
    if (!clickedNode) {
      el.innerHTML = '<div class="placeholder">Click a node to see its details here.</div>';
      return;
    }
    const props = clickedNode.props || {};
    const rows = Object.keys(props).map((k) => {
      let v = props[k];
      if (typeof v === 'string' && v.length > 220) v = v.slice(0, 220) + '\\u2026';
      return '<div class="row"><b>' + k + ':</b> ' + v + '</div>';
    }).join('');
    const imgHtml = clickedNode.image ? '<img src="' + clickedNode.image + '" />' : '';
    const videoHtml = clickedNode.video ? '<video src="' + clickedNode.video + '" controls></video>' : '';

    // Revoke the previous blob URL (if any) before making a new one, to avoid leaking memory
    // across repeated clicks.
    if (window.__lastBlobUrl) {
      URL.revokeObjectURL(window.__lastBlobUrl);
      window.__lastBlobUrl = null;
    }
    let fileHtml = '';
    if (clickedNode.file) {
      // Browsers block top-level navigation to data: URIs via <a href>/target="_blank" for
      // security reasons, and it fails completely silently. A blob: URL doesn't have that
      // restriction, but it must be the actual href (not set via an onclick handler) so that
      // right-click -> "open in new tab" also works, since that bypasses onclick and follows
      // the raw href directly.
      const [meta, b64] = clickedNode.file.split(',');
      const mimeMatch = meta.match(/data:(.*?);base64/);
      const mime = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
      window.__lastBlobUrl = blobUrl;
      fileHtml = '<div class="row"><a href="' + blobUrl + '" target="_blank" rel="noopener" ' +
                 'class="openfile">\\ud83d\\udcc4 Open PDF \\u2197</a></div>';
    }
    el.innerHTML = '<div class="panel-title">' + clickedNode.name + '</div>' + imgHtml + videoHtml + fileHtml +
                    '<div class="row"><b>class:</b> ' + clickedNode.cls + '</div>' + rows;
  }, [clickedNode]);

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={graphData}
      nodeId="id"
      nodeLabel="name"
      nodeRelSize={6}
      nodeColor={(n) => (highlightNodes.size === 0 || highlightNodes.has(n.id)) ? n.color : 'rgba(120,120,120,0.25)'}
      nodeCanvasObjectMode={() => 'after'}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const dim = highlightNodes.size > 0 && !highlightNodes.has(node.id);
        const fontSize = Math.max(14 / globalScale, 4);
        ctx.font = `600 ${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = dim ? 'rgba(160,160,160,0.35)' : '#eaeaea';
        ctx.fillText(node.name, node.x, node.y + 7);
      }}
      linkColor={(l) => highlightLinks.has(l) ? '#2ca02c' : (l.color || 'rgba(255,255,255,0.55)')}
      linkWidth={(l) => highlightLinks.has(l) ? 3.5 : 2.2}
      linkDirectionalArrowLength={5}
      linkDirectionalArrowRelPos={1}
      linkLabel="name"
      backgroundColor="#0b1020"
      onNodeHover={handleHover}
      onNodeClick={handleClick}
      width={window.innerWidth}
      height={window.innerHeight}
    />
  );
}

createRoot(document.getElementById('graph')).render(<Graph/>);
</script>
</body>
</html>
"""


def render_graph(nodes, edges, title, filename, client=None, node_colors=None,
                  highlight_ids=None, height=600, open_browser=True, embed=False):
    """Render a graph with react-force-graph-2d: hover to highlight connections,
    click a node for a details panel (with an inline image if it's an Image node).

    - client: if given, Image-class nodes get their actual picture fetched and
      embedded, Video-class nodes get an embedded playable preview, and Blob-class
      nodes get an "Open PDF" link (data URI, so it's the real file, not an
      external link) — all shown in the fixed corner panel. Omit if you don't
      need any of these for this render.
    - highlight_ids: node ids to pre-highlight on load (e.g. search results) —
      useful for showing "these are the nodes a query just returned" visually.
    - open_browser: opens the rendered HTML in a new tab (default). Works from
      any Jupyter environment since it's just a real file on disk.
    - embed: also return an IFrame for inline preview in the notebook. Off by
      default — full-tab viewing has more room and sidesteps iframe sandboxing
      that some hosted notebook environments apply to local files.
    """
    seen = set()
    out_nodes = []
    for n in nodes:
        if n["id"] in seen:
            continue
        seen.add(n["id"])
        node = {"id": n["id"], "name": str(n["label"])[:60], "cls": n["cls"],
                "color": _color_for(n["cls"], node_colors), "props": n.get("props", {})}
        if client is not None and n["cls"] == "Image":
            img = get_image_data_uri(client, n["id"])
            if img:
                node["image"] = img
        if client is not None and n["cls"] == "Blob":
            f = get_blob_data_uri(client, n["id"])
            if f:
                node["file"] = f
        if client is not None and n["cls"] == "Video":
            v = get_video_data_uri(client, n["id"])
            if v:
                node["video"] = v
        out_nodes.append(node)

    out_links = [{"source": s, "target": d, "name": label, "color": color}
                 for s, d, label, color in edges if s in seen and d in seen]
    graph_data = {"nodes": out_nodes, "links": out_links}

    html = (GRAPH_HTML_TEMPLATE
            .replace("__GRAPH_DATA__", json.dumps(graph_data))
            .replace("__TITLE__", title)
            .replace("__HIGHLIGHT_IDS__", json.dumps(list(highlight_ids or []))))
    with open(filename, "w") as f:
        f.write(html)

    display(Markdown(f"**{title}**"))
    abspath = os.path.abspath(filename)
    if open_browser:
        webbrowser.open(f"file://{abspath}")
        print(f"Opened in a new browser tab: {abspath}")
    if embed or not open_browser:
        return IFrame(filename, width=940, height=height + 50)
    return None


def get_schema_graph(client):
    """Fetches ApertureDB's real schema (GetSchema) and reshapes it into class-level
    nodes and edges — one node per entity class (with its live count and property
    list), one edge per connection class (with its live count) — ready for
    render_graph. This is a polished alternative to GetSchema's raw JSON, built from
    the same call, not a separate/fake schema view."""
    resp, _ = client.query([{"GetSchema": {}}])
    schema = resp[0].get("GetSchema", {})
    ent_classes = schema.get("entities", {}).get("classes", {}) or {}
    conn_classes = schema.get("connections", {}).get("classes", {}) or {}

    nodes = []
    for cls, info in ent_classes.items():
        count = info.get("matched", 0)
        props = info.get("properties", {}) or {}
        display = cls.lstrip("_")
        nodes.append({
            "id": cls,
            "label": f"{display} ({count})",
            "cls": cls,
            "props": {"count": count, "properties": ", ".join(sorted(props.keys())) or "(none)"},
        })

    node_ids = {n["id"] for n in nodes}
    edges = []
    for conn_cls, pairs in conn_classes.items():
        # System-generated connections (has_embedding, _BoundingBoxToImage, etc.) get a
        # muted color so user-defined relationships (AFFILIATED_WITH, MONITORS, ...) stand out.
        is_system = conn_cls.startswith("_") or conn_cls.islower()
        color = "rgba(150,150,150,0.5)" if is_system else None
        for pair in pairs:
            src, dst, count = pair.get("src"), pair.get("dst"), pair.get("matched", 0)
            if src not in node_ids or dst not in node_ids:
                continue
            edges.append((src, dst, f"{conn_cls} ({count})", color))
    return nodes, edges


def render_schema(client, title="ApertureDB Schema", filename="schema.html", **kwargs):
    """One call: fetches the live schema and renders it as a polished, navigable graph
    — hover a class to see its connections, click for its properties — instead of
    GetSchema's raw JSON dump. Accepts the same kwargs as render_graph
    (open_browser, embed, height, highlight_ids, node_colors)."""
    nodes, edges = get_schema_graph(client)
    return render_graph(nodes, edges, title, filename, **kwargs)
