"""
graph_viz.py - ApertureDB -> interactive graph: schema view, neighbourhood navigation, query results.

Started in aperture-nexus (MIT, ApertureData Inc.) and rewritten here so that *any* ApertureDB graph
can be explored without per-schema code.  Nothing in this file knows about a particular schema.

    import graph_viz as gv

    gv.render_schema(client)                                          # classes + connections, live counts
    gv.render_neighborhood(client, "_Image", {"id": ["==", "abc"]})   # start somewhere, expand outwards
    gv.render_query(client, [{"FindImage": {...}}])                   # draw what ANY query returned

How navigation works
    GetSchema says which connection classes join which object classes.  ``expand`` then asks, for
    each frontier node, for its neighbours with one batched query per (connection class, neighbour
    class) using ``group_by_source`` + ``per_group``: caps and sort order apply *per source node*,
    so a poster with 50 labels shows its best 8 and nothing is fetched node by node.

In the page
    * hover highlights connections; click shows the node's properties, its picture / video / crop,
      the bounding boxes of an image drawn over it, and the query that fetches it
    * the legend lists classes with counts - click one to hide or show it
    * the search box dims every node that does not match
    Pages load react-force-graph from a CDN, so the browser needs internet access.

Command line
    python graph_viz.py schema
    python graph_viz.py node _Image --where gcs_object_stem=9614 --depth 1
    python graph_viz.py query my_query.json

Compared with the aperture-nexus version: native types (_Image, _BoundingBox, ...) work everywhere,
neighbours come from batched, capped queries instead of one query per node, nodes are identified by
_uniqueid (no ``id`` property required), payloads are sniffed instead of assumed, the page escapes
all data, and template substitution is single-pass.  ``get_entities`` / ``get_edges`` / ``get_blobs``
and the three ``get_*_data_uri`` helpers are gone - ``find_nodes`` / ``expand`` / ``get_payload_uri``
replace them.
"""
import argparse
import base64
import copy
import json
import os
import re
import sys
import webbrowser

try:  # the CLI and file output work without IPython; notebooks get inline display
    from IPython.display import IFrame, Markdown, display
except ImportError:  # pragma: no cover
    IFrame = Markdown = display = None

# ---------------------------------------------------------------------------------------------
# Colors and labels
# ---------------------------------------------------------------------------------------------

_DEFAULT_PALETTE = [
    "#1f4e8c", "#c9560c", "#3d7ec9", "#e88b3f", "#8ec1f0",
    "#f5c187", "#b34d9e", "#d6a0c9", "#2ca02c", "#d62728",
]
_auto_colors = {}

# Properties tried, in order, to name a node when ``label_props`` has nothing for its class.
_LABEL_CANDIDATES = ("name", "title", "label", "description", "_label", "filename", "id")
_HIDDEN_PROPS = ("_blob_index",)
SYSTEM_EDGE_COLOR = "rgba(150,150,150,0.5)"


def _color_for(cls, node_colors):
    if node_colors and cls in node_colors:
        return node_colors[cls]
    if cls not in _auto_colors:
        _auto_colors[cls] = _DEFAULT_PALETTE[len(_auto_colors) % len(_DEFAULT_PALETTE)]
    return _auto_colors[cls]


def _label_for(entity, cls, label_props):
    prop = (label_props or {}).get(cls)
    if prop and entity.get(prop) not in (None, ""):
        return str(entity[prop])
    for cand in _LABEL_CANDIDATES:
        value = entity.get(cand)
        if value not in (None, ""):
            value = str(value)
            # hash-like ids make unreadable labels: shorten them
            return value if cand != "id" or len(value) <= 16 else value[:8] + "\u2026"
    if cls == "_Descriptor":  # a vector: its only readable property would be an internal id
        return "embedding"
    return str(entity.get("_uniqueid", cls))


def _edge4(edge):
    """Accept (src, dst), (src, dst, label) or (src, dst, label, color)."""
    src, dst, *rest = edge
    label = rest[0] if rest and rest[0] is not None else ""
    color = rest[1] if len(rest) > 1 else None
    return src, dst, label, color


# ---------------------------------------------------------------------------------------------
# Object types and how to find them
# ---------------------------------------------------------------------------------------------

# Native classes -> Find command.  Any other class name is a user entity (FindEntity with_class).
NATIVE_FIND = {
    "_Image": "FindImage", "_Video": "FindVideo", "_Blob": "FindBlob", "_Clip": "FindClip",
    "_BoundingBox": "FindBoundingBox", "_Polygon": "FindPolygon", "_Frame": "FindFrame",
    "_Descriptor": "FindDescriptor", "_DescriptorSet": "FindDescriptorSet",
}
_FIND_TO_CLASS = {v: k for k, v in NATIVE_FIND.items()}
_HAS_BLOB = ("_Image", "_Video", "_Blob", "_Clip", "_Frame", "_BoundingBox", "_Polygon", "_Descriptor")
_PIXELS = {"_Image": "image", "_BoundingBox": "image", "_Frame": "image", "_Video": "video", "_Blob": "file"}


def _find(cls, constraints=None, ref=None, limit=None, sort=None, extra=None, results=None):
    """One Find* command for objects of ``cls``; by default returns all their properties."""
    cmd = NATIVE_FIND.get(cls, "FindEntity")
    body = {"uniqueids": True, "results": results or {"all_properties": True}}
    if cmd == "FindEntity":
        body["with_class"] = cls
    if cls in _HAS_BLOB:
        body["blobs"] = False
    if cls in ("_BoundingBox", "_Polygon") and results is None:
        body["coordinates"] = True
    if constraints:
        body["constraints"] = constraints
    if ref is not None:
        body["_ref"] = ref
    if limit is not None:
        body["limit"] = limit
    if sort:
        body["sort"] = {"key": sort[0], "order": sort[1]}
    body.update(extra or {})
    return {cmd: body}


def snippet(cls, uniqueid=None, limit=10):
    """The query that fetches one object (or a few of a class) - shown in the page."""
    cmd = NATIVE_FIND.get(cls, "FindEntity")
    body = {}
    if cmd == "FindEntity":
        body["with_class"] = cls
    if uniqueid:
        body["constraints"] = {"_uniqueid": ["==", uniqueid]}
    else:
        body["limit"] = limit
    if cls in _HAS_BLOB:
        body["blobs"] = False
    body["results"] = {"all_properties": True}
    return json.dumps([{cmd: body}], indent=2)


def get_schema(client):
    """Live schema: ``{"classes": {cls: {count, properties}}, "connections": [{class, src, dst, count}]}``."""
    resp, _ = client.query([{"GetSchema": {}}])
    schema = resp[0].get("GetSchema", {})
    classes = {cls: {"count": info.get("matched", 0), "properties": sorted((info.get("properties") or {}).keys())}
               for cls, info in (schema.get("entities", {}).get("classes") or {}).items()}
    connections = [{"class": conn, "src": p.get("src"), "dst": p.get("dst"), "count": p.get("matched", 0)}
                   for conn, pairs in (schema.get("connections", {}).get("classes") or {}).items() for p in pairs]
    return {"classes": classes, "connections": connections}


def _node(entity, cls, label_props):
    props = {k: v for k, v in entity.items() if k != "_uniqueid" and k not in _HIDDEN_PROPS}
    return {"id": entity["_uniqueid"], "label": _label_for(entity, cls, label_props), "cls": cls,
            "props": props, "snippet": snippet(cls, entity["_uniqueid"])}


def find_nodes(client, cls, constraints=None, limit=25, sort=None, label_props=None):
    """Objects of ``cls`` matching ``constraints``, as graph nodes."""
    resp, _ = client.query([_find(cls, constraints, limit=limit, sort=sort)])
    body = list(resp[0].values())[0]
    if body.get("status", 0) != 0:
        raise ValueError(f"{cls}: {body.get('info')}")
    return [_node(e, cls, label_props) for e in body.get("entities", [])]


_BATCHED = {}   # id(client) -> does this server support group_by_source + per_group?


def _neighbours(client, cls, ids, conn_class, other, per_class, sort):
    """``{source _uniqueid: [neighbour entity, ...]}`` for the nodes ``ids`` of class ``cls``.

    Uses one batched query (``group_by_source`` + ``per_group``, newer servers).  If the server
    rejects those parameters it falls back - once, remembered per connection - to one query per
    source node, which gives the same result more slowly.
    """
    key = id(client)
    anchor = _find(cls, {"_uniqueid": ["in", ids]}, ref=1, results={"list": ["_uniqueid"]})
    if _BATCHED.get(key, True):
        batched = _find(other, sort=sort, limit=per_class, extra={
            "is_connected_to": {"ref": 1, "connection_class": conn_class},
            "group_by_source": True, "per_group": True})
        resp, _ = client.query([anchor, batched])
        status = None if isinstance(resp, dict) else list(resp[1].values())[0]
        if status is not None and status.get("status", 0) == 0 and isinstance(status.get("entities"), dict):
            _BATCHED[key] = True
            return status["entities"]
        info = str(resp.get("info") if isinstance(resp, dict) else (status or {}).get("info"))
        if "group_by_source" not in info and "per_group" not in info:
            return {}  # this class pair cannot be walked generically (e.g. descriptors need a set name): skip it
        _BATCHED[key] = False
    groups = {}
    for uid in ids:
        one = _find(cls, {"_uniqueid": ["==", uid]}, ref=1, results={"list": ["_uniqueid"]})
        near = _find(other, sort=sort, limit=per_class,
                     extra={"is_connected_to": {"ref": 1, "connection_class": conn_class}})
        resp, _ = client.query([one, near])
        if isinstance(resp, dict):
            continue
        body = list(resp[1].values())[0]
        groups[uid] = body.get("entities", []) if body.get("status", 0) == 0 else []
    return groups


def expand(client, nodes, depth=1, per_class=8, sort=None, include=None, exclude=None,
           max_nodes=200, label_props=None, schema=None):
    """Grow ``nodes`` outwards along the schema's connections, ``depth`` hops.

    per_class  at most this many neighbours per source node and connection class
    sort       ``{class: (property, "descending")}`` - which neighbours to keep when capping
    include / exclude   restrict which neighbour classes may be added
    max_nodes  stop adding once the graph is this big (a note is printed)

    Returns ``(nodes, edges)``; edges are ``(src, dst, connection_class)``.
    """
    schema = schema or get_schema(client)
    sort = sort or {}
    by_id = {n["id"]: n for n in nodes}
    edges, seen_edges, skipped = [], set(), 0
    frontier = list(nodes)

    for _ in range(depth):
        by_class = {}
        for n in frontier:
            by_class.setdefault(n["cls"], []).append(n["id"])
        added = []
        for cls, ids in by_class.items():
            for conn in schema["connections"]:
                if cls not in (conn["src"], conn["dst"]):
                    continue
                outgoing = cls == conn["src"]
                other = conn["dst"] if outgoing else conn["src"]
                if other not in schema["classes"] or (include and other not in include) \
                        or (exclude and other in exclude):
                    continue
                for start in range(0, len(ids), 100):
                    groups = _neighbours(client, cls, ids[start:start + 100], conn["class"], other,
                                         per_class, sort.get(other))
                    for src_id, found in groups.items():
                        for entity in found:
                            nid = entity["_uniqueid"]
                            if nid not in by_id:
                                if len(by_id) >= max_nodes:
                                    skipped += 1
                                    continue
                                by_id[nid] = _node(entity, other, label_props)
                                added.append(by_id[nid])
                            edge = (src_id, nid, conn["class"]) if outgoing else (nid, src_id, conn["class"])
                            if edge not in seen_edges:
                                seen_edges.add(edge)
                                edges.append(edge)
        frontier = added
        if not frontier:
            break
    if skipped:
        print(f"graph_viz: {skipped} more neighbour(s) not shown (max_nodes={max_nodes}).")
    return list(by_id.values()), edges


# ---------------------------------------------------------------------------------------------
# Payloads: pictures, crops, videos, files
# ---------------------------------------------------------------------------------------------

def _sniff_mime(data):
    """MIME type from magic bytes - no stored property is trusted or required."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if data[:5] == b"%PDF-":
        return "application/pdf"
    return "application/octet-stream"


sniff_mime = _sniff_mime


def get_payload_uri(client, cls, uniqueid, max_side=None, dims=None, max_bytes=8_000_000):
    """The stored bytes of one object (image, bounding-box crop, video, blob) as a data: URI, or None.

    For images, ``dims=(width, height)`` plus ``max_side`` shrinks the picture on the server
    (resize operation) so pages stay small.
    """
    cmd = NATIVE_FIND.get(cls)
    if cmd is None or cls not in _PIXELS:
        return None
    body = {"constraints": {"_uniqueid": ["==", uniqueid]}, "blobs": True, "results": {"list": ["_uniqueid"]}}
    if cls == "_Image" and max_side and dims and max(dims) > max_side:
        body["operations"] = [{"type": "resize", "scale": round(max_side / max(dims), 4)}]
    resp, blobs = client.query([{cmd: body}])
    if isinstance(resp, dict) or not blobs or len(blobs[0]) > max_bytes:
        return None
    return f"data:{_sniff_mime(blobs[0])};base64,{base64.b64encode(blobs[0]).decode('ascii')}"


def _dims(props):
    w, h = props.get("width") or props.get("_image_width"), props.get("height") or props.get("_image_height")
    return (int(w), int(h)) if isinstance(w, (int, float)) and isinstance(h, (int, float)) else None


def _attach_overlays(out_nodes, edges, node_color):
    """Bounding boxes of each image, drawn over it in the details panel."""
    by_id = {n["id"]: n for n in out_nodes}
    drawn = set()
    for src, dst, _, _ in map(_edge4, edges):
        for image_id, box_id in ((src, dst), (dst, src)):
            image, box = by_id.get(image_id), by_id.get(box_id)
            if not image or not box or image["cls"] != "_Image" or box["cls"] != "_BoundingBox":
                continue
            if (image_id, box_id) in drawn:  # several edge classes may join the same pair
                continue
            drawn.add((image_id, box_id))
            c = (box["props"] or {}).get("_coordinates")
            if isinstance(c, dict) and {"x", "y", "width", "height"} <= set(c):
                image.setdefault("overlays", []).append({
                    "x": c["x"], "y": c["y"], "w": c["width"], "h": c["height"],
                    "label": box["name"], "color": _color_for("box:" + box["name"], None)})


# ---------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------

GRAPH_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE_TEXT__</title>
<style>
  html, body { margin: 0; padding: 0; background: #0b1020; overflow: hidden; font-family: -apple-system, sans-serif; }
  #graph { width: 100vw; height: 100vh; position: relative; }
  .box { position: absolute; z-index: 10; color: #eaeaea; background: rgba(0,0,0,0.55); border-radius: 8px; }
  #info { top: 14px; left: 14px; font-size: 15px; padding: 8px 16px; max-width: 62vw; }
  #titlebox { top: 14px; right: 18px; font-size: 19px; font-weight: 700; background: none; text-shadow: 0 1px 4px rgba(0,0,0,0.6); }
  #legend { top: 62px; left: 14px; font-size: 14px; padding: 8px 12px; max-height: 40vh; overflow-y: auto; }
  #legend .item { cursor: pointer; margin: 3px 0; white-space: nowrap; }
  #legend .item.off { opacity: 0.35; text-decoration: line-through; }
  #legend .dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 7px; }
  #search { top: 56px; right: 18px; padding: 0; background: none; }
  #search input { width: 260px; padding: 7px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.25);
                  background: rgba(10,14,30,0.9); color: #eaeaea; font-size: 14px; outline: none; }
  #panel { bottom: 18px; right: 18px; width: 400px; max-height: 68vh; overflow-y: auto; font-size: 15px;
           background: rgba(10,14,30,0.96); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px;
           padding: 16px 18px; box-shadow: 0 6px 24px rgba(0,0,0,0.4); }
  #panel .head { display: flex; align-items: flex-start; gap: 10px; }
  #panel .head .panel-title { flex: 1; }
  #pclose { background: rgba(255,255,255,0.12); color: #eaeaea; border: 0; border-radius: 6px; width: 30px; height: 30px;
            font-size: 20px; line-height: 1; cursor: pointer; flex: none; }
  #pclose:hover { background: rgba(255,255,255,0.25); }
  #controls { bottom: 18px; left: 14px; display: flex; gap: 8px; padding: 6px; }
  #controls button { background: rgba(255,255,255,0.12); color: #eaeaea; border: 0; border-radius: 6px; padding: 7px 12px;
                     font-size: 14px; cursor: pointer; }
  #controls button:hover { background: rgba(255,255,255,0.25); }
  #panel .placeholder { color: rgba(234,234,234,0.55); font-style: italic; }
  #panel .panel-title { font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 10px; }
  #panel b { color: #8ec1f0; }
  #panel .row { margin: 5px 0; word-break: break-word; line-height: 1.4; }
  #panel .imgwrap { position: relative; margin-bottom: 10px; }
  #panel img, #panel video { max-width: 100%; border-radius: 8px; display: block; }
  #panel svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
  #panel pre { background: rgba(255,255,255,0.07); border-radius: 6px; padding: 8px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
  #panel a.openfile { color: #8ec1f0; font-weight: 700; text-decoration: none; }
</style>
<script type="importmap">{ "imports": {
  "react": "https://esm.sh/react@18",
  "react-dom": "https://esm.sh/react-dom@18/client"
}}</script>
</head>
<body>
<div id="info" class="box"></div>
<div id="titlebox" class="box"></div>
<div id="legend" class="box"></div>
<div id="search" class="box"><input id="q" type="search" placeholder="Search nodes..." autocomplete="off"></div>
<div id="panel" class="box"><div class="placeholder">Click a node to see its details here.</div></div>
<div id="controls" class="box"><button id="fit" title="Fit the whole graph in the window">Fit view</button><button id="relayout" title="Let the layout move again">Re-layout</button></div>
<div id="graph"></div>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone"></script>
<script type="text/jsx" data-type="module">
import ForceGraph2D from 'https://esm.sh/react-force-graph-2d?external=react';
import { forceX, forceY } from 'https://esm.sh/d3-force@3';
import React from 'react';
import { createRoot } from 'react-dom';

const graphData = __GRAPH_DATA__;
const preHighlight = __HIGHLIGHT_IDS__;
const preSelect = __SELECT_ID__;
const TITLE = __TITLE_JSON__;

// A node's own colour, faded (instead of grey) so context stays readable when something else is in focus.
const faded = (hex) => {
  const m = /^#([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return 'rgba(120,120,120,0.35)';
  const v = parseInt(m[1], 16);
  return 'rgba(' + (v >> 16) + ',' + ((v >> 8) & 255) + ',' + (v & 255) + ',0.6)';
};

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v) => {
  if (v !== null && typeof v === 'object') v = JSON.stringify(v);
  v = String(v);
  return v.length > 220 ? v.slice(0, 220) + '\u2026' : v;
};
const clsOf = {}, degree = {}, classCount = {}, classColor = {};
graphData.nodes.forEach((n) => { clsOf[n.id] = n.cls; classCount[n.cls] = (classCount[n.cls] || 0) + (n.count || 1); classColor[n.cls] = n.color; });
graphData.links.forEach((l) => { degree[l.source] = (degree[l.source] || 0) + 1; degree[l.target] = (degree[l.target] || 0) + 1; });
const endId = (e) => (typeof e === 'object' && e !== null) ? e.id : e;
document.getElementById('titlebox').textContent = TITLE;

function Graph() {
  const fgRef = React.useRef();
  const [highlightNodes, setHighlightNodes] = React.useState(new Set(preHighlight));
  const [highlightLinks, setHighlightLinks] = React.useState(new Set());
  const [hoverNode, setHoverNode] = React.useState(null);
  const [clickedNode, setClickedNode] = React.useState(null);
  const [hidden, setHidden] = React.useState(new Set());
  const [query, setQuery] = React.useState('');

  // Spread the layout out and fit it to the window once it settles.
  React.useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    const n = graphData.nodes.length;
    fg.d3Force('charge').strength(n > 60 ? -90 : -260);
    fg.d3Force('link').distance(n > 60 ? 45 : 85);
    window.__gv = {fg, graphData};  // handle for tests and debugging
    fg.d3Force('x', forceX(0).strength(0.07));
    fg.d3Force('y', forceY(0).strength(0.07));
    if (preSelect) {
      const node = graphData.nodes.find((x) => x.id === preSelect);
      if (node) setClickedNode(node);
    }
  }, []);

  const matches = (n) => {
    if (!query) return true;
    const q = query.toLowerCase();
    if (String(n.name).toLowerCase().includes(q) || String(n.cls).toLowerCase().includes(q)) return true;
    return Object.values(n.props || {}).some((v) => String(v).toLowerCase().includes(q));
  };
  const dim = (n) => (highlightNodes.size > 0 && !highlightNodes.has(n.id)) || (query && !matches(n));

  const handleHover = (node) => {
    const hn = new Set(), hl = new Set();
    if (node) {
      hn.add(node.id);
      graphData.links.forEach((l) => {
        const a = endId(l.source), b = endId(l.target);
        if (a === node.id || b === node.id) { hl.add(l); hn.add(a); hn.add(b); }
      });
    } else {
      preHighlight.forEach((id) => hn.add(id));
    }
    setHighlightNodes(hn); setHighlightLinks(hl); setHoverNode(node || null);
  };
  // Clicking only opens the details: the view and the layout stay where they are.
  const handleClick = (node) => setClickedNode(node);
  const settled = React.useRef(false);

  React.useEffect(() => {
    const fg = () => fgRef.current;
    const fit = () => { if (fg()) fg().zoomToFit(400, 90); };
    const relayout = () => {
      graphData.nodes.forEach((n) => { n.fx = undefined; n.fy = undefined; });
      settled.current = false;
      if (fg()) fg().d3ReheatSimulation();
    };
    const onKey = (e) => { if (e.key === 'Escape') setClickedNode(null); };
    document.getElementById('fit').addEventListener('click', fit);
    document.getElementById('relayout').addEventListener('click', relayout);
    window.addEventListener('keydown', onKey);
    return () => {
      document.getElementById('fit').removeEventListener('click', fit);
      document.getElementById('relayout').removeEventListener('click', relayout);
      window.removeEventListener('keydown', onKey);
    };
  }, []);

  React.useEffect(() => {
    document.getElementById('info').textContent = hoverNode
      ? (hoverNode.cls + ': ' + hoverNode.name)
      : (preHighlight.length ? preHighlight.length + ' highlighted node(s). Hover or click for details.'
                             : 'Hover to highlight connections; click for details.');
  }, [hoverNode]);

  React.useEffect(() => {
    const input = document.getElementById('q');
    const onInput = () => setQuery(input.value.trim());
    input.addEventListener('input', onInput);
    return () => input.removeEventListener('input', onInput);
  }, []);

  React.useEffect(() => {
    const el = document.getElementById('legend');
    el.innerHTML = Object.keys(classCount).sort().map((c) =>
      '<div class="item' + (hidden.has(c) ? ' off' : '') + '" data-cls="' + esc(c) + '"><span class="dot" style="background:' +
      esc(classColor[c]) + '"></span>' + esc(c) + ' (' + classCount[c] + ')</div>').join('');
    el.querySelectorAll('.item').forEach((item) => item.addEventListener('click', () => {
      const next = new Set(hidden);
      next.has(item.dataset.cls) ? next.delete(item.dataset.cls) : next.add(item.dataset.cls);
      setHidden(next);
    }));
  }, [hidden]);

  React.useEffect(() => {
    const el = document.getElementById('panel');
    if (window.__lastBlobUrl) { URL.revokeObjectURL(window.__lastBlobUrl); window.__lastBlobUrl = null; }
    if (!clickedNode) { el.innerHTML = '<div class="placeholder">Click a node to see its details here.</div>'; return; }
    const n = clickedNode;
    const rows = Object.keys(n.props || {}).map((k) => '<div class="row"><b>' + esc(k) + ':</b> ' + esc(fmt(n.props[k])) + '</div>').join('');
    let media = '';
    if (n.image) media = '<div class="imgwrap"><img id="pimg" src="' + n.image + '"><svg id="pov"></svg></div>';
    else if (n.video) media = '<video src="' + n.video + '" controls></video>';
    let file = '';
    if (n.file) {
      // data: URIs cannot be opened via <a href>; a blob: URL can (and survives right-click -> new tab).
      const [meta, b64] = n.file.split(',');
      const mime = (meta.match(/data:(.*?);base64/) || [])[1] || 'application/octet-stream';
      const bin = atob(b64), bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      window.__lastBlobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
      file = '<div class="row"><a class="openfile" target="_blank" rel="noopener" href="' + window.__lastBlobUrl + '">Open file \u2197</a></div>';
    }
    const snip = n.snippet ? '<div class="row"><b>query:</b></div><pre>' + esc(n.snippet) + '</pre>' : '';
    el.innerHTML = '<div class="head"><div class="panel-title">' + esc(n.name) + '</div>' +
                   '<button id="pclose" title="Close (Esc)" aria-label="Close details">\u00d7</button></div>' + media + file +
                   '<div class="row"><b>class:</b> ' + esc(n.cls) + '</div>' + rows + snip;
    el.querySelector('#pclose').addEventListener('click', () => setClickedNode(null));
    const img = el.querySelector('#pimg'), svg = el.querySelector('#pov');
    if (img && svg && n.overlays) {
      const draw = () => {
        const W = img.naturalWidth, H = img.naturalHeight, s = n.refw ? W / n.refw : 1;
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        const sw = Math.max(2, W / 180), fs = Math.max(12, W / 32);
        svg.innerHTML = n.overlays.map((o) => '<rect x="' + o.x * s + '" y="' + o.y * s + '" width="' + o.w * s + '" height="' + o.h * s +
          '" fill="none" stroke="' + esc(o.color) + '" stroke-width="' + sw + '"><title>' + esc(o.label) + '</title></rect>' +
          '<text x="' + (o.x * s + sw * 2) + '" y="' + (o.y * s + fs) + '" font-size="' + fs + '" font-weight="700" fill="' + esc(o.color) +
          '" stroke="#000" stroke-width="' + sw / 2.5 + '" paint-order="stroke">' + esc(o.label) + '</text>').join('');
      };
      img.complete ? draw() : img.addEventListener('load', draw);
    }
  }, [clickedNode]);

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={graphData}
      nodeId="id"
      nodeLabel={(n) => esc(n.cls + ': ' + n.name)}
      nodeRelSize={5}
      nodeVal={(n) => 1 + Math.sqrt(degree[n.id] || 0)}
      nodeVisibility={(n) => !hidden.has(n.cls)}
      onEngineStop={() => {
        const fg = fgRef.current;
        if (!fg || settled.current) return;
        settled.current = true;
        // Once the layout has settled, pin every node so nothing drifts afterwards (drag one to move it).
        graphData.nodes.forEach((n) => { n.fx = n.x; n.fy = n.y; });
        fg.zoomToFit(500, 90);
        // a handful of nodes would otherwise fill the whole frame with huge circles
        setTimeout(() => { if (fg.zoom() > 2) fg.zoom(2, 300); }, 650);
      }}
      onNodeDragEnd={(n) => { n.fx = n.x; n.fy = n.y; }}
      onBackgroundClick={() => setClickedNode(null)}
      warmupTicks={150}
      cooldownTicks={120}
      nodeColor={(n) => dim(n) ? faded(n.color) : n.color}
      nodeCanvasObjectMode={() => 'after'}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const fontSize = Math.max(14 / globalScale, 4);
        ctx.font = `600 ${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        if (clickedNode && clickedNode.id === node.id) {
          ctx.beginPath(); ctx.arc(node.x, node.y, 9, 0, 2 * Math.PI);
          ctx.lineWidth = 2.5 / globalScale; ctx.strokeStyle = '#ffd24a'; ctx.stroke();
        }
        ctx.fillStyle = dim(node) ? 'rgba(234,234,234,0.65)' : '#eaeaea';
        ctx.fillText(node.name, node.x, node.y + 7);
      }}
      linkVisibility={(l) => !hidden.has(clsOf[endId(l.source)]) && !hidden.has(clsOf[endId(l.target)])}
      linkColor={(l) => highlightLinks.has(l) ? '#2ca02c' : (l.color || 'rgba(255,255,255,0.55)')}
      linkWidth={(l) => highlightLinks.has(l) ? 3.5 : 2}
      linkDirectionalArrowLength={5}
      linkDirectionalArrowRelPos={1}
      linkLabel={(l) => esc(l.name)}
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


def _json_for_script(value):
    """JSON that is safe inside a <script> element (no closing tags, no line separators)."""
    return (json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _fill(template, values):
    """Single-pass substitution: markers inside the data itself are never re-substituted."""
    return re.sub(r"__(GRAPH_DATA|HIGHLIGHT_IDS|SELECT_ID|TITLE_JSON|TITLE_TEXT)__", lambda m: values[m.group(1)], template)


def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def render_graph(nodes, edges, title, filename, client=None, node_colors=None, highlight_ids=None,
                 select=None, height=600, open_browser=False, embed=None, max_pixels=40, crops=True,
                 image_max_side=800):
    """Write an interactive page for ``nodes`` / ``edges`` and show it.

    nodes   ``{"id", "label", "cls", "props"[, "snippet", "image"]}`` dicts
    edges   ``(src, dst[, label[, color]])`` tuples
    client  if given, pictures / crops / videos / files of the nodes are fetched and embedded
            (at most ``max_pixels`` of them; images are shrunk server-side to ``image_max_side``)
    highlight_ids  node ids shown in full colour while the rest is faded (e.g. a query's answer)
    select         node id whose details panel is open when the page loads
    open_browser   also open the page in a browser tab
    embed          return an inline IFrame; default: yes in a notebook, no in a plain script
    """
    seen, out_nodes, budget = set(), [], max_pixels
    for n in nodes:
        if n["id"] in seen:
            continue
        seen.add(n["id"])
        cls = n["cls"]
        node = {"id": n["id"], "name": str(n["label"])[:60], "cls": cls, "color": _color_for(cls, node_colors),
                "props": n.get("props", {}), "snippet": n.get("snippet", ""), "count": n.get("count", 1)}
        for key in ("image", "video", "file"):
            if n.get(key):
                node[key] = n[key]
        pixel_kind = _PIXELS.get(cls)
        if client is not None and pixel_kind and not any(k in node for k in ("image", "video", "file")) and budget > 0 \
                and (cls != "_BoundingBox" or crops):
            dims = _dims(node["props"])
            uri = get_payload_uri(client, cls, n["id"], max_side=image_max_side, dims=dims)
            if uri:
                budget -= 1
                node[pixel_kind] = uri
                if cls == "_Image" and dims and max(dims) > image_max_side:
                    node["refw"] = dims[0]
        out_nodes.append(node)
    if client is not None and budget <= 0:
        print(f"graph_viz: pictures embedded for the first {max_pixels} nodes only (max_pixels).")

    merged = {}
    for s_, d_, label, color in map(_edge4, edges):
        if s_ in seen and d_ in seen:
            link = merged.setdefault(tuple(sorted((s_, d_))), {"source": s_, "target": d_, "names": [], "color": color})
            if label and label not in link["names"]:
                link["names"].append(label)
    links = [{"source": l["source"], "target": l["target"], "name": " + ".join(l["names"]), "color": l["color"]}
             for l in merged.values()]  # parallel edges (e.g. HasDetection next to _BoundingBoxToImage) draw as one line
    _attach_overlays(out_nodes, edges, None)
    html = _fill(GRAPH_HTML_TEMPLATE, {
        "GRAPH_DATA": _json_for_script({"nodes": out_nodes, "links": links}),
        "HIGHLIGHT_IDS": _json_for_script(list(highlight_ids or [])),
        "SELECT_ID": _json_for_script(select),
        "TITLE_JSON": _json_for_script(title),
        "TITLE_TEXT": re.sub(r"[<>&]", "", str(title)),
    })
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    abspath = os.path.abspath(filename)
    print(f"{title}: {len(out_nodes)} nodes, {len(links)} edges -> {abspath}")
    if open_browser:
        webbrowser.open(f"file://{abspath}")
    if embed is None:
        embed = _in_notebook()
    if embed and IFrame is not None:
        return IFrame(filename, width=940, height=height + 50)
    return None


# ---------------------------------------------------------------------------------------------
# Schema view
# ---------------------------------------------------------------------------------------------

def get_schema_graph(client):
    """The real schema as class-level nodes and edges: one node per class (live count, properties),
    one edge per connection class and pair (live count); system connections are muted."""
    schema = get_schema(client)
    nodes = [{"id": cls, "label": f"{cls.lstrip('_')} ({info['count']})", "cls": cls,
              "props": {"count": info["count"], "properties": ", ".join(info["properties"]) or "(none)"},
              "snippet": snippet(cls), "count": info["count"]} for cls, info in schema["classes"].items()]
    edges = []
    for c in schema["connections"]:
        if c["src"] in schema["classes"] and c["dst"] in schema["classes"]:
            is_system = c["class"].startswith("_") or c["class"].islower()
            edges.append((c["src"], c["dst"], f"{c['class']} ({c['count']})", SYSTEM_EDGE_COLOR if is_system else None))
    return nodes, edges


def render_schema(client, title="ApertureDB schema", filename="graphs/schema.html", **kwargs):
    """One call: the live schema as a navigable graph - hover a class for its connections, click for
    its properties and a query that lists it."""
    nodes, edges = get_schema_graph(client)
    return render_graph(nodes, edges, title, filename, **kwargs)


# ---------------------------------------------------------------------------------------------
# Neighbourhoods and query results
# ---------------------------------------------------------------------------------------------

def neighborhood(client, cls, constraints=None, depth=1, limit=10, per_class=8, sort=None, include=None,
                 exclude=None, max_nodes=200, label_props=None):
    """Find objects of ``cls`` and expand ``depth`` hops around them -> ``(nodes, edges, start_ids)``."""
    start = find_nodes(client, cls, constraints, limit=limit, label_props=label_props)
    if not start:
        raise ValueError(f"no {cls} objects match {constraints}")
    nodes, edges = expand(client, start, depth=depth, per_class=per_class, sort=sort, include=include,
                          exclude=exclude, max_nodes=max_nodes, label_props=label_props)
    return nodes, edges, [n["id"] for n in start]


def render_neighborhood(client, cls, constraints=None, title=None, filename=None, depth=1, limit=10,
                        per_class=8, sort=None, include=None, exclude=None, max_nodes=200,
                        label_props=None, **render_kwargs):
    """Start from ``cls`` objects matching ``constraints`` and draw them with their surroundings."""
    nodes, edges, start_ids = neighborhood(client, cls, constraints, depth, limit, per_class, sort, include,
                                           exclude, max_nodes, label_props)
    return render_graph(nodes, [(s, d, c) for s, d, c in edges], title or f"{cls} neighbourhood",
                        filename or "graphs/neighborhood.html", client=client,
                        select=start_ids[0] if len(start_ids) == 1 else None, **render_kwargs)


def _start_points(client, query, max_start):
    """Run ``query`` (made to return unique ids, no pixels) and list ``(class, uniqueid)`` of its answers."""
    q = copy.deepcopy(query)
    picks = []
    for i, command in enumerate(q):
        name, body = next(iter(command.items()))
        if name not in _FIND_TO_CLASS and name != "FindEntity":
            continue
        results = body.get("results") or {}
        if results.get("count") and not results.get("list") and not results.get("all_properties"):
            continue
        body["uniqueids"] = True
        if not results.get("list") and not results.get("all_properties"):
            body["results"] = {**results, "list": ["_uniqueid"]}
        if _FIND_TO_CLASS.get(name) in _HAS_BLOB:
            body["blobs"] = False
        picks.append((i, name, body.get("with_class") if name == "FindEntity" else _FIND_TO_CLASS[name]))
    resp, _ = client.query(q)
    if isinstance(resp, dict):
        raise ValueError(resp.get("info"))
    points = []
    for i, name, cls in picks:
        found = resp[i][name].get("entities")
        if isinstance(found, list):
            points += [(cls, e["_uniqueid"]) for e in found if "_uniqueid" in e]
    return list(dict.fromkeys(points))[:max_start]


def render_query(client, query, title="Query result", filename=None, depth=1, per_class=6, sort=None,
                 include=None, exclude=None, max_nodes=150, max_start=30, label_props=None, **render_kwargs):
    """Run any query and draw the objects it returned, highlighted, with their direct neighbours."""
    points = _start_points(client, query, max_start)
    if not points:
        raise ValueError("the query returned no objects")
    start = []
    for cls in dict.fromkeys(c for c, _ in points):
        ids = [u for c, u in points if c == cls]
        resp, _ = client.query([_find(cls, {"_uniqueid": ["in", ids]})])
        start += [_node(e, cls, label_props) for e in list(resp[0].values())[0].get("entities", [])]
    nodes, edges = expand(client, start, depth=depth, per_class=per_class, sort=sort, include=include,
                          exclude=exclude, max_nodes=max_nodes, label_props=label_props)
    return render_graph(nodes, edges, title, filename or "graphs/query.html", client=client,
                        highlight_ids=[n["id"] for n in start], **render_kwargs)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _client(env_file=None):
    try:  # inside the GCS demo folder the demo's connection settings are used
        import config
        return config.get_client(env_file)
    except ImportError:
        from aperturedb.CommonLibrary import create_connector
        return create_connector()


def _parse_where(items, typed=True):
    """``PROP=VALUE`` items -> equality constraints.  VALUE is read as JSON (so 2 is a number)
    when ``typed``; otherwise it stays a string (ids like 10028 are often stored as text)."""
    constraints = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if typed:
            try:
                value = json.loads(value)
            except ValueError:
                pass
        constraints[key] = ["==", value]
    return constraints


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-file", help="file with APERTUREDB_KEY etc. (the GCS demo's config.py is used when present)")
    common.add_argument("--open", action="store_true", help="open the page in a browser")
    common.add_argument("--out", help="output HTML file")
    common.add_argument("--label", action="append", metavar="CLASS=PROP",
                        help="property that names the nodes of a class, e.g. _Image=gcs_object_stem (repeatable)")

    parser = argparse.ArgumentParser(description="Interactive graph views of an ApertureDB database")
    sub = parser.add_subparsers(dest="view", required=True)
    sub.add_parser("schema", parents=[common], help="classes and connections with live counts")
    node = sub.add_parser("node", parents=[common], help="objects of a class and their surroundings")
    node.add_argument("cls", help="class, e.g. _Image or a user class")
    node.add_argument("--where", action="append", metavar="PROP=VALUE", help="equality filter (repeatable)")
    node.add_argument("--depth", type=int, default=1)
    node.add_argument("--limit", type=int, default=5, help="how many starting objects")
    node.add_argument("--per-class", type=int, default=8, help="neighbours kept per node and connection")
    query = sub.add_parser("query", parents=[common], help="the objects a query (JSON file) returns, with their neighbours")
    query.add_argument("file")
    query.add_argument("--depth", type=int, default=1)
    args = parser.parse_args(argv)
    labels = dict(item.split("=", 1) for item in args.label or [])

    client = _client(args.env_file)
    opts = {"open_browser": args.open, "embed": False}
    if args.view == "schema":
        render_schema(client, filename=args.out or "graphs/schema.html", **opts)
    elif args.view == "node":
        for typed in (True, False):  # 10028 as a number, then as text
            try:
                render_neighborhood(client, args.cls, _parse_where(args.where, typed), depth=args.depth,
                                    limit=args.limit, per_class=args.per_class, label_props=labels,
                                    filename=args.out or "graphs/neighborhood.html", **opts)
                break
            except ValueError as exc:
                if not typed:
                    raise SystemExit(str(exc))
    else:
        with open(args.file) as f:
            render_query(client, json.load(f), title=os.path.basename(args.file), depth=args.depth,
                         label_props=labels, filename=args.out or "graphs/query.html", **opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
