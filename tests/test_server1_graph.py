"""Graph identities and native Find queries without a class selector."""
import importlib.util
import io
import os
from pathlib import Path
import uuid
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).parents[1]


def graph_module(relative="demo/webapp/graph_viz.py"):
    spec = importlib.util.spec_from_file_location(
        "nexus_graph_review_" + uuid.uuid4().hex, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", ["graph_viz.py", "demo/webapp/graph_viz.py"])
def test_schema_keeps_reserved_prefix_entity_classes_distinct(relative):
    graph = graph_module(relative)
    client = MagicMock()
    client.query.return_value = ([{"GetSchema": {"status": 0, "nodes": [
        {"type": "entity", "class": "image:Same", "matched": 1, "properties": {}},
        {"type": "image", "class": "Same", "matched": 1, "properties": {}},
    ], "edges": [{"class": {"src": "image:Same", "connection": "ShownIn", "dst": "Same"},
                  "target": {"src": "entity", "dst": "image"}, "matched": 1}]}}], [])
    nodes, edges = graph.get_schema_graph(client)
    assert {node["id"] for node in nodes} == {"entity:image:Same", "image:Same"}
    assert edges[0][:2] == ("entity:image:Same", "image:Same")


def test_unscoped_find_queries_take_each_returned_objects_class():
    graph = graph_module()
    client = MagicMock()
    client.query.return_value = ([
        {"FindImage": {"status": 0, "entities": [
            {"_class": "Photos", "_uniqueid": "1.0.1"},
            {"_class": "Scans", "_uniqueid": "1.0.2"}]}},
        {"FindEntity": {"status": 0, "entities": [
            {"_class": "image:Photos", "_uniqueid": "2.0.1"}]}},
        {"FindDescriptorBatch": {"status": 0, "entities": [
            {"_class": "Vectors", "_uniqueid": "3.0.1"}]}},
    ], [])
    query = [{"FindImage": {}}, {"FindEntity": {}}, {"FindDescriptorBatch": {}}]
    assert graph._start_points(client, query, 30) == [
        ("image:Photos", "1.0.1"), ("image:Scans", "1.0.2"),
        ("entity:image:Photos", "2.0.1"), ("descriptor:Vectors", "3.0.1")]
    assert query == [{"FindImage": {}}, {"FindEntity": {}}, {"FindDescriptorBatch": {}}]
    assert graph._find("entity:image:Photos")["FindEntity"]["with_class"] == "image:Photos"


def test_scoped_query_can_use_its_selector_when_result_omits_class():
    graph = graph_module()
    client = MagicMock()
    client.query.return_value = ([{"FindEntity": {"status": 0, "entities": [
        {"_uniqueid": "1.0.1"}]}}], [])
    assert graph._start_points(client, [{"FindEntity": {"with_class": "NexusContext"}}], 30) == [
        ("NexusContext", "1.0.1")]
    with pytest.raises(ValueError, match="missing its _class"):
        graph._start_points(client, [{"FindEntity": {}}], 30)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("NEXUS_TEST_PORT"), reason="Set NEXUS_TEST_PORT for a live native graph test")
def test_live_unscoped_find_keeps_reserved_entity_and_image_classes_distinct():
    from aperturedb.Connector import Connector
    from PIL import Image

    graph = graph_module()
    name = "GraphReview" + uuid.uuid4().hex[:10]
    entity_class = "image:" + name
    client = Connector(host=os.getenv("NEXUS_TEST_HOST", "127.0.0.1"),
                       port=int(os.environ["NEXUS_TEST_PORT"]), user="admin", password="admin", use_ssl=False)
    output = io.BytesIO()
    Image.new("RGB", (2, 3), (10, 20, 30)).save(output, format="PNG")
    selector = ["$graph_review", "==", name]
    try:
        response, _ = client.query([
            {"AddEntity": {"class": entity_class, "_ref": 1, "properties": {"graph_review": name}}},
            {"AddImage": {"class": name, "_ref": 2, "format": "png", "properties": {"graph_review": name}}},
            {"AddConnection": {"class": name, "src": 1, "dst": 2}},
        ], [output.getvalue()])
        assert all(next(iter(command.values()))["status"] == 0 for command in response), response
        points = graph._start_points(client, [{"FindEntity": {"constraints": selector}},
                                              {"FindImage": {"constraints": selector}}], 30)
        assert {cls for cls, _ in points} == {"entity:" + entity_class, "image:" + name}
        schema = graph.get_schema(client)
        assert "entity:" + entity_class in schema["classes"]
        assert "image:" + name in schema["classes"]
        nodes = graph.find_nodes(client, "entity:" + entity_class, selector)
        assert len(nodes) == 1 and nodes[0]["cls"] == "entity:" + entity_class
        expanded, edges = graph.expand(client, nodes, schema=schema)
        assert {node["cls"] for node in expanded} == {"entity:" + entity_class, "image:" + name}
        assert edges
    finally:
        client.query([{"DeleteEntity": {"with_class": entity_class}},
                     {"DeleteImage": {"with_class": name}}])
        client.query([{"RemoveClass": {"target": {"src": "entity", "dst": "image"},
                                       "class": {"src": entity_class, "connection": name, "dst": name}}},
                      {"RemoveClass": {"target": "entity", "class": entity_class}},
                      {"RemoveClass": {"target": "image", "class": name}}])
        client.close()
