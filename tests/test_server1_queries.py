"""Validate the server 1.0 query identities and index safety checks."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aperture_nexus import Memory
from aperture_nexus.exceptions import NexusStorageError


def _graph_module():
    path = Path(__file__).parents[1] / 'demo' / 'webapp' / 'graph_viz.py'
    spec = importlib.util.spec_from_file_location('nexus_graph_v1_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_descriptor_index_metadata_and_explicit_engine(mock_connector):
    mock_connector.query.side_effect = [
        ([{'GetIndexes': {'status': 0, 'indexes': []}}], []),
        ([{'CreateIndex': {'status': 0}}], []),
    ]
    memory = Memory(db_client=mock_connector)
    memory._ensure_descriptor_set('nexus_text__model', np.ones(4), 'text', 'model', 'CS', 'HNSW')
    query = mock_connector.query.call_args_list[1].args[0][0]['CreateIndex']
    assert query == {
        'target': 'descriptor', 'class': 'nexus_text__model',
        'property': '_blob', 'kind': 'hnsw',
        'params': {'dimensions': 4, 'metric': 'CS'},
        'metadata': {'embeddings_model': 'model', 'modality': 'text'},
    }


@pytest.mark.parametrize('kind,params,metadata', [
    ('hnsw', {'dimensions': 8}, {}),
    ('hnsw', {'metric': ['L2']}, {}),
    ('flat', {}, {}),
    ('hnsw', {}, {'embeddings_model': 'another-model'}),
    ('hnsw', {}, {'modality': 'image'}),
])
def test_existing_descriptor_configuration_is_not_silently_replaced(mock_connector, kind, params, metadata):
    mock_connector.query.return_value = ([{'GetIndexes': {'status': 0, 'indexes': [
        {'kind': kind, 'params': params, 'metadata': metadata},
    ]}}], [])
    memory = Memory(db_client=mock_connector)
    with pytest.raises(NexusStorageError):
        memory._ensure_descriptor_set('nexus_text__model', np.ones(4),
                                      'text', 'model', 'CS', 'HNSW')
    assert mock_connector.query.call_count == 1


def test_existing_index_accepts_server_metric_array_and_retains_metadata(mock_connector):
    metadata = {'embeddings_model': 'model', 'modality': 'text', 'fingerprint': 'sha256:test'}
    mock_connector.query.return_value = ([{'GetIndexes': {'status': 0, 'indexes': [{
        'kind': 'hnsw', 'params': {'dimensions': 4, 'metric': ['CS']},
        'metadata': metadata,
    }]}}], [])
    Memory(db_client=mock_connector)._ensure_descriptor_set(
        'nexus_text__model', np.ones(4), 'text', 'model', 'CS', 'HNSW')
    assert mock_connector.query.call_count == 1
    assert metadata['fingerprint'] == 'sha256:test'


def test_metadata_search_scopes_native_types_and_limit(mock_connector):
    mock_connector.query.return_value = ([{'FindImage': {'status': 0, 'entities': []}}], [])
    memory = Memory(db_client=mock_connector)
    memory.search(filters={'session_id': 's'}, modality='image', k=2)
    assert mock_connector.query.call_args.args[0] == [{'FindImage': {
        'with_class': 'NexusImage', 'limit': 2,
        'results': {'all_properties': True}, 'constraints': ['$session_id', '==', 's'],
    }}]


def test_typed_graph_schema_keeps_equal_class_names_distinct():
    graph = _graph_module()
    client = MagicMock()
    client.query.return_value = ([{'GetSchema': {'status': 0, 'nodes': [
        {'type': 'entity', 'class': 'Same', 'matched': 1, 'properties': {}},
        {'type': 'image', 'class': 'Same', 'matched': 2, 'properties': {}},
    ], 'edges': [{'class': 'ShownIn', 'src_type': 'entity', 'src_class': 'Same',
                  'dst_type': 'image', 'dst_class': 'Same', 'matched': 2}]}}], [])
    schema = graph.get_schema(client)
    assert set(schema['classes']) == {'Same', 'image:Same'}
    assert schema['connections'][0]['dst'] == 'image:Same'
    assert graph._find('image:Same')['FindImage']['with_class'] == 'Same'
    assert 'FindDescriptorSet' not in graph.NATIVE_FIND.values()


def test_graph_where_uses_expression_constraints():
    graph = _graph_module()
    assert graph._parse_where(['session_id=s', 'index=2']) == [
        'all', ['$session_id', '==', 's'], ['$index', '==', 2],
    ]


def test_filter_values_remain_literal_application_data():
    assert Memory._build_constraints({'session_id': '$special'}) == [
        '$session_id', '==', '$$special',
    ]
    graph = _graph_module()
    assert graph._parse_where(['session_id=$special']) == ['$session_id', '==', '$$special']
    assert graph._find('NexusContext', ['$session_id', '==', '$$special'])
