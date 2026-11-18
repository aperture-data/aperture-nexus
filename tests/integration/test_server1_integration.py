"""Server 1.0 index, content and native class boundaries without model downloads."""
import importlib.util
from pathlib import Path
import uuid

import numpy as np
import pytest

from aperture_nexus import Context, Information


@pytest.mark.integration
def test_reuses_vector_index_and_retains_customer_metadata(memory_engine, test_principal):
    name = f'migration-{uuid.uuid4().hex[:10]}'
    sid = f'nexus-{name}'
    context = Context(principal=test_principal, session_id=sid)
    vector = np.array([1., 0., 0., 0.], dtype=np.float32)
    descriptor_class = f'nexus_text__{name}'
    selector = {'target': 'descriptor', 'class': descriptor_class,
                'property': '_blob', 'kind': memory_engine._cfg.processing.descriptor_engine.lower()}
    try:
        for text in ('first', 'second'):
            info = Information(context_id=context.id)
            info.log(text=text, embedding=vector, embedding_model=name)
            memory_engine.commit(context, info)
            if text == 'first':
                response, _ = memory_engine._db.query([{'UpdateIndex': {
                    **selector, 'metadata': {'embeddings_model': name,
                                             'modality': 'text', 'fingerprint': 'sha256:caller-owned'},
                }}])
                assert response[0]['UpdateIndex']['status'] == 0
        results = memory_engine.search(
            query=vector, modality='text', embedding_model=name,
            filters={'session_id': sid}, return_content=True,
        )
        assert {result.content for result in results} == {b'first', b'second'}
        response, _ = memory_engine._db.query([{'GetIndexes': selector}])
        assert response[0]['GetIndexes']['indexes'][0]['metadata']['fingerprint'] == 'sha256:caller-owned'
    finally:
        memory_engine.remove(session_id=sid)
        memory_engine._db.query([{'RemoveIndex': selector}])


@pytest.mark.integration
def test_remove_keeps_other_app_native_objects(memory_engine, test_principal):
    sid = f'nexus-scope-{uuid.uuid4().hex[:10]}'
    foreign_class = f'OtherApp{uuid.uuid4().hex[:10]}'
    context = Context(principal=test_principal, session_id=sid)
    db = memory_engine._db
    try:
        response, _ = db.query([{'AddBlob': {'class': foreign_class,
                                             'properties': {'session_id': sid}}}], [b'other application'])
        assert response[0]['AddBlob']['status'] == 0
        info = Information(context_id=context.id)
        info.log(text='Nexus content')
        memory_engine.commit(context, info)
        memory_engine.remove(session_id=sid)
        response, blobs = db.query([{'FindBlob': {
            'with_class': foreign_class, 'constraints': ['$session_id', '==', sid],
            'blobs': True,
        }}])
        assert response[0]['FindBlob']['status'] == 0
        assert blobs == [b'other application']
    finally:
        db.query([{'DeleteBlob': {'with_class': foreign_class}}])


@pytest.mark.integration
def test_graph_reads_typed_schema_and_scoped_neighbours(memory_engine, test_principal):
    path = Path(__file__).parents[2] / 'demo' / 'webapp' / 'graph_viz.py'
    spec = importlib.util.spec_from_file_location('nexus_graph_live', path)
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    sid = f'nexus-graph-{uuid.uuid4().hex[:10]}'
    context = Context(principal=test_principal, session_id=sid)
    try:
        info = Information(context_id=context.id)
        info.log(text='Graph payload')
        memory_engine.commit(context, info)
        schema = graph.get_schema(memory_engine._db)
        assert 'blob:NexusBlob' in schema['classes']
        nodes = graph.find_nodes(memory_engine._db, 'NexusContext',
                                 ['$nexus_ctx_id', '==', context.id])
        expanded, edges = graph.expand(memory_engine._db, nodes, schema=schema)
        assert any(node['cls'] == 'blob:NexusBlob' for node in expanded)
        assert edges
        payload = next(node for node in expanded if node['cls'] == 'blob:NexusBlob')
        assert graph.get_payload_uri(
            memory_engine._db, payload['cls'], payload['id']).startswith('data:')
    finally:
        memory_engine.remove(session_id=sid)


@pytest.mark.integration
def test_session_ids_starting_with_dollar_are_literal(memory_engine, test_principal):
    sid = f'$nexus-literal-{uuid.uuid4().hex[:10]}'
    context = Context(principal=test_principal, session_id=sid)
    try:
        info = Information(context_id=context.id)
        info.log(text='Literal ID content')
        memory_engine.commit(context, info)
        results = memory_engine.search(filters={'session_id': sid})
        assert len(results) == 1
        assert results[0].session_id == sid
    finally:
        memory_engine.remove(session_id=sid)
