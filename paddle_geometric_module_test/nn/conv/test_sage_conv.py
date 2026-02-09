import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MLPAggregation, SAGEConv
from paddle_geometric.testing import (
    assert_module,
    is_full_test,
    onlyLinux,
    withDevice,
    withPackage,
)
from paddle_geometric.typing import SparseTensor


@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('aggr', ['mean', 'sum'])
def test_sage_conv(project, aggr):
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(8, 32, project=project, aggr=aggr)
    assert str(conv) == f'SAGEConv(8, 32, aggr={aggr})'

    out = assert_module(conv, x, edge_index, expected_size=(4, 32))

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)
        assert paddle.allclose(jit(x, edge_index, size=(4, 4)), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])

    conv = SAGEConv((8, 16), 32, project=project, aggr=aggr)
    assert str(conv) == f'SAGEConv((8, 16), 32, aggr={aggr})'

    out1 = assert_module(conv, (x1, x2), edge_index, expected_size=(2, 32))
    out2 = assert_module(conv, (x1, None), edge_index, size=(4, 2),
                         expected_size=(2, 32))

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index), out1, atol=1e-6)
        assert paddle.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert paddle.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
            assert paddle.allclose(jit((x1, x2), adj.t()), out1, atol=1e-6)
            assert paddle.allclose(jit((x1, None), adj.t()), out2, atol=1e-6)


@pytest.mark.parametrize('project', [False, True])
def test_lazy_sage_conv(project):
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    if project:
        with pytest.raises(ValueError, match="does not support lazy"):
            SAGEConv(-1, 32, project=project)
    else:
        conv = SAGEConv(-1, 32, project=project)
        assert str(conv) == 'SAGEConv(-1, 32, aggr=mean)'

        out = conv(x, edge_index)
        assert tuple(out.shape)== (4, 32)


def test_lstm_aggr_sage_conv():
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(8, 32, aggr='lstm')
    assert str(conv) == 'SAGEConv(8, 32, aggr=lstm)'

    assert_module(conv, x, edge_index, expected_size=(4, 32),
                  test_edge_permutation=False)

    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 0]])
    with pytest.raises(ValueError, match="'index' tensor is not sorted"):
        conv(x, edge_index)


def test_mlp_sage_conv():
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(
        in_channels=8,
        out_channels=32,
        aggr=MLPAggregation(
            in_channels=8,
            out_channels=8,
            max_num_elements=2,
            num_layers=1,
        ),
    )

    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 32)


@pytest.mark.parametrize('aggr_kwargs', [
    dict(mode='cat'),
    dict(mode='proj', mode_kwargs=dict(in_channels=8, out_channels=16)),
    dict(mode='attn', mode_kwargs=dict(in_channels=8, out_channels=16,
                                       num_heads=4)),
    dict(mode='sum'),
])
def test_multi_aggr_sage_conv(aggr_kwargs):
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    aggr_kwargs['aggrs_kwargs'] = [{}, {}, {}, dict(learn=True, t=1)]
    conv = SAGEConv(8, 32, aggr=['mean', 'max', 'sum', 'softmax'],
                    aggr_kwargs=aggr_kwargs)

    assert_module(conv, x, edge_index, expected_size=(4, 32))


@withDevice
@onlyLinux
@withPackage('paddle>=2.1.0')
def test_compile_multi_aggr_sage_conv(device):
    try:
        import paddle._dynamo as dynamo
    except Exception:
        import pytest
        pytest.skip("paddle._dynamo is not available")

    x = paddle.randn([4, 8], device=device)
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]], place=device)

    conv = SAGEConv(
        in_channels=8,
        out_channels=32,
        aggr=['mean', 'sum', 'min', 'max', 'std'],
    ).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    compiled_conv = paddle.compile(conv)

    expected = conv(x, edge_index)
    out = compiled_conv(x, edge_index)
    assert paddle.allclose(out, expected, atol=1e-6)
