import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import GENConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_coo_tensor


@pytest.mark.parametrize('aggr', [
    'softmax',
    'powermean',
    ['softmax', 'powermean'],
])
def test_gen_conv(aggr):
    x1 = paddle.randn(shape=[4, 16])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn(shape=[edge_index.shape[1], 16])
    adj1 = to_paddle_coo_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_coo_tensor(edge_index, value, size=(4, 4))

    conv = GENConv(16, 32, aggr, edge_dim=16, msg_norm=True)
    assert str(conv) == f'GENConv(16, 32, aggr={aggr})'
    out1 = conv(x1, edge_index)
    assert tuple(out1.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out1)
    assert paddle.allclose(conv(x1, adj1.t().coalesce()), out1)

    out2 = conv(x1, edge_index, value)
    assert tuple(out2.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, value, (4, 4)), out2)
    # t() expects a tensor with <= 2 sparse and 0 dense dimensions
    assert paddle.allclose(conv(x1, adj2.transpose(1, 0).coalesce()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x1, adj3.t()), out1, atol=1e-4)
        assert paddle.allclose(conv(x1, adj4.t()), out2, atol=1e-4)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out1, atol=1e-4)
        assert paddle.allclose(jit(x1, edge_index, size=(4, 4)), out1,
                              atol=1e-4)
        assert paddle.allclose(jit(x1, edge_index, value), out2, atol=1e-4)
        assert paddle.allclose(jit(x1, edge_index, value, size=(4, 4)), out2,
                              atol=1e-4)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj3.t()), out1, atol=1e-4)
            assert paddle.allclose(jit(x1, adj4.t()), out2, atol=1e-4)

    # Test bipartite message passing:
    adj1 = to_paddle_coo_tensor(edge_index, size=(4, 2))
    adj2 = to_paddle_coo_tensor(edge_index, value, size=(4, 2))

    out1 = conv((x1, x2), edge_index)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    assert paddle.allclose(conv((x1, x2), adj1.t().coalesce()), out1)

    out2 = conv((x1, x2), edge_index, value)
    assert tuple(out2.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out2)
    assert paddle.allclose(conv((x1, x2),
                               adj2.transpose(1, 0).coalesce()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj3.t()), out1, atol=1e-4)
        assert paddle.allclose(conv((x1, x2), adj4.t()), out2, atol=1e-4)

    if is_full_test():
        assert paddle.allclose(jit((x1, x2), edge_index), out1, atol=1e-4)
        assert paddle.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1,
                              atol=1e-4)
        assert paddle.allclose(jit((x1, x2), edge_index, value), out2,
                              atol=1e-4)
        assert paddle.allclose(jit((x1, x2), edge_index, value, (4, 2)), out2,
                              atol=1e-4)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj3.t()), out1, atol=1e-4)
            assert paddle.allclose(jit((x1, x2), adj4.t()), out2, atol=1e-4)

    # Test bipartite message passing with unequal feature dimensions:
    conv.reset_parameters()
    assert float(conv.msg_norm.scale) == 1

    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])

    conv = GENConv((8, 16), 32, aggr)
    assert str(conv) == f'GENConv((8, 16), 32, aggr={aggr})'

    out1 = conv((x1, x2), edge_index)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    assert paddle.allclose(conv((x1, x2), adj1.t().coalesce()), out1)

    out2 = conv((x1, None), edge_index, size=(4, 2))
    assert tuple(out2.shape)== (2, 32)
    assert paddle.allclose(conv((x1, None), adj1.t().coalesce()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv((x1, x2), adj3.t()), out1, atol=1e-4)
        assert paddle.allclose(conv((x1, None), adj3.t()), out2, atol=1e-4)

    # Test lazy initialization:
    conv = GENConv((-1, -1), 32, aggr, edge_dim=-1)
    assert str(conv) == f'GENConv((-1, -1), 32, aggr={aggr})'
    out1 = conv((x1, x2), edge_index, value)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, value, size=(4, 2)), out1)
    assert paddle.allclose(conv((x1, x2),
                               adj2.transpose(1, 0).coalesce()), out1)

    out2 = conv((x1, None), edge_index, value, size=(4, 2))
    assert tuple(out2.shape)== (2, 32)
    assert paddle.allclose(conv((x1, None),
                               adj2.transpose(1, 0).coalesce()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv((x1, x2), adj4.t()), out1, atol=1e-4)
        assert paddle.allclose(conv((x1, None), adj4.t()), out2, atol=1e-4)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index, value), out1,
                              atol=1e-4)
        assert paddle.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1, atol=1e-4)
        assert paddle.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2, atol=1e-4)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj4.t()), out1, atol=1e-4)
            assert paddle.allclose(jit((x1, None), adj4.t()), out2, atol=1e-4)
