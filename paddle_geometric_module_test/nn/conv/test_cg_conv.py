import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import CGConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('batch_norm', [False, True])
def test_cg_conv(batch_norm):
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = CGConv(8, batch_norm=batch_norm)
    assert str(conv) == 'CGConv(8, dim=0)'
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 8)
    assert paddle.allclose(conv(x1, adj1.t()), out)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = CGConv((8, 16))
    assert str(conv) == 'CGConv((8, 16), dim=0)'
    out = conv((x1, x2), edge_index)
    assert tuple(out.shape)== (2, 16)
    assert paddle.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)


def test_cg_conv_with_edge_features():
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])

    conv = CGConv(8, dim=3)
    assert str(conv) == 'CGConv(8, dim=3)'
    out = conv(x1, edge_index, value)
    assert tuple(out.shape)== (4, 8)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index, value), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out)

    # Test bipartite message passing:
    conv = CGConv((8, 16), dim=3)
    assert str(conv) == 'CGConv((8, 16), dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert tuple(out.shape)== (2, 16)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index, value), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out)
