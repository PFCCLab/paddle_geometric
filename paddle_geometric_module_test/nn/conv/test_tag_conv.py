import paddle

import paddle_geometric.typing
from paddle_geometric.nn import TAGConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_tag_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = TAGConv(16, 32)
    assert str(conv) == 'TAGConv(16, 32, K=3)'

    out1 = conv(x, edge_index)
    assert tuple(out1.shape) == (4, 32)
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert tuple(out2.shape) == (4, 32)
    assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert paddle.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out1, atol=1e-6)
        assert paddle.allclose(jit(x, edge_index, value), out2, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj3.t()), out1, atol=1e-6)
            assert paddle.allclose(jit(x, adj4.t()), out2, atol=1e-6)


def test_static_tag_conv():
    x = paddle.randn(shape=[3, 4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = TAGConv(16, 32)
    out = conv(x, edge_index)
    assert tuple(out.shape) == (3, 4, 32)
