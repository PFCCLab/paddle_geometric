import paddle
from paddle.nn import Linear, ReLU, Sequential

import paddle_geometric.typing
from paddle_geometric.nn import PointTransformerConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_point_transformer_conv():
    x1 = paddle.rand([4, 16])
    x2 = paddle.randn(shape=[2, 8])
    pos1 = paddle.rand([4, 3])
    pos2 = paddle.randn(shape=[2, 3])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert str(conv) == 'PointTransformerConv(16, 32)'

    out = conv(x1, pos1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, pos1, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, pos1, adj2.t()), out, atol=1e-6)

    pos_nn = Sequential(Linear(3, 16), ReLU(), Linear(16, 32))
    attn_nn = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
    conv = PointTransformerConv(16, 32, pos_nn, attn_nn)

    out = conv(x1, pos1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test biparitite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = PointTransformerConv((16, 8), 32)
    assert str(conv) == 'PointTransformerConv((16, 8), 32)'

    out = conv((x1, x2), (pos1, pos2), edge_index)
    assert tuple(out.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), (pos1, pos2), adj1.t()), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), (pos1, pos2), adj2.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), (pos1, pos2), edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), (pos1, pos2), adj2.t()), out)
