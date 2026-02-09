import paddle

import paddle_geometric.typing
from paddle_geometric.nn import WLConvContinuous
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


def test_wl_conv():
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=paddle.int64)
    x = paddle.to_tensor([[-1], [0], [1]], dtype=paddle.float)

    conv = WLConvContinuous()
    assert str(conv) == 'WLConvContinuous()'

    out = conv(x, edge_index)
    assert out.tolist() == [[-0.5], [0.0], [0.5]]

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
        assert paddle.allclose(conv(x, adj.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out)
        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = paddle.randn(shape=[edge_index.shape[1]])

    out1 = conv((x1, None), edge_index, edge_weight, size=(4, 2))
    assert tuple(out1.shape)== (2, 8)

    out2 = conv((x1, x2), edge_index, edge_weight)
    assert tuple(out2.shape)== (2, 8)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, (4, 2))
        assert paddle.allclose(conv((x1, None), adj.t()), out1)
        assert paddle.allclose(conv((x1, x2), adj.t()), out2)

    if is_full_test():
        assert paddle.allclose(
            jit((x1, None), edge_index, edge_weight, size=(4, 2)), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, edge_weight), out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, None), adj.t()), out1, atol=1e-6)
            assert paddle.allclose(jit((x1, x2), adj.t()), out2, atol=1e-6)
