import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MFConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


def test_mf_conv():
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MFConv(8, 32)
    assert str(conv) == 'MFConv(8, 32)'
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out)
        assert paddle.allclose(jit(x1, edge_index, size=(4, 4)), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out)

    # Test bipartite message passing:
    conv = MFConv((8, 16), 32)
    assert str(conv) == 'MFConv((8, 16), 32)'

    out1 = conv((x1, x2), edge_index)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, (4, 2))
    assert tuple(out2.shape)== (2, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out1)
        assert paddle.allclose(conv((x1, None), adj.t()), out2)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert paddle.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out1)
            assert paddle.allclose(jit((x1, None), adj.t()), out2)
