import paddle

import paddle_geometric.typing
from paddle_geometric.nn import FeaStConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_feast_conv():
    x1 = paddle.randn(shape=[4, 16])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = FeaStConv(16, 32, heads=2)
    assert str(conv) == 'FeaStConv(16, 32, heads=2)'

    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert tuple(out.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        assert paddle.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)
