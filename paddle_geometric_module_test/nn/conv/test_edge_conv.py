import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

import paddle_geometric.typing
from paddle_geometric.nn import DynamicEdgeConv, EdgeConv
from paddle_geometric.testing import is_full_test, withPackage
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_edge_conv_conv():
    x1 = paddle.randn(shape=[4, 16])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = EdgeConv(nn)
    assert str(conv) == (
        'EdgeConv(nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=16, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=32, dtype=float32)\n'
        '))')
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv((x1, x1), edge_index), out, atol=1e-6)
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)
    assert paddle.allclose(conv((x1, x1), adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)
        assert paddle.allclose(conv((x1, x1), adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out, atol=1e-6)
        assert paddle .allclose(jit((x1, x1), edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)
            assert paddle.allclose(jit((x1, x1), adj2.t()), out, atol=1e-6)

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


@withPackage('paddle_cluster')
def test_dynamic_edge_conv():
    x1 = paddle.randn(shape=[8, 16])
    x2 = paddle.randn(shape=[4, 16])
    batch1 = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = paddle.to_tensor([0, 0, 1, 1])

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = DynamicEdgeConv(nn, k=2)
    assert str(conv) == (
        'DynamicEdgeConv(nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=16, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=32, dtype=float32)\n'
        '), k=2)')
    out11 = conv(x1)
    assert tuple(out11.shape)== (8, 32)

    out12 = conv(x1, batch1)
    assert tuple(out12.shape)== (8, 32)

    out21 = conv((x1, x2))
    assert tuple(out21.shape)== (4, 32)

    out22 = conv((x1, x2), (batch1, batch2))
    assert tuple(out22.shape)== (4, 32)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1), out11)
        assert paddle.allclose(jit(x1, batch1), out12)
        assert paddle.allclose(jit((x1, x2)), out21)
        assert paddle.allclose(jit((x1, x2), (batch1, batch2)), out22)
