import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

import paddle_geometric.typing
from paddle_geometric.nn import GINConv, GINEConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_gin_conv():
    x1 = paddle.randn(shape=[4, 16])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINConv(nn, train_eps=True)
    assert str(conv) == (
        'GINConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=32, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, dtype=float32)\n'
        '))')
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out, atol=1e-6)
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out, atol=1e-6)
        assert paddle.allclose(jit(x1, edge_index, size=(4, 4)), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out1 = conv((x1, x2), edge_index)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, (4, 2)), out1, atol=1e-6)
    assert paddle.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-6)

    out2 = conv((x1, None), edge_index, (4, 2))
    assert tuple(out2.shape)== (2, 32)
    assert paddle.allclose(conv((x1, None), adj1.t()), out2, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj2.t()), out1, atol=1e-6)
        assert paddle.allclose(conv((x1, None), adj2.t()), out2, atol=1e-6)

    if is_full_test():
        assert paddle.allclose(jit((x1, x2), edge_index), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert paddle.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out1)
            assert paddle.allclose(jit((x1, None), adj2.t()), out2)


def test_gine_conv():
    x1 = paddle.randn(shape=[4, 16])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn(shape=[edge_index.shape[1], 16])

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True)
    assert str(conv) == (
        'GINEConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=32, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, dtype=float32)\n'
        '))')
    out = conv(x1, edge_index, value)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index, value), out)
        assert paddle.allclose(jit(x1, edge_index, value, size=(4, 4)), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out)

    # Test bipartite message passing:
    out1 = conv((x1, x2), edge_index, value)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert tuple(out2.shape)== (2, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out1)
        assert paddle.allclose(conv((x1, None), adj.t()), out2)

    if is_full_test():
        assert paddle.allclose(jit((x1, x2), edge_index, value), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1)
        assert paddle.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out1)
            assert paddle.allclose(jit((x1, None), adj.t()), out2)


def test_gine_conv_edge_dim():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 8])

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True, edge_dim=8)
    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 32)

    nn = Lin(16, 32)
    conv = GINEConv(nn, train_eps=True, edge_dim=8)
    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 32)


def test_static_gin_conv():
    x = paddle.randn(shape=[3, 4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINConv(nn, train_eps=True)
    out = conv(x, edge_index)
    assert tuple(out.shape)== (3, 4, 32)


def test_static_gine_conv():
    x = paddle.randn(shape=[3, 4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 16])

    nn = Seq(Lin(16, 32), ReLU(), Lin(32, 32))
    conv = GINEConv(nn, train_eps=True)
    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (3, 4, 32)
