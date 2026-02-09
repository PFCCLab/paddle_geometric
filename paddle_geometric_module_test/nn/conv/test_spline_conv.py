import warnings

import paddle

import paddle_geometric.typing
from paddle_geometric.nn import SplineConv
from paddle_geometric.testing import is_full_test, withPackage
from paddle_geometric.typing import SparseTensor


@withPackage('paddle_spline_conv')
def test_spline_conv():
    warnings.filterwarnings('ignore', '.*non-optimized CPU version.*')

    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])

    conv = SplineConv(8, 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv(8, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index, value), out, atol=1e-6)
        assert paddle.allclose(jit(x1, edge_index, value, size=(4, 4)), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    conv = SplineConv((8, 16), 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv((8, 16), 32, dim=3)'

    out1 = conv((x1, x2), edge_index, value)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert tuple(out2.shape)== (2, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out1, atol=1e-6)
        assert paddle.allclose(conv((x1, None), adj.t()), out2, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index, value), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1, atol=1e-6)
        assert paddle.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out1, atol=1e-6)
            assert paddle.allclose(jit((x1, None), adj.t()), out2, atol=1e-6)


@withPackage('paddle_spline_conv')
def test_lazy_spline_conv():
    warnings.filterwarnings('ignore', '.*non-optimized CPU version.*')

    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])

    conv = SplineConv(-1, 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv(-1, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert tuple(out.shape)== (4, 32)

    conv = SplineConv((-1, -1), 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv((-1, -1), 32, dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert tuple(out.shape)== (2, 32)
