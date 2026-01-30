import paddle

import paddle_geometric.typing
from paddle_geometric.nn import APPNP
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_appnp():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = APPNP(K=3, alpha=0.1, cached=True)
    assert str(conv) == 'APPNP(K=3, alpha=0.1)'
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 16)
    assert paddle.allclose(conv(x, adj1.t()), out)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out)

    # Run again to test the cached functionality:
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, edge_index), conv(x, adj1.t()))
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert conv._cached_adj_t is not None
        assert paddle.allclose(conv(x, edge_index), conv(x, adj2.t()))

    conv.reset_parameters()
    assert conv._cached_edge_index is None
    assert conv._cached_adj_t is None

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj2.t()), out)


def test_appnp_dropout():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    # With dropout probability of 1.0, the final output equals to alpha * x:
    conv = APPNP(K=2, alpha=0.1, dropout=1.0)
    assert paddle.allclose(0.1 * x, conv(x, edge_index))
    # Skip sparse matrix tests on CPU as they are not supported
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        assert paddle.allclose(0.1 * x, conv(x, adj1.t()))

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(0.1 * x, conv(x, adj2.t()))
