import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import EGConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_eg_conv_with_error():
    with pytest.raises(ValueError, match="must be divisible by the number of"):
        EGConv(16, 30, num_heads=8)

    with pytest.raises(ValueError, match="Unsupported aggregator"):
        EGConv(16, 32, aggregators=['xxx'])


@pytest.mark.parametrize('aggregators', [
    ['symnorm'],
    ['sum', 'symnorm', 'std'],
])
@pytest.mark.parametrize('add_self_loops', [True, False])
def test_eg_conv(aggregators, add_self_loops):
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = EGConv(
        in_channels=16,
        out_channels=32,
        aggregators=aggregators,
        add_self_loops=add_self_loops,
    )
    assert str(conv) == f"EGConv(16, 32, aggregators={aggregators})"
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-2)

    conv.cached = True
    assert paddle.allclose(conv(x, edge_index), out, atol=1e-2)
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, edge_index), out, atol=1e-2)
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-2)
        assert conv._cached_adj_t is not None
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-2)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj2.t()), out, atol=1e-2)


def test_eg_conv_with_sparse_input_feature():
    x = paddle.randn(shape=[4, 16]).to_sparse_coo(2)
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(16, 32)
    assert tuple(conv(x, edge_index).shape)== (4, 32)
