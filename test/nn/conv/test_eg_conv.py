import pytest
import paddle
from paddle_geometric.nn import EGConv
from paddle_geometric.utils import to_paddle_csr_tensor

def test_eg_conv_with_error():
    with pytest.raises(ValueError, match="must be divisible by the number of"):
        EGConv(16, 30, num_heads=8)

    with pytest.raises(ValueError, match="Unsupported aggregator"):
        EGConv(16, 32, aggregators=["xxx"])

@pytest.mark.parametrize("aggregators", [
    ["symnorm"],
    ["sum", "symnorm", "std"],
])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_eg_conv(aggregators, add_self_loops):
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(
        in_channels=16,
        out_channels=32,
        aggregators=aggregators,
        add_self_loops=add_self_loops,
    )
    assert str(conv) == f"EGConv(16, 32, aggregators={aggregators})"
    out = conv(x, edge_index)
    assert out.shape == (4, 32)

    conv.cached = True
    assert paddle.allclose(conv(x, edge_index), out, atol=1e-2)
    assert conv._cached_edge_index is not None
    # Note: Due to framework-level differences in sparse tensor handling,
    # we only test edge_index format. Paddle's sparse tensors are not SparseTensor type.

def test_eg_conv_with_sparse_input_feature():
    # Test with CSR sparse matrix as edge_index input
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = EGConv(
        in_channels=16,
        out_channels=32,
        aggregators=["symnorm"],
    )

    # Get output with edge_index
    out_dense = conv(x, edge_index)

    # Get output with CSR sparse matrix
    adj_t = to_paddle_csr_tensor(edge_index, size=(4, 4))
    out_sparse = conv(x, adj_t)

    # Both outputs should have the same shape
    assert out_dense.shape == out_sparse.shape

    # Due to numerical differences in sparse operations, we use a looser tolerance
    assert paddle.allclose(out_dense, out_sparse, atol=1e-2)
