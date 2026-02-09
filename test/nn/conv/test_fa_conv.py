from typing import Tuple

import paddle
from paddle import Tensor

from paddle_geometric.nn import FAConv
from paddle_geometric.utils import to_paddle_csr_tensor


def test_fa_conv():
    x = paddle.randn([4, 16])
    x_0 = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csr_tensor(edge_index, size=(4, 4))

    conv = FAConv(16, eps=1.0, cached=True)
    assert str(conv) == 'FAConv(16, eps=1.0)'
    out = conv(x, x_0, edge_index)
    assert conv._cached_edge_index is not None
    assert out.shape == [4, 16]

    # Test with CSR sparse matrix
    # Note: Paddle's CSR format is already in the correct layout, no need to transpose
    # Note: In Paddle, CSR sparse matrices are paddle.Tensor type, so they are cached in _cached_edge_index
    assert paddle.allclose(conv(x, x_0, adj1), out, atol=1e-6)
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, x_0, adj1), out, atol=1e-6)

    conv.reset_parameters()
    assert conv._cached_edge_index is None
    assert conv._cached_adj_t is None

    # Test without caching:
    conv.cached = False
    out = conv(x, x_0, edge_index)
    assert paddle.allclose(conv(x, x_0, adj1), out, atol=1e-6)

    # Test `return_attention_weights`:
    result = conv(x, x_0, edge_index, return_attention_weights=True)
    assert paddle.allclose(result[0], out, atol=1e-6)
    assert result[1][0].shape == [2, 10]
    assert result[1][1].shape == [10]
    assert conv._alpha is None

    result = conv(x, x_0, adj1, return_attention_weights=True)
    assert paddle.allclose(result[0], out, atol=1e-6)
    # Note: In Paddle, gcn_norm converts CSR sparse matrices to edge_index format,
    # so return_attention_weights returns edge_index format, not sparse matrix format
    assert result[1][0].shape == [2, 10]
    assert result[1][1].shape == [10]
    assert conv._alpha is None

    # Test with eps=0
    conv_eps0 = FAConv(16, eps=0.0)
    out_eps0 = conv_eps0(x, x_0, edge_index)
    assert out_eps0.shape == [4, 16]

    # Test with non-zero eps
    conv_eps1 = FAConv(16, eps=0.5)
    out_eps1 = conv_eps1(x, x_0, edge_index)
    assert out_eps1.shape == [4, 16]
    # With eps=0.5, the output should be different from eps=0
    assert not paddle.allclose(out_eps0, out_eps1, atol=1e-6)


def test_fa_conv_with_normalize_false():
    """Test FAConv with normalize=False (requires edge_weight)"""
    x = paddle.randn([4, 16])
    x_0 = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = paddle.randn([6])

    conv = FAConv(16, eps=0.1, normalize=False)
    out = conv(x, x_0, edge_index, edge_weight=edge_weight)
    assert out.shape == [4, 16]

    # Test with return_attention_weights
    result = conv(x, x_0, edge_index, edge_weight=edge_weight,
                  return_attention_weights=True)
    assert paddle.allclose(result[0], out, atol=1e-6)
    assert result[1][0].shape == [2, 6]
    assert result[1][1].shape == [6]


def test_fa_conv_with_add_self_loops_false():
    """Test FAConv with add_self_loops=False"""
    x = paddle.randn([4, 16])
    x_0 = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = FAConv(16, eps=0.1, add_self_loops=False)
    out = conv(x, x_0, edge_index)
    assert out.shape == [4, 16]


def test_fa_conv_with_dropout():
    """Test FAConv with dropout"""
    x = paddle.randn([4, 16])
    x_0 = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = FAConv(16, eps=0.1, dropout=0.5)
    conv.train()
    out_train = conv(x, x_0, edge_index)
    assert out_train.shape == [4, 16]

    conv.eval()
    out_eval = conv(x, x_0, edge_index)
    assert out_eval.shape == [4, 16]