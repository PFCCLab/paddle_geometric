import pytest
import paddle

from paddle_geometric.nn import DNAConv
from paddle_geometric.utils import to_paddle_csr_tensor


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv(channels, num_layers):
    x = paddle.randn((4, num_layers, channels))
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DNAConv(channels, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out = conv(x, edge_index)
    assert out.shape == (4, channels)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=1, groups=1)'
    out = conv(x, edge_index)
    assert out.shape == (4, channels)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)
    out = conv(x, edge_index)
    assert conv._cached_edge_index is not None
    out = conv(x, edge_index)
    assert out.shape == (4, channels)


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv_sparse_tensor(channels, num_layers):
    x = paddle.randn((4, num_layers, channels))
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csr_tensor(edge_index, size=(4, 4))

    conv = DNAConv(32, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out1 = conv(x, edge_index)
    assert out1.shape == (4, 32)
    assert paddle.allclose(conv(x, adj1), out1, atol=1e-6)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)

    out1 = conv(x, adj1)
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, adj1), out1, atol=1e-6)