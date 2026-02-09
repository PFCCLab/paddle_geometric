import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import DNAConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv(channels, num_layers):
    x = paddle.randn(shape=[4, num_layers, channels])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DNAConv(channels, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, channels)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=1, groups=1)'
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, channels)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)
    out = conv(x, edge_index)
    assert conv._cached_edge_index is not None
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, channels)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)


@pytest.mark.parametrize('channels', [32])
@pytest.mark.parametrize('num_layers', [3])
def test_dna_conv_sparse_tensor(channels, num_layers):
    x = paddle.randn(shape=[4, num_layers, channels])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = DNAConv(32, heads=4, groups=8, dropout=0.0)
    assert str(conv) == 'DNAConv(32, heads=4, groups=8)'
    out1 = conv(x, edge_index)
    assert tuple(out1.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, edge_index, value)
    assert tuple(out2.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert paddle.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out1, atol=1e-6)
        assert paddle.allclose(jit(x, edge_index, value), out2, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj3.t()), out1, atol=1e-6)
            assert paddle.allclose(jit(x, adj4.t()), out2, atol=1e-6)

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.0, cached=True)

    out1 = conv(x, adj1.t())
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert conv._cached_adj_t is not None
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
