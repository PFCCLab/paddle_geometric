import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import SuperGATConv
from paddle_geometric.typing import SparseTensor


@pytest.mark.parametrize('att_type', ['MX', 'SD'])
def test_supergat_conv(att_type):
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SuperGATConv(8, 32, heads=2, attention_type=att_type,
                        neg_sample_ratio=1.0, edge_sample_ratio=1.0)
    assert str(conv) == f'SuperGATConv(8, 32, heads=2, type={att_type})'

    out = conv(x, edge_index)
    assert tuple(out.shape) == (4, 64)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)

    # Negative samples are given:
    neg_edge_index = conv.negative_sampling(edge_index, x.shape[0])
    assert paddle.allclose(conv(x, edge_index, neg_edge_index), out)
    att_loss = conv.get_attention_loss()
    assert isinstance(att_loss, paddle.Tensor)
    assert float(att_loss) > 0

    # Batch of graphs:
    x = paddle.randn(shape=[8, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                               [0, 0, 1, 1, 4, 4, 5, 5]])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = conv(x, edge_index, batch=batch)
    assert tuple(out.shape) == (8, 64)

    # Batch of graphs and negative samples are given:
    neg_edge_index = conv.negative_sampling(edge_index, x.shape[0], batch)
    assert paddle.allclose(conv(x, edge_index, neg_edge_index), out)
    att_loss = conv.get_attention_loss()
    assert isinstance(att_loss, paddle.Tensor)
    assert float(att_loss) > 0
