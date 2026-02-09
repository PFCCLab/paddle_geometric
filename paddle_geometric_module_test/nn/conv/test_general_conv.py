import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import GeneralConv
from paddle_geometric.typing import SparseTensor


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(skip_linear=True),
    dict(directed_msg=False),
    dict(heads=3),
    dict(attention=True),
    dict(heads=3, attention=True),
    dict(heads=3, attention=True, attention_type='dot_product'),
    dict(l2_normalize=True),
])
def test_general_conv(kwargs):
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 16])

    conv = GeneralConv(8, 32, **kwargs)
    assert str(conv) == 'GeneralConv(8, 32)'

    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)

    conv = GeneralConv(8, 32, in_edge_channels=16, **kwargs)
    assert str(conv) == 'GeneralConv(8, 32)'

    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)
