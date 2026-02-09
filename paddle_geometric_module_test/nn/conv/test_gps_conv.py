import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import GPSConv, SAGEConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('attn_type', ['multihead', 'performer'])
@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm, attn_type):
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = paddle.to_tensor([0, 0, 1, 1])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = GPSConv(16, conv=SAGEConv(16, 16), heads=4, norm=norm,
                   attn_type=attn_type)
    conv.reset_parameters()
    assert str(conv) == (f'GPSConv(16, conv=SAGEConv(16, 16, aggr=mean), '
                         f'heads=4, attn_type={attn_type})')

    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 16)
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-6)

    out = conv(x, edge_index, batch)
    assert tuple(out.shape)== (4, 16)
    assert paddle.allclose(conv(x, adj1.t(), batch), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv(x, adj2.t(), batch), out, atol=1e-6)
