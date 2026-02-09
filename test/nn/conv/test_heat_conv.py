import paddle

import pytest

from paddle_geometric.nn import HEATConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('concat', [True, False])
def test_heat_conv(concat):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn([4, 2])
    node_type = paddle.to_tensor([0, 0, 1, 2])
    edge_type = paddle.to_tensor([0, 2, 1, 2])

    conv = HEATConv(in_channels=8, out_channels=16, num_node_types=3,
                    num_edge_types=3, edge_type_emb_dim=5, edge_dim=2,
                    edge_attr_emb_dim=6, heads=2, concat=concat)
    assert str(conv) == 'HEATConv(8, 16, heads=2)'

    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.shape == [4, 32 if concat else 16]

    # Test with SparseTensor
    adj = to_paddle_csc_tensor(edge_index, edge_attr, (4, 4))
    assert paddle.allclose(conv(x, adj.t(), node_type, edge_type), out,
                           atol=1e-5)