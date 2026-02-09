import random

import pytest
import paddle

from paddle_geometric.data import HeteroData
from paddle_geometric.datasets import FakeHeteroDataset
from paddle_geometric.nn import (
    GATConv,
    GCN2Conv,
    GCNConv,
    HeteroConv,
    Linear,
    MessagePassing,
    SAGEConv,
)


def get_random_edge_index(num_src, num_dst, num_edges):
    src = paddle.randint(0, num_src, [num_edges])
    dst = paddle.randint(0, num_dst, [num_edges])
    return paddle.stack([src, dst])


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat', None])
def test_hetero_conv(aggr):
    data = HeteroData()
    data['paper'].x = paddle.randn([50, 32])
    data['author'].x = paddle.randn([30, 64])
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['paper', 'author'].edge_attr = paddle.randn([100, 3])
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = paddle.rand([200])

    # Unspecified edge types should be ignored:
    data['author', 'author'].edge_index = get_random_edge_index(30, 30, 100)

    conv = HeteroConv(
        {
            ('paper', 'to', 'paper'):
            GCNConv(-1, 64),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 64),
            ('paper', 'to', 'author'):
            GATConv((-1, -1), 64, edge_dim=3, add_self_loops=False),
        },
        aggr=aggr,
    )

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out_dict = conv(
        data.x_dict,
        data.edge_index_dict,
        data.edge_attr_dict,
        edge_weight_dict=data.edge_weight_dict,
    )

    assert len(out_dict) == 2
    if aggr == 'cat':
        assert out_dict['paper'].shape == [50, 128]
        assert out_dict['author'].shape == [30, 64]
    elif aggr is not None:
        assert out_dict['paper'].shape == [50, 64]
        assert out_dict['author'].shape == [30, 64]
    else:
        assert out_dict['paper'].shape == [50, 2, 64]
        assert out_dict['author'].shape == [30, 1, 64]


def test_gcn2_hetero_conv():
    data = HeteroData()
    data['paper'].x = paddle.randn([50, 32])
    data['author'].x = paddle.randn([30, 64])
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['author', 'author'].edge_index = get_random_edge_index(30, 30, 100)
    data['paper', 'paper'].edge_weight = paddle.rand([200])

    conv = HeteroConv({
        ('paper', 'to', 'paper'): GCN2Conv(32, alpha=0.1),
        ('author', 'to', 'author'): GCN2Conv(64, alpha=0.2),
    })

    out_dict = conv(
        data.x_dict,
        data.x_dict,
        data.edge_index_dict,
        edge_weight_dict=data.edge_weight_dict,
    )

    assert len(out_dict) == 2
    assert out_dict['paper'].shape == [50, 32]
    assert out_dict['author'].shape == [30, 64]


class CustomConv(MessagePassing):
    def __init__(self, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, y, z):
        return self.propagate(edge_index, x=x, y=y, z=z)

    def message(self, x_j, y_j, z_j):
        return self.lin(paddle.concat([x_j, y_j, z_j], axis=-1))


def test_hetero_conv_with_custom_conv():
    data = HeteroData()
    data['paper'].x = paddle.randn([50, 32])
    data['paper'].y = paddle.randn([50, 3])
    data['paper'].z = paddle.randn([50, 3])
    data['author'].x = paddle.randn([30, 64])
    data['author'].y = paddle.randn([30, 3])
    data['author'].z = paddle.randn([30, 3])
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)

    conv = HeteroConv({key: CustomConv(64) for key in data.edge_types})
    # Test node `args_dict` and `kwargs_dict` with `y_dict` and `z_dict`:
    out_dict = conv(
        data.x_dict,
        data.edge_index_dict,
        data.y_dict,
        z_dict=data.z_dict,
    )
    assert len(out_dict) == 2
    assert out_dict['paper'].shape == [50, 64]
    assert out_dict['author'].shape == [30, 64]


class MessagePassingLoops(MessagePassing):
    def __init__(self):
        super().__init__()
        self.add_self_loops = True


def test_hetero_conv_self_loop_error():
    HeteroConv({('a', 'to', 'a'): MessagePassingLoops()})
    with pytest.raises(ValueError, match="incorrect message passing"):
        HeteroConv({('a', 'to', 'b'): MessagePassingLoops()})