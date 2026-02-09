import paddle

from paddle_geometric.data import Batch, Data
from paddle_geometric.nn import ChebConv, GCNConv, MessagePassing


class MyConv(MessagePassing):
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


def test_static_graph():
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x1, x2 = paddle.randn(shape=[3, 8]), paddle.randn(shape=[3, 8])

    data1 = Data(edge_index=edge_index, x=x1)
    data2 = Data(edge_index=edge_index, x=x2)
    batch = Batch.from_data_list([data1, data2])

    x = paddle.stack([x1, x2], dim=0)
    for conv in [MyConv(), GCNConv(8, 16), ChebConv(8, 16, K=2)]:
        out1 = conv(batch.x, batch.edge_index)
        assert out1.shape[0] == 6
        conv.node_dim = 1
        out2 = conv(x, edge_index)
        assert tuple(out2.shape[:2]) == (2, 3)
        assert paddle.allclose(out1, out2.view(-1, out2.shape[-1]))
