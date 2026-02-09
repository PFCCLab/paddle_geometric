import paddle

from paddle_geometric.nn import MessagePassing
from paddle_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = paddle.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        row, col = edge_index
        deg = degree(row, x.shape[0], dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        return self.propagate(edge_index, size=(x.shape[0], x.shape[0]), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


def test_create_gnn():
    conv = GCNConv(16, 32)
    x = paddle.randn(shape=[5, 16])
    edge_index = paddle.randint(0, 5, shape=[2, 64], dtype=paddle.int64)
    out = conv(x, edge_index)
    assert tuple(out.shape)== (5, 32)
