import paddle

from paddle_geometric.nn import DirGNNConv, SAGEConv


def test_dir_gnn_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))
    assert str(conv) == 'DirGNNConv(SAGEConv(16, 32, aggr=mean), alpha=0.5)'

    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 32)


def test_static_dir_gnn_conv():
    x = paddle.randn(shape=[3, 4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))

    out = conv(x, edge_index)
    assert tuple(out.shape)== (3, 4, 32)
