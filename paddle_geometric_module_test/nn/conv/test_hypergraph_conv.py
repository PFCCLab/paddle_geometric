import paddle

from paddle_geometric.nn import HypergraphConv


def test_hypergraph_conv_with_more_nodes_than_edges():
    in_channels, out_channels = (16, 32)
    hyperedge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = paddle.randn(shape=[num_nodes, in_channels])
    hyperedge_weight = paddle.to_tensor([1.0, 0.5])
    hyperedge_attr = paddle.randn(shape=[num_edges, in_channels])

    conv = HypergraphConv(in_channels, out_channels)
    assert str(conv) == 'HypergraphConv(16, 32)'
    out = conv(x, hyperedge_index)
    assert tuple(out.shape)== (num_nodes, out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight)
    assert tuple(out.shape)== (num_nodes, out_channels)

    conv = HypergraphConv(in_channels, out_channels, use_attention=True,
                          heads=2)
    out = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
    assert tuple(out.shape)== (num_nodes, 2 * out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tuple(out.shape)== (num_nodes, 2 * out_channels)

    conv = HypergraphConv(in_channels, out_channels, use_attention=True,
                          heads=2, concat=False, dropout=0.5)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tuple(out.shape)== (num_nodes, out_channels)


def test_hypergraph_conv_with_more_edges_than_nodes():
    in_channels, out_channels = (16, 32)
    hyperedge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                    [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
    hyperedge_weight = paddle.to_tensor([1.0, 0.5, 0.8, 0.2, 0.7])
    num_nodes = hyperedge_index[0].max().item() + 1
    x = paddle.randn(shape=[num_nodes, in_channels])

    conv = HypergraphConv(in_channels, out_channels)
    assert str(conv) == 'HypergraphConv(16, 32)'
    out = conv(x, hyperedge_index)
    assert tuple(out.shape)== (num_nodes, out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight)
    assert tuple(out.shape)== (num_nodes, out_channels)
