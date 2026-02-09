import paddle

from paddle_geometric.nn import FeaStConv
from paddle_geometric.typing import SparseTensor


def test_feast_conv():
    x1 = paddle.randn([4, 16])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = FeaStConv(16, 32, heads=2)
    assert str(conv) == 'FeaStConv(16, 32, heads=2)'

    out = conv(x1, edge_index)
    assert out.shape == [4, 32]

    # Test with SparseTensor (normal graph)
    adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
    assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    out = conv((x1, x2), edge_index)
    assert out.shape == [2, 32]

    # Test bipartite with SparseTensor
    adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
    assert paddle.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)


def test_feast_conv_no_self_loops():
    """Test FeaStConv with add_self_loops=False"""
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = FeaStConv(16, 32, heads=2, add_self_loops=False)
    out = conv(x, edge_index)
    assert out.shape == [4, 32]


def test_feast_conv_no_bias():
    """Test FeaStConv with bias=False"""
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = FeaStConv(16, 32, heads=2, bias=False)
    out = conv(x, edge_index)
    assert out.shape == [4, 32]


def test_feast_conv_single_head():
    """Test FeaStConv with heads=1"""
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = FeaStConv(16, 32, heads=1)
    assert str(conv) == 'FeaStConv(16, 32, heads=1)'
    out = conv(x, edge_index)
    assert out.shape == [4, 32]