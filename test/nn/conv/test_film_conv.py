import paddle

from paddle_geometric.nn import FiLMConv
from paddle_geometric.typing import SparseTensor


def test_film_conv():
    x1 = paddle.randn([4, 4])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])
    edge_type = paddle.to_tensor([0, 1, 1, 0, 0, 1])

    conv = FiLMConv(4, 32)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=1)'
    out = conv(x1, edge_index)
    assert out.shape == [4, 32]

    # Test with SparseTensor (normal graph)
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
    assert paddle.allclose(conv(x1, adj.t()), out, atol=1e-6)

    # Test with multiple relations
    conv = FiLMConv(4, 32, num_relations=2)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=2)'
    out = conv(x1, edge_index, edge_type)
    assert out.shape == [4, 32]

    # Test with multiple relations SparseTensor
    adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 4))
    assert paddle.allclose(conv(x1, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    conv = FiLMConv((4, 16), 32)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=1)'
    out = conv((x1, x2), edge_index)
    assert out.shape == [2, 32]

    # Test bipartite with SparseTensor
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
    assert paddle.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)

    # Test bipartite with multiple relations
    conv = FiLMConv((4, 16), 32, num_relations=2)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=2)'
    out = conv((x1, x2), edge_index, edge_type)
    assert out.shape == [2, 32]

    # Test bipartite with multiple relations SparseTensor
    adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 2))
    assert paddle.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)


def test_film_conv_custom_nn():
    x1 = paddle.randn([4, 4])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])

    # Test with custom nn
    from paddle.nn import Sequential, Linear as PaddleLinear, ReLU
    custom_nn = Sequential(
        PaddleLinear(4, 8),
        ReLU(),
        PaddleLinear(8, 64)  # 2 * out_channels
    )
    
    conv = FiLMConv(4, 32, nn=custom_nn)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=1)'
    out = conv(x1, edge_index)
    assert out.shape == [4, 32]


def test_film_conv_custom_act():
    x1 = paddle.randn([4, 4])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])

    # Test with custom activation
    from paddle.nn import Sigmoid
    conv = FiLMConv(4, 32, act=Sigmoid())
    assert str(conv) == 'FiLMConv(4, 32, num_relations=1)'
    out = conv(x1, edge_index)
    assert out.shape == [4, 32]


def test_film_conv_different_aggr():
    x1 = paddle.randn([4, 4])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])

    # Test with different aggregation schemes
    for aggr in ['add', 'mean', 'max']:
        conv = FiLMConv(4, 32, aggr=aggr)
        out = conv(x1, edge_index)
        assert out.shape == [4, 32]