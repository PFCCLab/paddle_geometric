import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

from paddle_geometric.nn import DynamicEdgeConv, EdgeConv


def test_edge_conv_conv():
    x1 = paddle.randn([4, 16])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = EdgeConv(nn)
    # Check that repr contains expected components (Paddle format differs)
    assert 'EdgeConv(nn=Sequential(' in str(conv)
    assert 'Linear(in_features=32, out_features=16' in str(conv)
    assert 'ReLU()' in str(conv)
    assert 'Linear(in_features=16, out_features=32' in str(conv)
    out = conv(x1, edge_index)
    assert out.shape == (4, 32)
    assert paddle.allclose(conv((x1, x1), edge_index), out, atol=1e-6)

    # Test bipartite message passing:
    out = conv((x1, x2), edge_index)
    assert out.shape == (2, 32)


def test_dynamic_edge_conv():
    try:
        from paddle_cluster import knn
    except ImportError:
        print('Skipping DynamicEdgeConv test as paddle-cluster is not installed')
        return

    x1 = paddle.randn([8, 16])
    x2 = paddle.randn([4, 16])
    batch1 = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = paddle.to_tensor([0, 0, 1, 1])

    nn = Seq(Lin(32, 16), ReLU(), Lin(16, 32))
    conv = DynamicEdgeConv(nn, k=2)
    # Check that repr contains expected components (Paddle format differs)
    assert 'DynamicEdgeConv(nn=Sequential(' in str(conv)
    assert 'Linear(in_features=32, out_features=16' in str(conv)
    assert 'ReLU()' in str(conv)
    assert 'Linear(in_features=16, out_features=32' in str(conv)
    assert 'k=2' in str(conv)
    out11 = conv(x1)
    assert out11.shape == (8, 32)

    out12 = conv(x1, batch1)
    assert out12.shape == (8, 32)

    out21 = conv((x1, x2))
    assert out21.shape == (4, 32)

    out22 = conv((x1, x2), (batch1, batch2))
    assert out22.shape == (4, 32)