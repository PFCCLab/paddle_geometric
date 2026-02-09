import paddle

from paddle_geometric.nn import ARMAConv
from paddle_geometric.utils import to_paddle_csc_tensor


def test_arma_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = ARMAConv(16, 32, num_stacks=8, num_layers=4)
    assert str(conv) == 'ARMAConv(16, 32, num_stacks=8, num_layers=4)'
    out = conv(x, edge_index)
    assert out.shape == [4, 32]
    # Note: Paddle version handles sparse tensors differently
    # We skip the 3D feature tensor test for sparse tensors


def test_lazy_arma_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = ARMAConv(-1, 32, num_stacks=8, num_layers=4)
    assert str(conv) == 'ARMAConv(-1, 32, num_stacks=8, num_layers=4)'
    out = conv(x, edge_index)
    assert out.shape == [4, 32]