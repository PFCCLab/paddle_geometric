import paddle

import pytest

from paddle_geometric.nn import LEConv
from paddle_geometric.utils import to_paddle_csc_tensor



def skip_if_cuda_error():
    try:
        import paddle
        paddle.set_device('cpu')
        return False
    except Exception as e:
        if 'cuda' in str(e).lower() or 'CUDA' in str(e):
            pytest.skip(f"CUDA 兼容性问题: {e}")
        return False


@pytest.fixture(autouse=True)
def setup_cpu_mode():
    """自动设置 CPU 模式"""
    skip_if_cuda_error()


def test_le_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = LEConv(16, 32)
    assert str(conv) == 'LEConv(16, 32)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x, adj1.t()), out)


def test_le_conv_lazy():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = LEConv(-1, 32)
    assert str(conv) == 'LEConv(-1, 32)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]


def test_le_conv_bipartite():
    x_s = paddle.randn([4, 16])
    x_t = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = LEConv((16, 16), 32)
    assert str(conv) == 'LEConv((16, 16), 32)'

    out = conv((x_s, x_t), edge_index)
    assert out.shape == [2, 32]


def test_le_conv_bipartite_lazy():
    x_s = paddle.randn([4, 16])
    x_t = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = LEConv((-1, -1), 32)
    assert str(conv) == 'LEConv((-1, -1), 32)'

    out = conv((x_s, x_t), edge_index)
    assert out.shape == [2, 32]


def test_le_conv_edge_weight():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = paddle.randn([6])

    conv = LEConv(16, 32)
    out = conv(x, edge_index, edge_weight)
    assert out.shape == [4, 32]