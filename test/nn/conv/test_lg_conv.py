import paddle

import pytest

from paddle_geometric.nn import LGConv
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


def test_lg_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = LGConv()
    assert str(conv) == 'LGConv()'

    out1 = conv(x, edge_index)
    assert out1.shape == [4, 8]
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.shape == [4, 8]
    assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)


def test_lg_conv_no_normalize():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1]])

    conv = LGConv(normalize=False)
    assert str(conv) == 'LGConv(normalize=False)'

    out1 = conv(x, edge_index)
    assert out1.shape == [4, 8]

    out2 = conv(x, edge_index, value)
    assert out2.shape == [4, 8]