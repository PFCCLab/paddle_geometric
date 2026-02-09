import paddle

import pytest

from paddle_geometric.nn import XConv



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



def test_x_conv():
    x = paddle.randn([8, 16])
    pos = paddle.rand([8, 3])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])

    conv = XConv(16, 32, dim=3, kernel_size=2, dilation=2)
    assert str(conv) == 'XConv(16, 32)'

    paddle.seed(12345)
    out1 = conv(x, pos)
    assert out1.shape == [8, 32]

    paddle.seed(12345)
    out2 = conv(x, pos, batch)
    assert out2.shape == [8, 32]