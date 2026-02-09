import paddle
import pytest

from paddle_geometric.nn import GravNetConv



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


def test_gravnet_conv():
    x1 = paddle.randn([8, 16])
    x2 = paddle.randn([4, 16])
    batch1 = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = paddle.to_tensor([0, 0, 1, 1])

    conv = GravNetConv(16, 32, space_dimensions=4, propagate_dimensions=8, k=2)
    assert str(conv) == 'GravNetConv(16, 32, k=2)'

    out11 = conv(x1)
    assert out11.shape == [8, 32]

    out12 = conv(x1, batch1)
    assert out12.shape == [8, 32]

    out21 = conv((x1, x2))
    assert out21.shape == [4, 32]

    out22 = conv((x1, x2), (batch1, batch2))
    assert out22.shape == [4, 32]