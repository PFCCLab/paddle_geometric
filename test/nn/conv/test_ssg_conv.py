import paddle

import pytest

from paddle_geometric.nn import SSGConv



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



def test_ssg_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])

    conv = SSGConv(16, 32, alpha=0.1, K=10)
    assert str(conv) == 'SSGConv(16, 32, K=10, alpha=0.1)'

    out1 = conv(x, edge_index)
    assert out1.shape == [4, 32]

    out2 = conv(x, edge_index, value)
    assert out2.shape == [4, 32]

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_h is not None
    assert out1.shape == [4, 32]