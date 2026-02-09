import paddle

import pytest

from paddle_geometric.nn import WLConv
from paddle_geometric.utils import one_hot



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



def test_wl_conv():
    x1 = paddle.to_tensor([1, 0, 0, 1])
    x2 = one_hot(x1)
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    conv = WLConv()
    assert str(conv) == 'WLConv()'

    out = conv(x1, edge_index)
    assert out.tolist() == [0, 1, 1, 0]
    assert paddle.allclose(conv(x2, edge_index), out)

    assert conv.histogram(out).tolist() == [[2, 2]]
    assert paddle.allclose(conv.histogram(out, norm=True),
                          paddle.to_tensor([[0.7071, 0.7071]]), atol=1e-4)