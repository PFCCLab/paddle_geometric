import paddle

import pytest

from paddle_geometric.nn import MixHopConv
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


def test_mixhop_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = MixHopConv(16, 32, powers=[0, 1, 2, 4])
    assert str(conv) == 'MixHopConv(16, 32, powers=[0, 1, 2, 4])'

    out1 = conv(x, edge_index)
    assert out1.shape == [4, 128]
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.shape == [4, 128]
    assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)