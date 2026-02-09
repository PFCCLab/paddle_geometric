import paddle

import pytest

from paddle_geometric.nn import AGNNConv
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


@pytest.mark.parametrize('requires_grad', [True, False])
def test_agnn_conv(requires_grad):
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = AGNNConv(requires_grad=requires_grad)
    assert str(conv) == 'AGNNConv()'
    
    out = conv(x, edge_index)
    assert out.shape == [4, 16]
    
    # 测试稀疏张量
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-6)