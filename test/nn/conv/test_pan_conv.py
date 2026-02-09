import paddle

import pytest

from paddle_geometric.nn import PANConv
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
    skip_if_cuda_error()


def test_pan_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = PANConv(16, 32, filter_size=2)
    assert str(conv) == 'PANConv(16, 32, filter_size=2)'
    
    out1, M1 = conv(x, edge_index)
    assert out1.shape == [4, 32]

    out2, M2 = conv(x, adj1.t())
    assert paddle.allclose(out1, out2, atol=1e-6)
    assert paddle.allclose(M1.to_dense(), M2.to_dense())