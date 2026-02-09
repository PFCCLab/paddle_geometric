import paddle

import pytest

from paddle_geometric.nn import MLP, PointGNNConv
from paddle_geometric.typing import SparseTensor
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


def test_point_gnn_conv():
    x = paddle.randn([6, 8])
    pos = paddle.randn([6, 3])
    edge_index = paddle.to_tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(6, 6))

    conv = PointGNNConv(
        mlp_h=MLP([8, 16, 3]),
        mlp_f=MLP([3 + 8, 16, 8]),
        mlp_g=MLP([8, 16, 8]),
    )
    
    # 检查字符串表示（适配Paddle的输出格式）
    assert 'PointGNNConv(' in str(conv)
    assert 'mlp_h=MLP(' in str(conv)
    assert 'mlp_f=MLP(' in str(conv)
    assert 'mlp_g=MLP(' in str(conv)

    out = conv(x, pos, edge_index)
    assert out.shape == [6, 8]
    assert paddle.allclose(conv(x, pos, adj1.t()), out, atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert paddle.allclose(conv(x, pos, adj2.t()), out, atol=1e-6)