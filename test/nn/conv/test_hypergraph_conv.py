import paddle

import pytest

from paddle_geometric.nn import HypergraphConv



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



def test_hypergraph_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])

    conv = HypergraphConv(16, 32)
    assert str(conv) == 'HypergraphConv(16, 32)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]