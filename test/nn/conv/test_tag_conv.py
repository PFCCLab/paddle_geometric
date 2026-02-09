import paddle

import pytest

from paddle_geometric.nn import TAGConv



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



def test_tag_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])

    conv = TAGConv(16, 32, K=2)
    assert str(conv) == 'TAGConv(16, 32, K=2)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]

    # Test with edge weights
    edge_weight = paddle.rand([edge_index.shape[1]])
    out = conv(x, edge_index, edge_weight)
    assert out.shape == [4, 32]