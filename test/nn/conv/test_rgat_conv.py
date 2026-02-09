import paddle

import pytest

from paddle_geometric.nn import RGATConv



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



def test_rgat_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_type = paddle.to_tensor([0, 1, 0, 1])

    conv = RGATConv(16, 32, num_relations=2, heads=2)
    assert 'RGATConv(16, 32, heads=2)' in str(conv)

    out = conv(x, edge_index, edge_type)
    assert out.shape == [4, 64]

    # Test without concat
    conv = RGATConv(16, 32, num_relations=2, heads=2, concat=False)
    out = conv(x, edge_index, edge_type)
    assert out.shape == [4, 32]