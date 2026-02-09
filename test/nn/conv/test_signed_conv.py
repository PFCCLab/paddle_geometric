import paddle

import pytest

from paddle_geometric.nn import SignedConv



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



def test_signed_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])

    conv1 = SignedConv(16, 32, first_aggr=True)
    assert str(conv1) == 'SignedConv(16, 32, first_aggr=True)'

    conv2 = SignedConv(32, 48, first_aggr=False)
    assert str(conv2) == 'SignedConv(32, 48, first_aggr=False)'

    out1 = conv1(x, edge_index, edge_index)
    assert out1.shape == [4, 64]

    out2 = conv2(out1, edge_index, edge_index)
    assert out2.shape == [4, 96]

    # Test bipartite message passing:
    x_src = paddle.randn([4, 16])
    x_dst = paddle.randn([2, 16])

    out_bipartite1 = conv1((x_src, x_dst), edge_index, edge_index)
    assert out_bipartite1.shape == [2, 64]

    out_bipartite2 = conv2((out1, out1[:2]), edge_index, edge_index)
    assert out_bipartite2.shape == [2, 96]