import paddle

import pytest

from paddle_geometric.nn import SplineConv
from paddle_geometric.testing import withPackage



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



@withPackage('paddle_spline_conv')
def test_spline_conv():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[0], 3])

    conv = SplineConv(8, 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv(8, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)

    # Test bipartite message passing:
    conv = SplineConv((8, 16), 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv((8, 16), 32, dim=3)'

    out1 = conv((x1, x2), edge_index, value)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.shape == [2, 32]


@withPackage('paddle_spline_conv')
def test_lazy_spline_conv():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[0], 3])

    conv = SplineConv(-1, 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv(-1, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]

    conv = SplineConv((-1, -1), 32, dim=3, kernel_size=5)
    assert str(conv) == 'SplineConv((-1, -1), 32, dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert out.shape == [2, 32]