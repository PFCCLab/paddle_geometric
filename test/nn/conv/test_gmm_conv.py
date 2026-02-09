import paddle

import pytest

from paddle_geometric.nn import GMMConv
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



@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_gmm_conv(separate_gaussians):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])
    adj1 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = GMMConv(8, 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv(8, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)
    # 验证SparseTensor转置后的兼容性
    assert paddle.allclose(conv(x1, adj1.t()), out)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, value, size=(4, 2))

    conv = GMMConv((8, 16), 32, dim=3, kernel_size=5,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv((8, 16), 32, dim=3)'

    out1 = conv((x1, x2), edge_index, value)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)
    # 验证二部图SparseTensor兼容性
    assert paddle.allclose(conv((x1, x2), adj1.t()), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.shape == [2, 32]
    # 验证x2为None时的SparseTensor兼容性
    assert paddle.allclose(conv((x1, None), adj1.t()), out2)


@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_lazy_gmm_conv(separate_gaussians):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])

    conv = GMMConv(-1, 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv(-1, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]

    conv = GMMConv((-1, -1), 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv((-1, -1), 32, dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert out.shape == [2, 32]