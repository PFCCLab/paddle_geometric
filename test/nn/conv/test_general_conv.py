import pytest
import paddle

from paddle_geometric.nn import GeneralConv
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



@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(skip_linear=True),
    dict(directed_msg=False),
    dict(heads=3),
    dict(attention=True),
    dict(heads=3, attention=True),
    dict(heads=3, attention=True, attention_type='dot_product'),
    dict(l2_normalize=True),
])
def test_general_conv(kwargs):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn([edge_index.shape[1], 16])

    conv = GeneralConv(8, 32, **kwargs)
    assert str(conv) == 'GeneralConv(8, 32)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]
    # 验证SparseTensor转置后的兼容性
    adj = to_paddle_csc_tensor(edge_index, size=(4, 4))
    assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)

    conv = GeneralConv(8, 32, in_edge_channels=16, **kwargs)
    assert str(conv) == 'GeneralConv(8, 32)'

    out = conv(x, edge_index, edge_attr)
    assert out.shape == [4, 32]
    # 验证带edge_attr的SparseTensor兼容性
    adj = to_paddle_csc_tensor(edge_index, edge_attr, size=(4, 4))
    assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)