import paddle

import pytest

from paddle_geometric.nn import MFConv
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


def test_mf_conv():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MFConv(8, 32)
    assert str(conv) == 'MFConv(8, 32)'
    out = conv(x1, edge_index)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out)

    # Test bipartite message passing:
    conv = MFConv((8, 16), 32)
    assert str(conv) == 'MFConv((8, 16), 32)'

    out1 = conv((x1, x2), edge_index)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, (4, 2))
    assert out2.shape == [2, 32]


def test_mf_conv_with_sparse_tensor():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MFConv(8, 32)
    out = conv(x1, edge_index)
    
    adj = to_paddle_csc_tensor(edge_index, size=(4, 4))
    assert paddle.allclose(conv(x1, adj.t()), out)

    # Test bipartite message passing:
    conv = MFConv((8, 16), 32)
    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, (4, 2))

    adj = to_paddle_csc_tensor(edge_index, size=(4, 2))
    assert paddle.allclose(conv((x1, x2), adj.t()), out1)
    assert paddle.allclose(conv((x1, None), adj.t()), out2)