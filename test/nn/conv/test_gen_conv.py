import paddle
import pytest

from paddle_geometric.nn import GENConv
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



@pytest.mark.parametrize('aggr', [
    'softmax',
    'powermean',
    ['softmax', 'powermean'],
])
def test_gen_conv(aggr):
    x1 = paddle.randn([4, 16])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn([edge_index.shape[1], 16])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = GENConv(16, 32, aggr, edge_dim=16, msg_norm=True)
    assert str(conv) == f'GENConv(16, 32, aggr={aggr})'
    out1 = conv(x1, edge_index)
    assert out1.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out1)
    # 验证SparseTensor转置后的兼容性
    assert paddle.allclose(conv(x1, adj1.t()), out1, atol=1e-4)

    out2 = conv(x1, edge_index, value)
    assert out2.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, (4, 4)), out2)
    # 验证带edge_weight的SparseTensor兼容性
    assert paddle.allclose(conv(x1, adj2.t()), out2, atol=1e-4)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 2))

    out1 = conv((x1, x2), edge_index)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    # 验证二部图SparseTensor兼容性
    assert paddle.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-4)

    out2 = conv((x1, x2), edge_index, value)
    assert out2.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out2)
    # 验证二部图带edge_weight的SparseTensor兼容性
    assert paddle.allclose(conv((x1, x2), adj2.t()), out2, atol=1e-4)

    # Test bipartite message passing with unequal feature dimensions:
    conv.reset_parameters()

    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])

    conv = GENConv((8, 16), 32, aggr)
    assert str(conv) == f'GENConv((8, 16), 32, aggr={aggr})'

    out1 = conv((x1, x2), edge_index)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    # 验证不等特征维度的SparseTensor兼容性
    assert paddle.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-4)

    out2 = conv((x1, None), edge_index, size=(4, 2))
    assert out2.shape == [2, 32]
    assert paddle.allclose(conv((x1, None), edge_index, size=(4, 2)), out2)
    # 验证x2为None时的SparseTensor兼容性
    assert paddle.allclose(conv((x1, None), adj1.t()), out2, atol=1e-4)

    # Test lazy initialization:
    conv = GENConv((-1, -1), 32, aggr, edge_dim=-1)
    assert str(conv) == f'GENConv((-1, -1), 32, aggr={aggr})'
    out1 = conv((x1, x2), edge_index, value)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, size=(4, 2)), out1)
    # 验证懒初始化的SparseTensor兼容性
    assert paddle.allclose(conv((x1, x2), adj2.t()), out1, atol=1e-4)

    out2 = conv((x1, None), edge_index, value, size=(4, 2))
    assert out2.shape == [2, 32]
    assert paddle.allclose(conv((x1, None), edge_index, value, size=(4, 2)), out2)
    # 验证懒初始化x2为None时的SparseTensor兼容性
    assert paddle.allclose(conv((x1, None), adj2.t()), out2, atol=1e-4)