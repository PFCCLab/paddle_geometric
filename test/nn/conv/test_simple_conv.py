import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import SimpleConv
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



@pytest.mark.parametrize('aggr, combine_root', [
    ('mean', None),
    ('sum', 'sum'),
    (['mean', 'max'], 'cat'),
    ('mean', 'self_loop'),
])
def test_simple_conv(aggr, combine_root):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = SimpleConv(aggr, combine_root)
    assert str(conv) == 'SimpleConv()'

    num_aggrs = 1 if isinstance(aggr, str) else len(aggr)
    output_size = sum([8] * num_aggrs) + (8 if combine_root == 'cat' else 0)

    out = conv(x1, edge_index)
    assert out.shape == [4, output_size]
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out)
    assert paddle.allclose(conv(x1, adj1.t()), out)

    # Test bipartite message passing:
    if combine_root != 'self_loop':
        adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

        out = conv((x1, x2), edge_index)
        assert out.shape == [2, output_size]
        assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out)
        assert paddle.allclose(conv((x1, x2), adj1.t()), out)