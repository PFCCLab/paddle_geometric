from typing import Optional, Tuple

import pytest
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.nn import TransformerConv
from paddle_geometric.typing import Adj, SparseTensor
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



@pytest.mark.parametrize('edge_dim', [None, 8])
@pytest.mark.parametrize('concat', [True, False])
def test_transformer_conv(edge_dim, concat):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    out_channels = 32
    heads = 2
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn([edge_index.shape[1], edge_dim]) if edge_dim else None
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = TransformerConv(8, out_channels, heads, beta=True,
                           edge_dim=edge_dim, concat=concat)
    assert str(conv) == f'TransformerConv(8, {out_channels}, heads={heads})'

    out = conv(x1, edge_index, edge_attr)
    assert out.shape == [4, out_channels * (heads if concat else 1)]
    assert paddle.allclose(conv(x1, adj1.t(), edge_attr), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, edge_attr, return_attention_weights=True)
    assert paddle.allclose(result[0], out)
    assert result[1][0].shape == [2, 4]
    assert result[1][1].shape == [4, 2]
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = TransformerConv((8, 16), out_channels, heads=heads, beta=True,
                           edge_dim=edge_dim, concat=concat)
    assert str(conv) == (f'TransformerConv((8, 16), {out_channels}, '
                         f'heads={heads})')

    out = conv((x1, x2), edge_index, edge_attr)
    assert out.shape == [2, out_channels * (heads if concat else 1)]
    assert paddle.allclose(conv((x1, x2), adj1.t(), edge_attr), out, atol=1e-6)