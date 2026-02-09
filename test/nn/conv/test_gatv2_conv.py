import paddle

import pytest

from paddle_geometric.nn import GATv2Conv
from paddle_geometric.typing import Adj
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('residual', [False, True])
def test_gatv2_conv(residual):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = GATv2Conv(8, 32, heads=2, residual=residual)
    assert str(conv) == 'GATv2Conv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert out.shape == [4, 64]
    assert paddle.allclose(conv(x1, edge_index), out)
    # 验证SparseTensor转置后的兼容性
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert paddle.allclose(result[0], out)
    assert result[1][0].shape == [2, 7]
    assert result[1][1].shape == [7, 2]
    assert paddle.all(result[1][1] >= 0) and paddle.all(result[1][1] <= 1)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert out.shape == [2, 64]
    assert paddle.allclose(conv((x1, x2), edge_index), out)
    # 验证二部图SparseTensor转置后的兼容性
    assert paddle.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)


def test_gatv2_conv_with_edge_attr():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = paddle.randn([edge_index.shape[1]])
    edge_attr = paddle.randn([edge_index.shape[1], 4])

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value=0.5)
    out = conv(x, edge_index, edge_weight)
    assert out.shape == [4, 64]

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value='mean')
    out = conv(x, edge_index, edge_weight)
    assert out.shape == [4, 64]

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value=0.5)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == [4, 64]

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value='mean')
    out = conv(x, edge_index, edge_attr)
    assert out.shape == [4, 64]