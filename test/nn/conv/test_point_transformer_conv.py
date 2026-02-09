import paddle

import pytest

from paddle.nn import Linear, ReLU, Sequential

from paddle_geometric.nn import PointTransformerConv
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


def test_point_transformer_conv():
    x1 = paddle.rand([4, 16])
    x2 = paddle.randn([2, 8])
    pos1 = paddle.rand([4, 3])
    pos2 = paddle.randn([2, 3])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert str(conv) == 'PointTransformerConv(16, 32)'

    out = conv(x1, pos1, edge_index)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test with custom pos_nn and attn_nn
    pos_nn = Sequential(Linear(3, 16), ReLU(), Linear(16, 32))
    attn_nn = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
    conv = PointTransformerConv(16, 32, pos_nn, attn_nn)

    out = conv(x1, pos1, edge_index)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)
    if SparseTensor is not None:
        assert paddle.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = PointTransformerConv((16, 8), 32)
    assert str(conv) == 'PointTransformerConv((16, 8), 32)'

    out = conv((x1, x2), (pos1, pos2), edge_index)
    assert out.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), (pos1, pos2), adj1.t()), out)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), (pos1, pos2), adj2.t()), out)