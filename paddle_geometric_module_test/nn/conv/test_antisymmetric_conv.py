import paddle
import pytest

import paddle_geometric.typing
from paddle_geometric.nn import AntiSymmetricConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_antisymmetric_conv():
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])

    conv = AntiSymmetricConv(8)
    assert str(conv) == ('AntiSymmetricConv(8, phi=GCNConv(8, 8), '
                         'num_iters=1, epsilon=0.1, gamma=0.1)')

    out1 = conv(x, edge_index)
    assert tuple(out1.shape) == (4, 8)

    out2 = conv(x, edge_index, value)
    assert tuple(out2.shape) == (4, 8)

    # Skip sparse matrix tests on CPU as they are not supported
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
        adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))
        
        assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)
        assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
            adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
            assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
            assert paddle.allclose(conv(x, adj4.t()), out2, atol=1e-6)
