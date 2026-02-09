import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import AGNNConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('requires_grad', [True, False])
def test_agnn_conv(requires_grad):
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = AGNNConv(requires_grad=requires_grad)
    assert str(conv) == 'AGNNConv()'
    out = conv(x, edge_index)
    assert tuple(out.shape) == (4, 16)
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj2.t()), out, atol=1e-6)
