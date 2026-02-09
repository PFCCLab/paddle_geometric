import paddle

import paddle_geometric.typing
from paddle_geometric.nn import ClusterGCNConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_cluster_gcn_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = ClusterGCNConv(16, 32, diag_lambda=1.)
    assert str(conv) == 'ClusterGCNConv(16, 32, diag_lambda=1.0)'
    out = conv(x, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-5)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-5)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj2.t()), out, atol=1e-5)
