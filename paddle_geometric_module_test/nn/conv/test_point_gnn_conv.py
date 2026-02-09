import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MLP, PointGNNConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_point_gnn_conv():
    x = paddle.randn(shape=[6, 8])
    pos = paddle.randn(shape=[6, 3])
    edge_index = paddle.to_tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(6, 6))

    conv = PointGNNConv(
        mlp_h=MLP([8, 16, 3]),
        mlp_f=MLP([3 + 8, 16, 8]),
        mlp_g=MLP([8, 16, 8]),
    )
    assert str(conv) == ('PointGNNConv(\n'
                         '  mlp_h=MLP(8, 16, 3),\n'
                         '  mlp_f=MLP(11, 16, 8),\n'
                         '  mlp_g=MLP(8, 16, 8),\n'
                         ')')

    out = conv(x, pos, edge_index)
    assert tuple(out.shape)== (6, 8)
    assert paddle.allclose(conv(x, pos, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert paddle.allclose(conv(x, pos, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, pos, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, pos, adj2.t()), out, atol=1e-6)
