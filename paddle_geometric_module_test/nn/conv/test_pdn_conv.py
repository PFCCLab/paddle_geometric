import paddle

import paddle_geometric.typing
from paddle_geometric.nn import PDNConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


def test_pdn_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 8])

    conv = PDNConv(16, 32, edge_dim=8, hidden_channels=128)
    assert str(conv) == "PDNConv(16, 32)"

    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index, edge_attr), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)


def test_pdn_conv_with_sparse_node_input_feature():
    x = paddle.sparse.sparse_coo_tensor(
        indices=paddle.to_tensor([[0, 0], [0, 1]]),
        values=paddle.to_tensor([1.0, 1.0]),
        shape=[4, 16],
    )
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 8])

    conv = PDNConv(16, 32, edge_dim=8, hidden_channels=128)

    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert paddle.allclose(conv(x, adj.t(), edge_attr), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index, edge_attr), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t(), edge_attr), out, atol=1e-6)
