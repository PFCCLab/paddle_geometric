import paddle

import paddle_geometric.typing
from paddle_geometric.nn import FiLMConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


def test_film_conv():
    x1 = paddle.randn(shape=[4, 4])
    x2 = paddle.randn(shape=[2, 16])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])
    edge_type = paddle.to_tensor([0, 1, 1, 0, 0, 1])

    conv = FiLMConv(4, 32)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=1)'
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out, atol=1e-6)

    conv = FiLMConv(4, 32, num_relations=2)
    assert str(conv) == 'FiLMConv(4, 32, num_relations=2)'
    out = conv(x1, edge_index, edge_type)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 4))
        assert paddle.allclose(conv(x1, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index, edge_type), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj.t()), out, atol=1e-6)

    # Test bipartite message passing:
    conv = FiLMConv((4, 16), 32)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=1)'
    out = conv((x1, x2), edge_index)
    assert tuple(out.shape)== (2, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out, atol=1e-6)

    conv = FiLMConv((4, 16), 32, num_relations=2)
    assert str(conv) == 'FiLMConv((4, 16), 32, num_relations=2)'
    out = conv((x1, x2), edge_index, edge_type)
    assert tuple(out.shape)== (2, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index, edge_type), out,
                              atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj.t()), out, atol=1e-6)
