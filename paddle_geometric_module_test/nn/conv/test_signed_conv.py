import paddle

import paddle_geometric.typing
from paddle_geometric.nn import SignedConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_signed_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv1 = SignedConv(16, 32, first_aggr=True)
    assert str(conv1) == 'SignedConv(16, 32, first_aggr=True)'

    conv2 = SignedConv(32, 48, first_aggr=False)
    assert str(conv2) == 'SignedConv(32, 48, first_aggr=False)'

    out1 = conv1(x, edge_index, edge_index)
    assert tuple(out1.shape)== (4, 64)
    assert paddle.allclose(conv1(x, adj1.t(), adj1.t()), out1)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv1(x, adj2.t(), adj2.t()), out1)

    out2 = conv2(out1, edge_index, edge_index)
    assert tuple(out2.shape)== (4, 96)
    assert paddle.allclose(conv2(out1, adj1.t(), adj1.t()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv2(out1, adj2.t(), adj2.t()), out2)

    if is_full_test():
        jit1 = paddle.jit.to_static(conv1)
        jit2 = paddle.jit.to_static(conv2)
        assert paddle.allclose(jit1(x, edge_index, edge_index), out1)
        assert paddle.allclose(jit2(out1, edge_index, edge_index), out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit1(x, adj2.t(), adj2.t()), out1)
            assert paddle.allclose(jit2(out1, adj2.t(), adj2.t()), out2)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    assert paddle.allclose(conv1((x, x[:2]), edge_index, edge_index), out1[:2],
                          atol=1e-6)
    assert paddle.allclose(conv1((x, x[:2]), adj1.t(), adj1.t()), out1[:2],
                          atol=1e-6)
    assert paddle.allclose(conv2((out1, out1[:2]), edge_index, edge_index),
                          out2[:2], atol=1e-6)
    assert paddle.allclose(conv2((out1, out1[:2]), adj1.t(), adj1.t()),
                          out2[:2], atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv1((x, x[:2]), adj2.t(), adj2.t()), out1[:2],
                              atol=1e-6)
        assert paddle.allclose(conv2((out1, out1[:2]), adj2.t(), adj2.t()),
                              out2[:2], atol=1e-6)

    if is_full_test():
        assert paddle.allclose(jit1((x, x[:2]), edge_index, edge_index),
                              out1[:2], atol=1e-6)
        assert paddle.allclose(jit2((out1, out1[:2]), edge_index, edge_index),
                              out2[:2], atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit1((x, x[:2]), adj2.t(), adj2.t()),
                                  out1[:2], atol=1e-6)
            assert paddle.allclose(jit2((out1, out1[:2]), adj2.t(), adj2.t()),
                                  out2[:2], atol=1e-6)
