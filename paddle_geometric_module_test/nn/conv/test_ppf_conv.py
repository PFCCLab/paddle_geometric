import paddle
import paddle.nn.functional as F
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

import paddle_geometric.typing
from paddle_geometric.nn import PPFConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_ppf_conv():
    x1 = paddle.randn(shape=[4, 16])
    pos1 = paddle.randn(shape=[4, 3])
    pos2 = paddle.randn(shape=[2, 3])
    n1 = F.normalize(paddle.rand([4, 3]), axis=-1)
    n2 = F.normalize(paddle.rand([2, 3]), axis=-1)
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    local_nn = Seq(Lin(16 + 4, 32), ReLU(), Lin(32, 32))
    global_nn = Seq(Lin(32, 32))
    conv = PPFConv(local_nn, global_nn)
    assert str(conv) == (
        'PPFConv(local_nn=Sequential(\n'
        '  (0): Linear(in_features=20, out_features=32, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, dtype=float32)\n'
        '), global_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, dtype=float32)\n'
        '))')

    out = conv(x1, pos1, n1, edge_index)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, pos1, n1, adj1.t()), out, atol=1e-3)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, pos1, n1, adj2.t()), out, atol=1e-3)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, pos1, n1, edge_index), out, atol=1e-3)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, pos1, n1, adj2.t()), out, atol=1e-3)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out = conv(x1, (pos1, pos2), (n1, n2), edge_index)
    assert tuple(out.shape)== (2, 32)
    assert paddle.allclose(conv((x1, None), (pos1, pos2), (n1, n2), edge_index),
                          out, atol=1e-3)
    assert paddle.allclose(conv(x1, (pos1, pos2), (n1, n2), adj1.t()), out,
                          atol=1e-3)
    assert paddle.allclose(conv((x1, None), (pos1, pos2), (n1, n2), adj1.t()),
                          out, atol=1e-3)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv(x1, (pos1, pos2), (n1, n2), adj2.t()), out,
                              atol=1e-3)
        assert paddle.allclose(
            conv((x1, None), (pos1, pos2), (n1, n2), adj2.t()), out, atol=1e-3)

    if is_full_test():
        assert paddle.allclose(
            jit((x1, None), (pos1, pos2), (n1, n2), edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(
                jit((x1, None), (pos1, pos2), (n1, n2), adj2.t()), out,
                atol=1e-3)
