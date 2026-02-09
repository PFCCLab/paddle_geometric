import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

import paddle_geometric.typing
from paddle_geometric.nn import NNConv
from paddle_geometric.testing import is_full_test, withCUDA
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_coo_tensor


@withCUDA
def test_nn_conv(device):
    x1 = paddle.randn([4, 8], device=device)
    x2 = paddle.randn([2, 16], device=device)
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]], place=device)
    value = paddle.rand([edge_index.shape[1], 3]).to(device)
    adj1 = to_paddle_coo_tensor(edge_index, value, size=(4, 4))

    nn = Seq(Lin(3, 32), ReLU(), Lin(32, 8 * 32))
    conv = NNConv(8, 32, nn=nn).to(device)
    assert str(conv) == (
        'NNConv(8, 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, bias=True)\n'
        '))')

    out = conv(x1, edge_index, value)
    assert tuple(out.shape)== (4, 32)
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)
    assert paddle.allclose(conv(x1, adj1.transpose(0, 1).coalesce()), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index, value), out)
        assert paddle.allclose(jit(x1, edge_index, value, size=(4, 4)), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out)

    # Test bipartite message passing:
    adj1 = to_paddle_coo_tensor(edge_index, value, size=(4, 2))

    conv = NNConv((8, 16), 32, nn=nn).to(device)
    assert str(conv) == (
        'NNConv((8, 16), 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, bias=True)\n'
        '))')

    out1 = conv((x1, x2), edge_index, value)
    assert tuple(out1.shape)== (2, 32)
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)
    assert paddle.allclose(conv((x1, x2),
                               adj1.transpose(0, 1).coalesce()), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert tuple(out2.shape)== (2, 32)
    assert paddle.allclose(conv((x1, None),
                               adj1.transpose(0, 1).coalesce()), out2)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj2.t()), out1)
        assert paddle.allclose(conv((x1, None), adj2.t()), out2)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit((x1, x2), edge_index, value), out1)
        assert paddle.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1)
        assert paddle.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out1)
            assert paddle.allclose(jit((x1, None), adj2.t()), out2)
