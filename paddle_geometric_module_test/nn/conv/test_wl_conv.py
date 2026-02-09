import paddle

import paddle_geometric.typing
from paddle_geometric.nn import WLConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import one_hot, to_paddle_csc_tensor


def test_wl_conv():
    x1 = paddle.to_tensor([1, 0, 0, 1])
    x2 = one_hot(x1)
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = WLConv()
    assert str(conv) == 'WLConv()'

    out = conv(x1, edge_index)
    assert out.tolist() == [0, 1, 1, 0]
    assert paddle.equal(conv(x2, edge_index), out).all()
    assert paddle.equal(conv(x1, adj1.t()), out).all()
    assert paddle.equal(conv(x2, adj1.t()), out).all()

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.equal(conv(x1, adj2.t()), out).all()
        assert paddle.equal(conv(x2, adj2.t()), out).all()

    assert conv.histogram(out).tolist() == [[2, 2]]
    assert paddle.allclose(conv.histogram(out, norm=True),
                          paddle.to_tensor([[0.7071, 0.7071]]))
