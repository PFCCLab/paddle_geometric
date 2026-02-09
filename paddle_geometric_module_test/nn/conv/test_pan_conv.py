import paddle

import paddle_geometric.typing
from paddle_geometric.nn import PANConv
from paddle_geometric.testing import withPackage
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@withPackage('paddle_sparse')  # TODO `PANConv` returns a `SparseTensor`.
def test_pan_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))

    conv = PANConv(16, 32, filter_size=2)
    assert str(conv) == 'PANConv(16, 32, filter_size=2)'
    out1, M1 = conv(x, edge_index)
    assert tuple(out1.shape)== (4, 32)

    out2, M2 = conv(x, adj1.t())
    assert paddle.allclose(out1, out2, atol=1e-6)
    assert paddle.allclose(M1.to_dense(), M2.to_dense())

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        out3, M3 = conv(x, adj2.t())
        assert paddle.allclose(out1, out3, atol=1e-6)
        assert paddle.allclose(M1.to_dense(), M3.to_dense())
