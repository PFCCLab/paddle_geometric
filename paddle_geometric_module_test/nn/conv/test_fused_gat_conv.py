import paddle

from paddle_geometric.nn import FusedGATConv
from paddle_geometric.testing import onlyCUDA, withPackage


def test_to_graph_format() -> None:
    edge_index = paddle.to_tensor([[1, 0, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    assert csr[0].dtype == paddle.int32
    assert paddle.equal(csr[0], paddle.to_tensor([0, 1, 2, 3, 4], dtype=paddle.int32))
    assert csr[1].dtype == paddle.int32
    assert paddle.equal(csr[1], paddle.to_tensor([0, 0, 1, 1], dtype=paddle.int32))
    assert csc[0].dtype == paddle.int32
    assert paddle.equal(csc[0], paddle.to_tensor([0, 1, 2, 3], dtype=paddle.int32))
    assert csc[1].dtype == paddle.int32
    assert paddle.equal(csc[1], paddle.to_tensor([0, 2, 4, 4, 4], dtype=paddle.int32))
    assert perm.dtype == paddle.int32
    assert paddle.equal(perm, paddle.to_tensor([0, 1, 2, 3], dtype=paddle.int32))


@onlyCUDA
@withPackage('dgNN')
def test_fused_gat_conv() -> None:
    device = paddle.place('cuda')

    x = paddle.randn([4, 8], device=device)
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]], place=device)

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False).to(device)
    assert str(conv) == 'FusedGATConv(8, 32, heads=2)'

    out = conv(x, csr, csc, perm)
    assert tuple(out.shape)== (4, 64)
