import paddle

from paddle_geometric.nn import GravNetConv
from paddle_geometric.testing import is_full_test, withPackage


@withPackage('paddle_cluster')
def test_gravnet_conv():
    x1 = paddle.randn(shape=[8, 16])
    x2 = paddle.randn(shape=[4, 16])
    batch1 = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch2 = paddle.to_tensor([0, 0, 1, 1])

    conv = GravNetConv(16, 32, space_dimensions=4, propagate_dimensions=8, k=2)
    assert str(conv) == 'GravNetConv(16, 32, k=2)'

    out11 = conv(x1)
    assert tuple(out11.shape)== (8, 32)

    out12 = conv(x1, batch1)
    assert tuple(out12.shape)== (8, 32)

    out21 = conv((x1, x2))
    assert tuple(out21.shape)== (4, 32)

    out22 = conv((x1, x2), (batch1, batch2))
    assert tuple(out22.shape)== (4, 32)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1), out11)
        assert paddle.allclose(jit(x1, batch1), out12)

        assert paddle.allclose(jit((x1, x2)), out21)
        assert paddle.allclose(jit((x1, x2), (batch1, batch2)), out22)
