import paddle

from paddle_geometric.nn import XConv
from paddle_geometric.testing import is_full_test, withPackage


@withPackage('paddle_cluster')
def test_x_conv():
    x = paddle.randn(shape=[8, 16])
    pos = paddle.rand([8, 3])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])

    conv = XConv(16, 32, dim=3, kernel_size=2, dilation=2)
    assert str(conv) == 'XConv(16, 32)'

    paddle.seed(12345)
    out1 = conv(x, pos)
    assert tuple(out1.shape)== (8, 32)

    paddle.seed(12345)
    out2 = conv(x, pos, batch)
    assert tuple(out2.shape)== (8, 32)

    if is_full_test():
        jit = paddle.jit.to_static(conv)

        paddle.seed(12345)
        assert paddle.allclose(jit(x, pos), out1, atol=1e-6)

        paddle.seed(12345)
        assert paddle.allclose(jit(x, pos, batch), out2, atol=1e-6)
