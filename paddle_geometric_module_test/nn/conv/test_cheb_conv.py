import paddle

from paddle_geometric.data import Batch, Data
from paddle_geometric.nn import ChebConv
from paddle_geometric.testing import is_full_test


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = paddle.rand([edge_index.shape[1]])
    x = paddle.randn(shape=[num_nodes, in_channels])

    conv = ChebConv(in_channels, out_channels, K=3)
    assert str(conv) == 'ChebConv(16, 32, K=3, normalization=sym)'
    out1 = conv(x, edge_index)
    assert tuple(out1.shape)== (num_nodes, out_channels)
    out2 = conv(x, edge_index, edge_weight)
    assert tuple(out2.shape)== (num_nodes, out_channels)
    out3 = conv(x, edge_index, edge_weight, lambda_max=3.0)
    assert tuple(out3.shape)== (num_nodes, out_channels)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out1)
        assert paddle.allclose(jit(x, edge_index, edge_weight), out2)
        assert paddle.allclose(
            jit(x, edge_index, edge_weight, lambda_max=paddle.to_tensor(3.0)),
            out3)

    batch = paddle.to_tensor([0, 0, 1, 1])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = paddle.rand([edge_index.shape[1]])
    x = paddle.randn(shape=[num_nodes, in_channels])
    lambda_max = paddle.to_tensor([2.0, 3.0])

    out4 = conv(x, edge_index, edge_weight, batch)
    assert tuple(out4.shape)== (num_nodes, out_channels)
    out5 = conv(x, edge_index, edge_weight, batch, lambda_max)
    assert tuple(out5.shape)== (num_nodes, out_channels)

    if is_full_test():
        assert paddle.allclose(jit(x, edge_index, edge_weight, batch), out4)
        assert paddle.allclose(
            jit(x, edge_index, edge_weight, batch, lambda_max), out5)


def test_cheb_conv_batch():
    x1 = paddle.randn(shape=[4, 8])
    edge_index1 = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight1 = paddle.rand([edge_index1.shape[1]])
    data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)

    x2 = paddle.randn(shape=[3, 8])
    edge_index2 = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight2 = paddle.rand([edge_index2.shape[1]])
    data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)

    conv = ChebConv(8, 16, K=2)

    out1 = conv(x1, edge_index1, edge_weight1)
    out2 = conv(x2, edge_index2, edge_weight2)

    batch = Batch.from_data_list([data1, data2])
    out = conv(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    assert tuple(out.shape)== (7, 16)
    assert paddle.allclose(out1, out[:4], atol=1e-6)
    assert paddle.allclose(out2, out[4:], atol=1e-6)
