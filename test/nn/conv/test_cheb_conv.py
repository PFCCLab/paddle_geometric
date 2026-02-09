import paddle

from paddle_geometric.data import Batch, Data
import pytest

from paddle_geometric.nn import ChebConv



def skip_if_cuda_error():
    try:
        import paddle
        paddle.set_device('cpu')
        return False
    except Exception as e:
        if 'cuda' in str(e).lower() or 'CUDA' in str(e):
            pytest.skip(f"CUDA 兼容性问题: {e}")
        return False


@pytest.fixture(autouse=True)
def setup_cpu_mode():
    """自动设置 CPU 模式"""
    skip_if_cuda_error()



def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = paddle.rand([edge_index.shape[1]])
    x = paddle.randn([num_nodes, in_channels])

    conv = ChebConv(in_channels, out_channels, K=3)
    assert str(conv) == 'ChebConv(16, 32, K=3, normalization=sym)'
    out1 = conv(x, edge_index)
    assert out1.shape == [num_nodes, out_channels]
    out2 = conv(x, edge_index, edge_weight)
    assert out2.shape == [num_nodes, out_channels]
    out3 = conv(x, edge_index, edge_weight, lambda_max=3.0)
    assert out3.shape == [num_nodes, out_channels]

    batch = paddle.to_tensor([0, 0, 1, 1])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = paddle.rand([edge_index.shape[1]])
    x = paddle.randn([num_nodes, in_channels])
    lambda_max = paddle.to_tensor([2.0, 3.0])

    out4 = conv(x, edge_index, edge_weight, batch)
    assert out4.shape == [num_nodes, out_channels]
    out5 = conv(x, edge_index, edge_weight, batch, lambda_max)
    assert out5.shape == [num_nodes, out_channels]


def test_cheb_conv_batch():
    x1 = paddle.randn([4, 8])
    edge_index1 = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight1 = paddle.rand([edge_index1.shape[1]])
    data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)

    x2 = paddle.randn([3, 8])
    edge_index2 = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight2 = paddle.rand([edge_index2.shape[1]])
    data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)

    conv = ChebConv(8, 16, K=2)

    out1 = conv(x1, edge_index1, edge_weight1)
    out2 = conv(x2, edge_index2, edge_weight2)

    batch = Batch.from_data_list([data1, data2])
    out = conv(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    assert out.shape == [7, 16]
    assert paddle.allclose(out1, out[:4], atol=1e-6)
    assert paddle.allclose(out2, out[4:], atol=1e-6)