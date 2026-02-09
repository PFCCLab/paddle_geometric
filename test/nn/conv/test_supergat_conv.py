import paddle

import pytest

from paddle_geometric.nn import SuperGATConv



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



@pytest.mark.parametrize('att_type', ['MX', 'SD'])
def test_supergat_conv(att_type):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SuperGATConv(8, 32, heads=2, attention_type=att_type,
                        neg_sample_ratio=1.0, edge_sample_ratio=1.0)
    assert str(conv) == f'SuperGATConv(8, 32, heads=2, type={att_type})'

    out = conv(x, edge_index)
    assert out.shape == [4, 64]

    # Negative samples are given:
    neg_edge_index = conv.negative_sampling(edge_index, x.shape[0])
    assert out.shape == [4, 64]
    att_loss = conv.get_attention_loss()
    assert isinstance(att_loss, paddle.Tensor) and att_loss > 0

    # Batch of graphs:
    x = paddle.randn([8, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                   [0, 0, 1, 1, 4, 4, 5, 5]])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = conv(x, edge_index, batch=batch)
    assert out.shape == [8, 64]

    # Batch of graphs and negative samples are given:
    neg_edge_index = conv.negative_sampling(edge_index, x.shape[0], batch)
    assert out.shape == [8, 64]
    att_loss = conv.get_attention_loss()
    assert isinstance(att_loss, paddle.Tensor) and att_loss > 0