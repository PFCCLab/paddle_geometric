import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MLPAggregation, SAGEConv
from paddle_geometric.typing import SparseTensor



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


def test_sage_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(8, 32, project=False, aggr='mean')
    assert 'SAGEConv(8, 32, aggr=mean)' in str(conv)

    out = conv(x, edge_index)
    assert out.shape == [4, 32]

    # Test bipartite message passing:
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])

    conv = SAGEConv((8, 16), 32, project=False, aggr='mean')
    assert str(conv) == 'SAGEConv((8, 16), 32, aggr=mean)'

    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, size=(4, 2))
    assert out1.shape == [2, 32]
    assert out2.shape == [2, 32]


def test_lazy_sage_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    with pytest.raises(ValueError, match="does not support lazy"):
        SAGEConv(-1, 32, project=True)

    conv = SAGEConv(-1, 32, project=False)
    assert str(conv) == 'SAGEConv(-1, 32, aggr=mean)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]


def test_lstm_aggr_sage_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(8, 32, aggr='lstm')
    assert str(conv) == 'SAGEConv(8, 32, aggr=lstm)'

    out = conv(x, edge_index)
    assert out.shape == [4, 32]


def test_mlp_sage_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(
        in_channels=8,
        out_channels=32,
        aggr=MLPAggregation(
            in_channels=8,
            out_channels=8,
            max_num_elements=2,
            num_layers=1,
        ),
    )

    out = conv(x, edge_index)
    assert out.shape == [4, 32]


@pytest.mark.parametrize('aggr_kwargs', [
    dict(mode='cat'),
    dict(mode='proj', mode_kwargs=dict(in_channels=8, out_channels=16)),
    dict(mode='attn', mode_kwargs=dict(in_channels=8, out_channels=16,
                                       num_heads=4)),
    dict(mode='sum'),
])
def test_multi_aggr_sage_conv(aggr_kwargs):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    aggr_kwargs['aggrs_kwargs'] = [{}, {}, {}, dict(learn=True, t=1)]
    conv = SAGEConv(8, 32, aggr=['mean', 'max', 'sum', 'softmax'],
                    aggr_kwargs=aggr_kwargs)

    out = conv(x, edge_index)
    assert out.shape == [4, 32]