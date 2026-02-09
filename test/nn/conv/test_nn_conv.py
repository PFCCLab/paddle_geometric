import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

import pytest

from paddle_geometric.nn import NNConv
from paddle_geometric.utils import to_paddle_coo_tensor



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


def test_nn_conv():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.rand([edge_index.shape[1], 3])

    nn = Seq(Lin(3, 32), ReLU(), Lin(32, 8 * 32))
    conv = NNConv(8, 32, nn=nn)
    assert str(conv) == (
        'NNConv(8, 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, dtype=float32)\n'
        '))')

    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, size=(4, 4)), out)

    # Test bipartite message passing:
    conv = NNConv((8, 16), 32, nn=nn)
    assert str(conv) == (
        'NNConv((8, 16), 32, aggr=add, nn=Sequential(\n'
        '  (0): Linear(in_features=3, out_features=32, dtype=float32)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=256, dtype=float32)\n'
        '))')

    out1 = conv((x1, x2), edge_index, value)
    assert out1.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.shape == [2, 32]