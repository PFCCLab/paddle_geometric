import paddle

import pytest

from paddle_geometric.nn import PointNetConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor



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


def test_point_net_conv():
    x1 = paddle.randn([4, 16])
    pos1 = paddle.randn([4, 3])
    pos2 = paddle.randn([2, 3])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    local_nn = paddle.nn.Sequential(
        paddle.nn.Linear(16 + 3, 32),
        paddle.nn.ReLU(),
        paddle.nn.Linear(32, 32)
    )
    global_nn = paddle.nn.Sequential(
        paddle.nn.Linear(32, 32)
    )
    conv = PointNetConv(local_nn, global_nn)
    assert 'PointNetConv(local_nn=Sequential(' in str(conv)
    assert 'global_nn=Sequential(' in str(conv)
    assert 'Linear(in_features=19, out_features=32' in str(conv)
    
    out = conv(x1, pos1, edge_index)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, pos1, adj1.t()), out, atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, pos1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out = conv(x1, (pos1, pos2), edge_index)
    assert out.shape == [2, 32]
    assert paddle.allclose(conv((x1, None), (pos1, pos2), edge_index), out)
    assert paddle.allclose(conv(x1, (pos1, pos2), adj1.t()), out, atol=1e-6)
    assert paddle.allclose(conv((x1, None), (pos1, pos2), adj1.t()), out,
                          atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert paddle.allclose(conv(x1, (pos1, pos2), adj2.t()), out, atol=1e-6)
        assert paddle.allclose(conv((x1, None), (pos1, pos2), adj2.t()), out,
                              atol=1e-6)