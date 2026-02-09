import paddle

import pytest

from paddle_geometric.nn import ResGatedGraphConv
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



@pytest.mark.parametrize('edge_dim', [None, 4])
def test_res_gated_graph_conv(edge_dim):
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 32])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn([edge_index.shape[1], edge_dim]) if edge_dim else None
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = ResGatedGraphConv(8, 32, edge_dim=edge_dim)
    assert str(conv) == 'ResGatedGraphConv(8, 32)'

    out = conv(x1, edge_index, edge_attr)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, adj1.t(), edge_attr), out, atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1_bipartite = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = ResGatedGraphConv((8, 32), 32, edge_dim=edge_dim)
    assert str(conv) == 'ResGatedGraphConv((8, 32), 32)'

    out = conv((x1, x2), edge_index, edge_attr)
    assert out.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), adj1_bipartite.t(), edge_attr), out, atol=1e-6)

    if SparseTensor is not None:
        adj2_bipartite = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 2))
        assert paddle.allclose(conv((x1, x2), adj2_bipartite.t()), out, atol=1e-6)