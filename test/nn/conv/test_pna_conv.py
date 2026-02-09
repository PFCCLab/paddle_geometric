import paddle

from paddle_geometric.nn import PNAConv
from paddle_geometric.data import Data
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor
import pytest


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
    skip_if_cuda_error()


aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
scalers = [
    'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
]


def test_pna_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    deg = paddle.to_tensor([0, 3, 0, 1])
    value = paddle.rand([edge_index.shape[1], 3])
    adj1 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    aggregators = ['sum', 'mean']
    scalers = ['identity']
    conv = PNAConv(16, 32, aggregators, scalers, deg=deg, edge_dim=3, towers=1,
                   pre_layers=1, post_layers=1, divide_input=False)
    assert 'PNAConv(16, 32' in str(conv)

    out = conv(x, edge_index, value)
    assert out.shape == [4, 32]

    assert paddle.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if SparseTensor is not None:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x, adj2.t()), out, atol=1e-6)


@pytest.mark.parametrize('divide_input', [True, False])
def test_pna_conv_divide_input(divide_input):
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    deg = paddle.to_tensor([0, 3, 0, 1])
    value = paddle.rand([edge_index.shape[1], 3])
    
    aggregators = ['sum', 'mean']
    scalers = ['identity']
    conv = PNAConv(16, 32, aggregators, scalers, deg=deg, edge_dim=3, towers=2,
                   pre_layers=1, post_layers=1, divide_input=divide_input)
    
    out = conv(x, edge_index, value)
    assert out.shape == [4, 32]


def test_pna_conv_get_degree_histogram_dataloader():
    from paddle_geometric.loader import DataLoader
    
    edge_index_1 = paddle.to_tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 0, 0, 0]])
    edge_index_2 = paddle.to_tensor([[1, 1, 2, 2, 0, 3, 3], [2, 3, 3, 1, 1, 0, 2]])
    edge_index_3 = paddle.to_tensor([[1, 3, 2, 0, 0, 4, 2], [2, 0, 4, 1, 1, 0, 3]])
    edge_index_4 = paddle.to_tensor([[0, 1, 2, 4, 0, 1, 3], [2, 3, 3, 1, 1, 0, 2]])

    data_1 = Data(num_nodes=5, edge_index=edge_index_1)  # hist = [1, 2 ,1 ,1]
    data_2 = Data(num_nodes=5, edge_index=edge_index_2)  # hist = [1, 1, 3]
    data_3 = Data(num_nodes=5, edge_index=edge_index_3)  # hist = [0, 3, 2]
    data_4 = Data(num_nodes=5, edge_index=edge_index_4)  # hist = [1, 1, 3]

    loader = DataLoader(
        [data_1, data_2, data_3, data_4],
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    deg_hist = PNAConv.get_degree_histogram(loader)
    expected = paddle.to_tensor([3, 7, 9, 1])
    assert paddle.allclose(deg_hist, expected), f"Expected {expected}, got {deg_hist}"