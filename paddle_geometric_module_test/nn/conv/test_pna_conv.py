import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.data import Data
from paddle_geometric.loader import DataLoader, NeighborLoader
from paddle_geometric.nn import PNAConv
from paddle_geometric.testing import is_full_test, onlyNeighborSampler
from paddle_geometric.typing import SparseTensor

aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
scalers = [
    'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
]


@pytest.mark.parametrize('divide_input', [True, False])
def test_pna_conv(divide_input):
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    deg = paddle.to_tensor([0, 3, 0, 1])
    value = paddle.rand([edge_index.shape[1], 3])

    conv = PNAConv(16, 32, aggregators, scalers, deg=deg, edge_dim=3, towers=4,
                   pre_layers=2, post_layers=2, divide_input=divide_input)
    assert str(conv) == 'PNAConv(16, 32, towers=4, edge_dim=3)'

    out = conv(x, edge_index, value)
    assert tuple(out.shape)== (4, 32)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index, value), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)


@onlyNeighborSampler
def test_pna_conv_get_degree_histogram_neighbor_loader():
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 0, 0, 0]])
    data = Data(num_nodes=5, edge_index=edge_index)
    loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        input_nodes=None,
        batch_size=5,
        shuffle=False,
    )
    deg_hist = PNAConv.get_degree_histogram(loader)
    assert paddle.equal(deg_hist, paddle.to_tensor([1, 2, 1, 1]))


def test_pna_conv_get_degree_histogram_dataloader():
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
    )
    deg_hist = PNAConv.get_degree_histogram(loader)
    assert paddle.equal(deg_hist, paddle.to_tensor([3, 7, 9, 1]))
