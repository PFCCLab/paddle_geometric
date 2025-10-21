import paddle

from paddle_geometric.utils import to_paddle_coo_tensor
from paddle_geometric.utils.num_nodes import (
    maybe_num_nodes,
    maybe_num_nodes_dict,
)


def test_maybe_num_nodes():
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 2],
                                   [1, 2, 0, 2, 0, 1, 1]])

    assert maybe_num_nodes(edge_index, 4) == 4
    assert maybe_num_nodes(edge_index) == 3

    adj = to_paddle_coo_tensor(edge_index)
    assert maybe_num_nodes(adj, 4) == 4
    assert maybe_num_nodes(adj) == 3


def test_maybe_num_nodes_dict():
    edge_index_dict = {
        '1': paddle.to_tensor([[0, 0, 1, 1, 2, 2, 2], [1, 2, 0, 2, 0, 1, 1]]),
        '2': paddle.to_tensor([[0, 0, 1, 3], [1, 2, 0, 4]])
    }
    num_nodes_dict = {'2': 6}

    assert maybe_num_nodes_dict(edge_index_dict) == {'1': 3, '2': 5}
    assert maybe_num_nodes_dict(edge_index_dict, num_nodes_dict) == {
        '1': 3,
        '2': 6,
    }
