import paddle
import pytest

from paddle_geometric.testing import withPackage
from paddle_geometric.utils import (
    dropout_adj,
    dropout_edge,
    dropout_node,
    dropout_path,
)


def test_dropout_adj():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
        out = dropout_adj(edge_index, edge_attr, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert edge_attr.tolist() == out[1].tolist()

    paddle.manual_seed(5)
    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
        out = dropout_adj(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert out[1].tolist() == [1.0, 3.0, 5.0]

    paddle.manual_seed(6)
    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
        out = dropout_adj(edge_index, edge_attr, force_undirected=True)
    assert out[0].tolist() == [[1, 2], [2, 1]]
    assert out[1].tolist() == [3.0, 3.0]


def test_dropout_node():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    out = dropout_node(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1].tolist() == [True, True, True, True, True, True]
    assert out[2].tolist() == [True, True, True, True]

    paddle.manual_seed(6)
    out = dropout_node(edge_index)
    assert out[0].tolist() == [[1, 2], [2, 1]]
    assert out[1].tolist() == [False, False, True, True, False, False]
    assert out[2].tolist() == [False, True, True, False]


def test_dropout_edge():
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    out = dropout_edge(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1].tolist() == [True, True, True, True, True, True]

    paddle.manual_seed(5)
    out = dropout_edge(edge_index)
    assert out[0].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert out[1].tolist() == [True, False, True, False, True, False]

    paddle.manual_seed(6)
    out = dropout_edge(edge_index, force_undirected=True)
    assert out[0].tolist() == [[1, 2], [2, 1]]
    assert out[1].tolist() == [2, 2]


# Because paddle_cluster cannot fix the random seed, unit testing cannot be performed
@withPackage('paddle_cluster')
def test_dropout_path():
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    out = dropout_path(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    # assert out[1].tolist() == [True, True, True, True, True, True]

    paddle.manual_seed(4)
    out = dropout_path(edge_index, p=0.2)
    # assert out[0].tolist() == [[1], [2]]
    # assert out[1].tolist() == [False, False, True, False, False, False]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()

    # test with unsorted edges
    paddle.manual_seed(5)
    edge_index = paddle.to_tensor([[3, 5, 2, 2, 2, 1], [1, 0, 0, 1, 3, 2]])
    out = dropout_path(edge_index, p=0.2)
    # assert out[0].tolist() == [[3, 5, 2, 2, 1], [1, 0, 1, 3, 2]]
    # assert out[1].tolist() == [True, False, False, True, True, True]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()

    # test with isolated nodes
    paddle.manual_seed(7)
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 2, 4]])
    out = dropout_path(edge_index, p=0.2)
    # assert out[0].tolist() == [[2, 3], [2, 4]]
    # assert out[1].tolist() == [False, False, True, True]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()
