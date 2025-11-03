import warnings
from typing import List, Union

import numpy as np
import paddle

from paddle_geometric.utils import remove_self_loops, to_undirected


def erdos_renyi_graph(num_nodes: int, edge_prob: float,
                      directed: bool = False) -> paddle.Tensor:
    """Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)

    Examples:
        >>> erdos_renyi_graph(5, 0.2, directed=False)
        tensor([[0, 1, 1, 4],
                [1, 0, 4, 1]])

        >>> erdos_renyi_graph(5, 0.2, directed=True)
        tensor([[0, 1, 3, 3, 4, 4],
                [4, 3, 1, 2, 1, 3]])
    """
    if directed:
        idx = paddle.arange(end=(num_nodes - 1) * num_nodes)
        idx = idx.view(num_nodes - 1, num_nodes)
        idx = idx + paddle.arange(start=1, end=num_nodes).view(-1, 1)
        idx = idx.view(-1)
    else:
        warnings.filterwarnings("ignore", ".*pass the indexing argument.*")
        idx = paddle.combinations(x=paddle.arange(end=num_nodes), r=2)
    mask = paddle.rand(shape=idx.shape[0]) < edge_prob
    idx = idx[mask]
    if directed:
        row = idx.div(num_nodes, rounding_mode="floor")
        col = idx % num_nodes
        col = col.astype(row.dtype)
        edge_index = paddle.stack(x=[row, col], axis=0)
    else:
        edge_index = to_undirected(idx.t(), num_nodes=num_nodes)
    return edge_index


def stochastic_blockmodel_graph(
    block_sizes: Union[List[int], paddle.Tensor],
    edge_probs: Union[List[List[float]], paddle.Tensor],
    directed: bool = False,
) -> paddle.Tensor:
    """Returns the :obj:`edge_index` of a stochastic blockmodel graph.

    Args:
        block_sizes ([int] or LongTensor): The sizes of blocks.
        edge_probs ([[float]] or FloatTensor): The density of edges going
            from each block to each other block. Must be symmetric if the
            graph is undirected.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)

    Examples:
        >>> block_sizes = [2, 2, 4]
        >>> edge_probs = [[0.25, 0.05, 0.02],
        ...               [0.05, 0.35, 0.07],
        ...               [0.02, 0.07, 0.40]]
        >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
        ...                             directed=False)
        tensor([[2, 4, 4, 5, 5, 6, 7, 7],
                [5, 6, 7, 2, 7, 4, 4, 5]])

        >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
        ...                             directed=True)
        tensor([[0, 2, 3, 4, 4, 5, 5],
                [3, 4, 1, 5, 6, 6, 7]])
    """
    size, prob = block_sizes, edge_probs
    if not isinstance(size, paddle.Tensor):
        size = paddle.tensor(size, dtype="int64")
    if not isinstance(prob, paddle.Tensor):
        prob = paddle.tensor(prob, dtype="float32")
    assert size.dim() == 1
    assert prob.dim() == 2 and prob.shape[0] == prob.shape[1]
    assert size.shape[0] == prob.shape[0]
    if not directed:
        assert paddle.allclose(x=prob, y=prob.t()).item()
    node_idx = paddle.concat(x=[
        paddle.full(shape=(b, ), fill_value=i, dtype=size.dtype)
        for i, b in enumerate(size)
    ])
    num_nodes = node_idx.shape[0]
    if directed:
        idx = paddle.arange(end=(num_nodes - 1) * num_nodes)
        idx = idx.view(num_nodes - 1, num_nodes)
        idx = idx + paddle.arange(start=1, end=num_nodes).view(-1, 1)
        idx = idx.view(-1)
        row = idx.div(num_nodes, rounding_mode="floor")
        col = idx % num_nodes
    else:
        row, col = paddle.combinations(x=paddle.arange(end=num_nodes), r=2).t()
    mask = paddle.bernoulli(x=prob[node_idx[row.cast(paddle.int64)],
                                   node_idx[col]]).to("bool")
    edge_index = paddle.stack(x=[row[mask], col[mask].cast(row.dtype)], axis=0)
    if not directed:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def barabasi_albert_graph(num_nodes: int, num_edges: int) -> paddle.Tensor:
    """Returns the :obj:`edge_index` of a Barabasi-Albert preferential
    attachment model, where a graph of :obj:`num_nodes` nodes grows by
    attaching new nodes with :obj:`num_edges` edges that are preferentially
    attached to existing nodes with high degree.

    Args:
        num_nodes (int): The number of nodes.
        num_edges (int): The number of edges from a new node to existing nodes.

    Example:
        >>> barabasi_albert_graph(num_nodes=4, num_edges=3)
        tensor([[0, 0, 0, 1, 1, 2, 2, 3],
                [1, 2, 3, 0, 2, 0, 1, 0]])
    """
    assert num_edges > 0 and num_edges < num_nodes
    row, col = paddle.arange(end=num_edges), paddle.randperm(n=num_edges)
    for i in range(num_edges, num_nodes):
        row = paddle.concat(
            x=[row, paddle.full((num_edges, ), i, dtype="int64")])
        choice = np.random.choice(
            paddle.concat(x=[row, col]).numpy(), num_edges)
        col = paddle.concat(x=[col, paddle.to_tensor(data=choice)])
    edge_index = paddle.stack(x=[row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index
