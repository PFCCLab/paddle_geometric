from typing import Optional, Tuple

import paddle

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.deprecation import deprecated
from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import cumsum, degree, sort_edge_index, subgraph
from paddle_geometric.utils.num_nodes import maybe_num_nodes

if paddle_geometric.typing.WITH_PADDLE_CLUSTER:
    import paddle_cluster_ops


def filter_adj(
        row: paddle.Tensor, col: paddle.Tensor, edge_attr: OptTensor,
        mask: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


@deprecated("use 'dropout_edge' instead")
def dropout_adj(
    edge_index: paddle.Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[paddle.Tensor, OptTensor]:
    """Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    .. warning::

        :class:`~torch_geometric.utils.dropout_adj` is deprecated and will
        be removed in a future release.
        Use :class:`torch_geometric.utils.dropout_edge` instead.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> dropout_adj(edge_index, edge_attr)
        (tensor([[0, 1, 2, 3],
                [1, 2, 3, 2]]),
        tensor([1, 3, 5, 6]))

        >>> # The returned graph is kept undirected
        >>> dropout_adj(edge_index, edge_attr, force_undirected=True)
        (tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]]),
        tensor([1, 3, 5, 1, 3, 5]))
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Dropout probability has to be between 0 and 1 (got {p}")
    if not training or p == 0.0:
        return edge_index, edge_attr
    row, col = edge_index
    mask = paddle.rand(shape=row.shape[0]) >= p
    if force_undirected:
        mask[row > col] = False
    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)
    if force_undirected:
        edge_index = paddle.stack(
            x=[
                paddle.concat(x=[row, col], axis=0),
                paddle.concat(x=[col, row], axis=0),
            ],
            axis=0,
        )
        if edge_attr is not None:
            edge_attr = paddle.concat(x=[edge_attr, edge_attr], axis=0)
    else:
        edge_index = paddle.stack(x=[row, col], axis=0)
    return edge_index, edge_attr


def dropout_node(
    edge_index: paddle.Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
    relabel_nodes: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
        relabel_nodes (bool, optional): If set to `True`, the resulting
            `edge_index` will be relabeled to hold consecutive indices
            starting from zero.

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Dropout probability has to be between 0 and 1 (got {p}")
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if not training or p == 0.0:
        node_mask = paddle.ones(shape=num_nodes, dtype="bool")
        edge_mask = paddle.ones(shape=edge_index.shape[1], dtype="bool")
        return edge_index, edge_mask, node_mask
    prob = paddle.rand(shape=num_nodes)
    node_mask = prob > p
    edge_index, _, edge_mask = subgraph(
        node_mask,
        edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        return_edge_mask=True,
    )
    return edge_index, edge_mask, node_mask


def dropout_edge(
    edge_index: paddle.Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    training: bool = True,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Dropout probability has to be between 0 and 1 (got {p}")
    if not training or p == 0.0:
        edge_mask = paddle.ones(shape=edge_index.shape[1], dtype="bool")
        return edge_index, edge_mask
    row, col = edge_index
    edge_mask = paddle.rand(shape=row.shape[0]) >= p
    if force_undirected:
        edge_mask[row > col] = False
    edge_index = edge_index[:, edge_mask]
    if force_undirected:
        edge_index = paddle.concat(x=[edge_index,
                                      edge_index.flip(axis=0)], axis=1)
        edge_mask = edge_mask.nonzero().tile(repeat_times=(2, 1)).squeeze()
    return edge_index, edge_mask


def dropout_path(
    edge_index: paddle.Tensor,
    p: float = 0.2,
    walks_per_node: int = 1,
    walk_length: int = 3,
    num_nodes: Optional[int] = None,
    is_sorted: bool = False,
    training: bool = True,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Drops edges from the adjacency matrix :obj:`edge_index`
    based on random walks. The source nodes to start random walks from are
    sampled from :obj:`edge_index` with probability :obj:`p`, following
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Sample probability. (default: :obj:`0.2`)
        walks_per_node (int, optional): The number of walks per node, same as
            :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`1`)
        walk_length (int, optional): The walk length, same as
            :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`3`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_path(edge_index)
        >>> edge_index
        tensor([[1, 2],
                [2, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([False, False,  True, False,  True, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Sample probability has to be between 0 and 1 (got {p}")
    num_edges = edge_index.shape[1]
    edge_mask = paddle.ones(shape=num_edges, dtype="bool")
    if not training or p == 0.0:
        return edge_index, edge_mask
    if not paddle_geometric.typing.WITH_PADDLE_CLUSTER or is_compiling():
        raise ImportError("`dropout_path` requires `torch-cluster`.")
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_orders = None
    ori_edge_index = edge_index
    if not is_sorted:
        edge_orders = paddle.arange(end=num_edges)
        edge_index, edge_orders = sort_edge_index(edge_index, edge_orders,
                                                  num_nodes=num_nodes)
    row, col = edge_index
    sample_mask = paddle.rand(shape=row.shape[0]) <= p
    start = row[sample_mask].tile(repeat_times=walks_per_node)
    rowptr = cumsum(degree(row, num_nodes=num_nodes, dtype="int64"))

    if start.is_empty():
        n_id = paddle.empty([walk_length + 1, 0], dtype=start.dtype)
        e_id = paddle.empty([walk_length, 0], dtype=start.dtype)
        return edge_index, edge_mask
    else:
        n_id, e_id = paddle_cluster_ops.random_walk(rowptr, col, start,
                                                    walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)
    if edge_orders is not None:
        e_id = edge_orders[e_id]
    edge_mask[e_id] = False
    edge_index = ori_edge_index[:, edge_mask]
    return edge_index, edge_mask
