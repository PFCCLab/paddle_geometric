from typing import Optional, Tuple, Union, overload

import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.utils import scatter
from paddle_geometric.utils.num_nodes import maybe_num_nodes
from paddle_geometric.utils.sparse import (
    is_paddle_sparse_tensor,
    to_edge_index,
    to_paddle_coo_tensor,
    to_paddle_csr_tensor,
)


def contains_self_loops(edge_index: Tensor) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


@overload
def remove_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
) -> Tuple[Tensor, None]:
    ...


@overload
def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
) -> Tuple[Tensor, Tensor]:
    ...


@overload
def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    ...


def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    size: Optional[Tuple[int, int]] = None

    layout = None  # 'coo', 'csr'

    value: Optional[Tensor] = None
    if is_paddle_sparse_tensor(edge_index):
        size = (edge_index.shape[0], edge_index.shape[1])
        edge_index, value = to_edge_index(edge_index)

    is_undirected = False
    if isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if isinstance(edge_index, EdgeIndex):
        edge_index._is_undirected = is_undirected

    # if edge_attr is None:
    #     return edge_index, None
    # else:
    #     return edge_index, edge_attr[mask]
    if layout is not None:
        assert edge_attr is None
        assert value is not None
        value = value[mask]
        if str(layout) == 'coo':  # str(...) for TorchScript :(
            return to_paddle_coo_tensor(edge_index, value, size, True), None
        elif str(layout) == 'csr':
            return to_paddle_csr_tensor(edge_index, value, size, True), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


@overload
def segregate_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
) -> Tuple[Tensor, None, Tensor, None]:
    ...


@overload
def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    ...


@overload
def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    ...


def segregate_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    is_undirected = False
    if isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    if isinstance(edge_index, EdgeIndex):
        assert isinstance(loop_edge_index, EdgeIndex)
        edge_index._is_undirected = is_undirected
        loop_edge_index._is_undirected = is_undirected

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


@overload
def add_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


def add_self_loops(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`paddle.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`).
            (default: :obj:`1.`)
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`)

    Returns:
        A tuple containing the new `edge_index` and the associated `edge_attr`.

    """
    layout = None

    is_sparse = is_paddle_sparse_tensor(edge_index)
    value: Optional[paddle.Tensor] = None
    if is_sparse:
        assert edge_attr is None
        if edge_index.is_sparse_coo():
            layout = "coo"
        elif edge_index.is_sparse_csr():
            layout = "csr"
        else:
            raise NotImplementedError()

        size = edge_index.shape[0], edge_index.shape[1]
        N = min(size)
        edge_index, value = to_edge_index(edge_index)
    elif isinstance(num_nodes, (tuple, list)):
        size = num_nodes[0], num_nodes[1]
        N = min(size)
    else:
        N = maybe_num_nodes(edge_index, num_nodes)
        size = N, N

    if isinstance(edge_index, EdgeIndex):
        loop_index: Tensor = EdgeIndex(
            paddle.arange(start=0, end=N).view(1,
                                               -1).tile(repeat_times=[2, 1]),
            sparse_size=(N, N),
            is_undirected=True,
        )
    else:
        loop_index = paddle.arange(start=0,
                                   end=N).view(1, -1).tile(repeat_times=[2, 1])
    full_edge_index = paddle.concat(x=[edge_index, loop_index], axis=1)
    if is_sparse:
        assert edge_attr is None
        assert value is not None
        loop_attr = compute_loop_attr(edge_index, value, N, is_sparse,
                                      fill_value)
        value = paddle.concat(x=[value, loop_attr], axis=0)
        if layout == "coo":
            return to_paddle_coo_tensor(full_edge_index, value, size), None
        elif layout == "csr":
            return to_paddle_csr_tensor(full_edge_index, value, size), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")
    if edge_attr is not None:
        loop_attr = compute_loop_attr(edge_index, edge_attr, N, is_sparse,
                                      fill_value)
        edge_attr = paddle.concat(x=[edge_attr, loop_attr], axis=0)
    return full_edge_index, edge_attr


@overload
def add_remaining_self_loops(
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: None = None,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, None]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[float] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


@overload
def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    fill_value: Optional[str] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    pass


def add_remaining_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    fill_value: Optional[Union[float, Tensor, str]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Adds remaining self-loops :math:`(i, i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`paddle.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`).
            (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        Tuple[Tensor, Optional[Tensor]]: The updated edge indices and edge attributes.
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    if isinstance(edge_index, EdgeIndex):
        loop_index: Tensor = EdgeIndex(
            paddle.arange(start=0, end=N).view(1,
                                               -1).tile(repeat_times=[2, 1]),
            sparse_size=(N, N),
            is_undirected=True,
        )
    else:
        loop_index = paddle.arange(start=0,
                                   end=N).view(1, -1).tile(repeat_times=[2, 1])
    if edge_attr is not None:
        loop_attr = compute_loop_attr(edge_index, edge_attr, N, False,
                                      fill_value)
        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]
        edge_attr = paddle.concat(x=[edge_attr[mask], loop_attr], axis=0)
    is_undirected = False
    if isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected
    edge_index = edge_index[:, mask]
    if isinstance(edge_index, EdgeIndex):
        edge_index._is_undirected = is_undirected
    edge_index = paddle.concat(x=[edge_index, loop_index], axis=1)
    return edge_index, edge_attr


def get_self_loop_attr(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        Tensor: The self-loop attributes.

    Examples:
        >>> edge_index = paddle.to_tensor([[0, 1, 0],
        ...                                [1, 0, 0]])
        >>> edge_weight = paddle.to_tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        Tensor([0.5, 0.0])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        Tensor([0.5, 0.0, 0.0, 0.0])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = paddle.ones(loop_index.shape[0], dtype=edge_index.dtype)

    num_nodes = num_nodes if num_nodes is not None else int(
        paddle.max(edge_index) + 1)
    full_loop_attr = paddle.zeros(
        shape=(num_nodes, ) + tuple(loop_attr.shape)[1:],
        dtype=loop_attr.dtype)
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


@overload
def compute_loop_attr(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[float] = None,
) -> Tensor:
    pass


@overload
def compute_loop_attr(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[Tensor] = None,
) -> Tensor:
    pass


@overload
def compute_loop_attr(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[str] = None,
) -> Tensor:
    pass


def compute_loop_attr(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
    is_sparse: bool,
    fill_value: Optional[Union[float, Tensor, str]] = None,
) -> Tensor:
    """Computes the attributes of self-loops in the graph given by `edge_index` and
    `edge_attr`. Missing self-loops will be added according to `fill_value`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor): The edge weights or multi-dimensional edge features.
        num_nodes (int): The number of nodes.
        is_sparse (bool): Whether the graph is sparse.
        fill_value (float, Tensor or str, optional): How to compute self-loop
            attributes for missing self-loops. Defaults to 1.0 or as specified.

    Returns:
        Tensor: The computed self-loop attributes.
    """
    if fill_value is None:
        size = (num_nodes, ) + tuple(edge_attr.shape[1:])
        return paddle.ones(size, dtype=edge_attr.dtype)

    elif isinstance(fill_value, (int, float)):
        size = (num_nodes, ) + tuple(edge_attr.shape[1:])
        return paddle.full(size, fill_value, dtype=edge_attr.dtype)

    elif isinstance(fill_value, Tensor):
        size = (num_nodes, ) + tuple(edge_attr.shape)[1:]
        loop_attr = fill_value.to(edge_attr.place, edge_attr.dtype)
        if edge_attr.dim() != loop_attr.dim():
            loop_attr = loop_attr.unsqueeze(axis=0)
        return loop_attr.expand(shape=size).contiguous()

    elif isinstance(fill_value, str):
        col = edge_index[0] if is_sparse else edge_index[1]
        return scatter(edge_attr, col, 0, num_nodes, fill_value)

    raise AttributeError("No valid 'fill_value' provided")
