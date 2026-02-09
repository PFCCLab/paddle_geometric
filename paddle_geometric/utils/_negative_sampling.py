import random
from typing import Optional, Tuple, Union

import numpy as np
import paddle

from paddle_geometric.utils import coalesce, cumsum, degree, remove_self_loops
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def negative_sampling(
    edge_index: paddle.Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> paddle.Tensor:
    """Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ["sparse", "dense"]
    if edge_index.dtype != paddle.int64:
        edge_index = edge_index.astype("int64")
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(num_nodes, int):
        size = num_nodes, num_nodes
        bipartite = False
    else:
        size = num_nodes
        bipartite = True
        force_undirected = False
    idx, population = edge_index_to_vector(edge_index, size, bipartite,
                                           force_undirected)
    if idx.size >= population:
        return paddle.empty(shape=(2, 0), dtype=edge_index.dtype)
    if num_neg_samples is None:
        num_neg_samples = edge_index.shape[1]
    if force_undirected:
        num_neg_samples = num_neg_samples // 2
    prob = 1.0 - idx.size / population
    sample_size = int(1.1 * num_neg_samples / prob)
    neg_idx: Optional[paddle.Tensor] = None
    if method == "dense":
        mask = paddle.ones(shape=population, dtype="bool")
        mask[idx] = False
        for _ in range(3):
            rnd = sample(population, sample_size, idx.place)
            rnd = rnd[mask[rnd]]
            neg_idx = rnd if neg_idx is None else paddle.concat(
                x=[neg_idx, rnd])
            if neg_idx.size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False
    else:
        idx = idx.to("cpu")
        for _ in range(3):
            rnd = sample(population, sample_size, device="cpu")
            mask = np.isin(rnd.numpy(), idx.numpy())
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to("cpu"))
            mask = paddle.to_tensor(data=mask).to("bool")
            rnd = rnd[~mask].to(edge_index.place)
            neg_idx = rnd if neg_idx is None else paddle.concat(
                x=[neg_idx, rnd])
            if neg_idx.size >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
    assert neg_idx is not None
    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
    edge_index: paddle.Tensor,
    batch: Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> paddle.Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\\mathbf{b} \\in {\\{ 0, \\ldots, B-1\\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if edge_index.dtype != paddle.int64:
        edge_index = edge_index.astype("int64")
    if isinstance(batch, paddle.Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]
    split = degree(src_batch[edge_index[0]], dtype="int64").tolist()
    edge_indices = paddle.compat.split(edge_index, split, dim=1)
    num_src = degree(src_batch, dtype="int64")
    cum_src = cumsum(num_src)[:-1]
    if isinstance(batch, paddle.Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype="int64")
        cum_dst = cumsum(num_dst)[:-1]
        num_nodes = paddle.stack(x=[num_src, num_dst], axis=1).tolist()
        ptr = paddle.stack(x=[cum_src, cum_dst], axis=1).unsqueeze(axis=-1)
    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_sampling(edge_index, num_nodes[i],
                                           num_neg_samples, method,
                                           force_undirected)
        neg_edge_index += ptr[i].cast(neg_edge_index.dtype)
        neg_edge_indices.append(neg_edge_index)
    return paddle.concat(x=neg_edge_indices, axis=1)


def structured_negative_sampling(
    edge_index: paddle.Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    if edge_index.dtype != paddle.int64:
        edge_index = edge_index.astype("int64")
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = paddle.arange(end=num_nodes) * (num_nodes + 1)
        pos_idx = paddle.concat(x=[pos_idx, loop_idx], axis=0)
    rand = paddle.randint(low=0, high=num_nodes, shape=(row.shape[0], ),
                          dtype="int64")
    neg_idx = row * num_nodes + rand
    mask = paddle.to_tensor(data=np.isin(neg_idx, pos_idx)).to("bool")
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.size > 0:
        tmp = paddle.randint(low=0, high=num_nodes, shape=(rest.shape[0], ),
                             dtype="int64")
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp
        mask = paddle.to_tensor(data=np.isin(neg_idx, pos_idx)).to("bool")
        rest = rest[mask]
    return edge_index[0], edge_index[1], rand.to(edge_index.place)


def structured_negative_sampling_feasible(
    edge_index: paddle.Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    """Returns :obj:`True` if
    :meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
    if at least one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:
        >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1
    deg = degree(edge_index[0], num_nodes)
    return bool(paddle.all(x=deg < max_num_neighbors))


def sample(population: int, k: int,
           device: Optional[str] = None) -> paddle.Tensor:
    if population <= k:
        return paddle.arange(end=population, dtype="int64")
    else:
        return paddle.to_tensor(random.sample(range(population), k),
                                dtype="int64",
                                place=device)


def edge_index_to_vector(
    edge_index: paddle.Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[paddle.Tensor, int]:
    row, col = edge_index
    if bipartite:
        idx = (row * size[1]).add_(y=paddle.to_tensor(col))
        population = size[0] * size[1]
        return idx, population
    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]
        mask = row < col
        row, col = row[mask], col[mask]
        offset = paddle.arange(start=1, end=num_nodes).cumsum(0)[row]
        idx = (row.multiply_(y=paddle.to_tensor(num_nodes)).add_(
            y=paddle.to_tensor(col)).subtract_(y=paddle.to_tensor(offset)))
        population = num_nodes * (num_nodes + 1) // 2 - num_nodes
        return idx, population
    else:
        assert size[0] == size[1]
        num_nodes = size[0]
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.multiply_(y=paddle.to_tensor(num_nodes - 1)).add_(
            y=paddle.to_tensor(col))
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(
    idx: paddle.Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> paddle.Tensor:
    if bipartite:
        row = paddle.floor_divide(idx, size[1])
        col = idx % size[1]
        col = col.astype(row.dtype)
        return paddle.stack(x=[row, col], axis=0)
    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]
        offset = paddle.arange(start=1, end=num_nodes).cumsum(0)
        end = paddle.arange(start=num_nodes, end=num_nodes * num_nodes,
                            step=num_nodes)
        row = paddle.bucketize(
            x=idx, sorted_sequence=end.subtract_(y=paddle.to_tensor(offset)),
            right=True)
        col = offset[row].add_(y=paddle.to_tensor(idx)) % num_nodes
        return paddle.stack(
            x=[paddle.concat(x=[row, col]),
               paddle.concat(x=[col, row])], axis=0)
    else:
        assert size[0] == size[1]
        num_nodes = size[0]
        row = paddle.floor_divide(idx, num_nodes - 1)
        col = idx % (num_nodes - 1)
        col = col.astype(row.dtype)
        col[row <= col] += 1
        return paddle.stack(x=[row, col], axis=0)
