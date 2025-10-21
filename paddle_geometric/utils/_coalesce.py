import typing
from typing import List, Optional, Tuple, Union

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import EdgeIndex
from paddle_geometric.edge_index import SortOrder
from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import index_sort, scatter
from paddle_geometric.utils.num_nodes import maybe_num_nodes

if typing.TYPE_CHECKING:
    pass
else:
    pass

MISSING = '???'


def coalesce(
    edge_index: paddle.Tensor,
    edge_attr: Union[OptTensor, List[paddle.Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = "sum",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[
        paddle.Tensor,
        Tuple[paddle.Tensor, OptTensor],
        Tuple[paddle.Tensor, List[paddle.Tensor]],
]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (paddle.Tensor): The edge indices.
        edge_attr (paddle.Tensor or List[paddle.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    Returns:
        Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
        The sorted edge index and edge attributes, with duplicates removed.
    """
    num_edges = edge_index[0].shape[0]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if num_nodes * num_nodes > paddle_geometric.typing.MAX_INT64:
        raise ValueError("'coalesce' will result in an overflow")

    idx = paddle.empty(shape=[
        num_edges + 1,
    ], dtype=edge_index[0].dtype)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].multiply_(y=paddle.to_tensor(num_nodes)).add_(
        y=paddle.to_tensor(edge_index[int(sort_by_row)]))
    is_undirected = False
    if isinstance(edge_index, EdgeIndex):
        is_undirected = edge_index.is_undirected
    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        if isinstance(edge_index, paddle.Tensor):
            edge_index = edge_index[:, perm]
        elif isinstance(edge_index, tuple):
            edge_index = edge_index[0][perm], edge_index[1][perm]
        else:
            raise NotImplementedError
        if isinstance(edge_attr, paddle.Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    if isinstance(edge_index, EdgeIndex):
        edge_index._sort_order = SortOrder("row" if sort_by_row else "col")
        edge_index._is_undirected = is_undirected

    mask = idx[1:] > idx[:-1]

    if mask.all():
        if edge_attr is None or isinstance(edge_attr, Tensor):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index

    if isinstance(edge_index, Tensor):
        edge_index = edge_index[:, mask]
        if isinstance(edge_index, EdgeIndex):
            edge_index._is_undirected = is_undirected
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size: Optional[int] = None
    if isinstance(edge_attr, (Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.shape[1]
        idx = paddle.arange(0, num_edges)
        _x_dtype_ = mask.dtype
        idx.subtract_(
            y=paddle.to_tensor(mask.logical_not_().cast_(_x_dtype_).cumsum(
                axis=0)))

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index
