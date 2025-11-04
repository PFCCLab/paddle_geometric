from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.utils import coalesce


def grid(
    height: int,
    width: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the edge indices of a two-dimensional grid graph with height
    :attr:`height` and width :attr:`width` and its node positions.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        dtype (paddle.dtype, optional): The desired data type of the returned
            position tensor. (default: :obj:`None`)
        device (str, optional): The desired device of the returned
            tensors. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """
    edge_index = grid_index(height, width, device)
    pos = grid_pos(height, width, dtype, device)
    return edge_index, pos


def grid_index(
    height: int,
    width: int,
    device: Optional[str] = None,
) -> Tensor:

    w = width
    kernel = paddle.to_tensor([-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1],
                              place=device)
    row = paddle.arange(dtype="int64", end=height * width)
    row = row.view(-1, 1).tile(repeat_times=[1, kernel.shape[0]])
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = paddle.arange(start=3, end=row.shape[1] - 3, dtype="int64")
    row, col = row[:, index].view(-1), col[:, index].view(-1)
    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]
    edge_index = paddle.stack(x=[row, col], axis=0)
    edge_index = coalesce(edge_index, num_nodes=height * width)
    return edge_index


def grid_pos(
    height: int,
    width: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:

    dtype = "float32" if dtype is None else dtype
    x = paddle.arange(dtype=dtype, end=width, device=device)
    y = height - 1 - paddle.arange(dtype=dtype, end=height, device=device)
    x = x.tile(repeat_times=height)
    y = y.unsqueeze(axis=-1).tile(repeat_times=[1, width]).view(-1)
    return paddle.stack(x=[x, y], axis=-1)
