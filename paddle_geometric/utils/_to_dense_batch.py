from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.experimental import (
    disable_dynamic_shapes,
    is_experimental_mode_enabled,
)
from paddle_geometric.utils import cumsum, scatter


@disable_dynamic_shapes(required_args=['batch_size', 'max_num_nodes'])
def to_dense_batch(
    x: Tensor,
    batch: Optional[Tensor] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features, this function creates a dense
    node feature tensor and returns a mask indicating the presence of fake nodes.

    Args:
        x (Tensor): Node feature matrix.
        batch (Tensor, optional): Batch vector which assigns each node to a specific example.
        fill_value (float, optional): The value for invalid entries in the resulting dense tensor.
        max_num_nodes (int, optional): The size of the output node dimension.
        batch_size (int, optional): The batch size.

    :rtype: (Tensor, Tensor)

    """
    if batch is None and max_num_nodes is None:
        mask = paddle.ones(shape=[1, x.shape[0]], dtype="bool")
        return x.unsqueeze(axis=0), mask
    if batch is None:
        batch = paddle.zeros(shape=x.shape[0], dtype="int64")
    if batch_size is None:
        batch_size = int(batch._max()) + 1
    num_nodes = scatter(
        paddle.ones(shape=x.shape[0], dtype=batch.dtype),
        batch,
        dim=0,
        dim_size=batch_size,
        reduce="sum",
    )
    cum_nodes = cumsum(num_nodes)
    filter_nodes = False
    dynamic_shapes_disabled = is_experimental_mode_enabled(
        "disable_dynamic_shapes")
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes._max())
    elif not dynamic_shapes_disabled and num_nodes._max() > max_num_nodes:
        filter_nodes = True
    tmp = paddle.arange(end=batch.shape[0]) - cum_nodes[batch]
    idx = tmp + batch * max_num_nodes
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]
    size = [batch_size * max_num_nodes] + list(tuple(x.shape))[1:]
    out = paddle.as_tensor(fill_value, device=x.place)
    out = out.to(x.dtype).tile(repeat_times=size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(tuple(x.shape))[1:])
    mask = paddle.zeros(shape=batch_size * max_num_nodes, dtype="bool")
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    return out, mask
