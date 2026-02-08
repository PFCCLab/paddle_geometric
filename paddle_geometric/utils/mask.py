from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.paddle_utils import dim2perm  # noqa
from paddle_geometric.typing import TensorFrame


def mask_select(src: Tensor, dim: int, mask: Tensor) -> Tensor:
    """Returns a new tensor which masks the src tensor along the
    dimension dim according to the boolean mask mask.
    """
    assert mask.ndim == 1

    if isinstance(src, TensorFrame):
        assert dim == 0 and src.shape[0] == mask.numel()
        return src[mask]

    assert src.shape[dim] == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.ndim

    src = src.transpose(perm=dim2perm(src.ndim, 0, dim)) if dim != 0 else src
    src = src.contiguous()
    out = src[mask]
    out = out.transpose(perm=dim2perm(out.ndim, 0, dim)) if dim != 0 else out
    out = out.contiguous()
    return out


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    """Converts indices to a mask representation."""
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = paddle.zeros(shape=size, dtype="bool", device=index.place)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    """Converts a mask to an index representation."""
    return mask.nonzero(as_tuple=False).view(-1)
