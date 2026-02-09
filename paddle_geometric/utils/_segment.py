from enum import Enum
from typing import Optional, Union

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.paddle_utils import *  # noqa
from paddle_geometric.typing import paddle_scatter


def segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://paddle-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of the :obj:`paddle_scatter` package for more
    information.

    Args:
        src (paddle.Tensor): The source tensor.
        ptr (paddle.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not paddle_geometric.typing.WITH_PADDLE_SCATTER or is_compiling():
        return _paddle_segment(src, ptr, reduce)

    if (ptr.dim() == 1 and src.place.is_gpu_place() and reduce == 'mean'):
        return _paddle_segment(src, ptr, reduce)

    return paddle_scatter.segment_csr(src, ptr, reduce=reduce)


def _paddle_segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:

    if ptr.dim() > 1:
        raise ImportError("'segment' in an arbitrary dimension "
                          "requires the 'paddle-scatter' package")

    if reduce == 'min' or reduce == 'max':
        reduce = f'a{reduce}'  # `amin` or `amax`
    initial = 0 if reduce == 'mean' else None
    out = segment_reduce(src, reduce, offsets=ptr, initial=initial)
    if reduce == 'amin' or reduce == 'amax':
        out = paddle.where(out.isinf(), paddle.zeros_like(out), out)
    return out


class ReductionType(Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PROD = "prod"


def get_reduction_enum(reduce: str) -> ReductionType:
    reduce = reduce.lower()
    if reduce == "sum":
        return ReductionType.SUM
    elif reduce == "mean":
        return ReductionType.MEAN
    elif reduce == "max" or reduce == "amax":
        return ReductionType.MAX
    elif reduce == "min" or reduce == "amin":
        return ReductionType.MIN
    elif reduce == "prod":
        return ReductionType.PROD
    else:
        raise ValueError(f"Unsupported reduction type: {reduce}")


def segment_reduce(
    data: paddle.Tensor,
    reduce: str,
    lengths: Optional[paddle.Tensor] = None,
    offsets: Optional[paddle.Tensor] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial: Optional[Union[int, float]] = None,
) -> paddle.Tensor:
    """Segment reduce operation for PyTorch.

    Args:
        data: Input tensor of shape (..., N, ...)
        reduce: Reduction type: 'sum', 'mean', 'max', 'min', 'prod'
        lengths: Optional tensor of segment lengths
        offsets: Optional tensor of segment offsets
        axis: Axis along which to perform reduction
        unsafe: If True, skip safety checks
        initial: Initial value for reduction

    Returns:
        Reduced tensor of shape (..., M, ...) where M is number of segments
    """
    if lengths is None and offsets is None:
        raise ValueError("Either lengths or offsets must be provided")
    if lengths is not None and offsets is not None:
        raise ValueError("Only one of lengths or offsets should be provided")
    reduction = get_reduction_enum(reduce)
    data = data.contiguous()
    if lengths is not None:
        lengths = lengths.contiguous()
        return _segment_reduce_lengths(data, reduction, lengths, axis, initial,
                                       unsafe)
    else:
        offsets = offsets.contiguous()
        return _segment_reduce_offsets(data, reduction, offsets, axis, initial,
                                       unsafe)


def _segment_reduce_lengths(
    data: paddle.Tensor,
    reduction: ReductionType,
    lengths: paddle.Tensor,
    axis: int,
    initial: Optional[Union[int, float]],
    unsafe: bool,
) -> paddle.Tensor:
    """Segment reduction using lengths tensor."""
    if not unsafe:
        min_length = lengths._min().item()
        if min_length < 0:
            raise ValueError("Lengths contains negative values")
        total_length = lengths.sum(axis=-1)
        expected_length = data.shape[axis]
        if not paddle.all(x=total_length == expected_length):
            raise ValueError(
                f"Sum of lengths ({total_length.item()}) does not "
                f"match data size along axis ({expected_length})")
    output_shape = list(tuple(data.shape))
    output_shape[axis] = lengths.shape[-1]
    output = paddle.empty(shape=output_shape, dtype=data.dtype)
    outer_dims = tuple(data.shape)[:axis]
    inner_dims = tuple(data.shape)[axis + 1:]
    outer_offset = (paddle.prod(
        x=paddle.to_tensor(data=outer_dims, place=data.place))
                    if outer_dims else 1)
    inner_offset = (paddle.prod(
        x=paddle.to_tensor(data=inner_dims, place=data.place))
                    if inner_dims else 1)
    data_flat = data.view(-1, tuple(data.shape)[axis], inner_offset)
    output_flat = output.view(-1, tuple(output.shape)[axis], inner_offset)
    lengths_flat = lengths.view(-1, tuple(lengths.shape)[-1])
    for outer_idx in range(outer_offset):
        lengths_vec = lengths_flat[outer_idx % lengths_flat.shape[0]]
        data_slice = data_flat[outer_idx]
        output_slice = output_flat[outer_idx]
        start_idx = 0
        for seg_idx, seg_length in enumerate(lengths_vec):
            if seg_length == 0:
                if reduction == ReductionType.MEAN and initial is None:
                    output_slice[seg_idx].fill_(value=paddle.nan)
                elif initial is not None:
                    output_slice[seg_idx].fill_(value=initial)
                elif reduction == ReductionType.MAX:
                    output_slice[seg_idx].fill_(value=-paddle.inf)
                elif reduction in [ReductionType.SUM, ReductionType.MEAN]:
                    output_slice[seg_idx].fill_(value=0)
                elif reduction == ReductionType.MIN:
                    output_slice[seg_idx].fill_(value=paddle.inf)
                elif reduction == ReductionType.PROD:
                    output_slice[seg_idx].fill_(value=1)
                continue
            seg_data = data_slice[start_idx:start_idx + seg_length]
            if reduction == ReductionType.SUM:
                result = seg_data.sum(axis=0)
            elif reduction == ReductionType.MEAN:
                result = seg_data.mean(axis=0)
            elif reduction == ReductionType.MAX:
                result = (seg_data.max(axis=0), seg_data.argmax(axis=0))[0]
            elif reduction == ReductionType.MIN:
                result = (seg_data.min(axis=0), seg_data.argmin(axis=0))[0]
            elif reduction == ReductionType.PROD:
                result = seg_data.prod(axis=0)
            if initial is not None and seg_length == 0:
                result = paddle.full_like(x=result, fill_value=initial)
            output_slice[seg_idx] = result
            start_idx += seg_length
    return output.view(output_shape)


def _segment_reduce_offsets(
    data: paddle.Tensor,
    reduction: ReductionType,
    offsets: paddle.Tensor,
    axis: int,
    initial: Optional[Union[int, float]],
    unsafe: bool,
) -> paddle.Tensor:
    """Segment reduction using offsets tensor."""
    output_shape = list(tuple(data.shape))
    output_shape[axis] = offsets.shape[-1] - 1
    output = paddle.empty(shape=output_shape, dtype=data.dtype)
    outer_dims = tuple(data.shape)[:axis]
    inner_dims = tuple(data.shape)[axis + 1:]
    outer_offset = (paddle.prod(
        x=paddle.to_tensor(data=outer_dims, place=data.place))
                    if outer_dims else 1)
    inner_offset = (paddle.prod(
        x=paddle.to_tensor(data=inner_dims, place=data.place))
                    if inner_dims else 1)
    data_flat = data.view(-1, tuple(data.shape)[axis], inner_offset)
    output_flat = output.view(-1, tuple(output.shape)[axis], inner_offset)
    offsets_flat = offsets.view(-1, tuple(offsets.shape)[-1])
    for outer_idx in range(outer_offset):
        offsets_vec = offsets_flat[outer_idx % offsets_flat.shape[0]]
        data_slice = data_flat[outer_idx]
        output_slice = output_flat[outer_idx]
        for seg_idx in range(len(offsets_vec) - 1):
            start_idx = offsets_vec[seg_idx]
            end_idx = offsets_vec[seg_idx + 1]
            seg_length = end_idx - start_idx
            if seg_length == 0:
                if reduction == ReductionType.MEAN and initial is None:
                    output_slice[seg_idx].fill_(value=paddle.nan)
                elif initial is not None:
                    output_slice[seg_idx].fill_(value=initial)
                elif reduction == ReductionType.MAX:
                    output_slice[seg_idx].fill_(value=-paddle.inf)
                elif reduction in [ReductionType.SUM, ReductionType.MEAN]:
                    output_slice[seg_idx].fill_(value=0)
                elif reduction == ReductionType.MIN:
                    output_slice[seg_idx].fill_(value=paddle.inf)
                elif reduction == ReductionType.PROD:
                    output_slice[seg_idx].fill_(value=1)
                continue
            seg_data = data_slice[start_idx:end_idx]
            if seg_data.shape[0] == 0:
                if reduction == ReductionType.MEAN and initial is None:
                    output_slice[seg_idx].fill_(value=paddle.nan)
                elif initial is not None:
                    output_slice[seg_idx].fill_(value=initial)
                elif reduction == ReductionType.MAX:
                    output_slice[seg_idx].fill_(value=-paddle.inf)
                elif reduction in [ReductionType.SUM, ReductionType.MEAN]:
                    output_slice[seg_idx].fill_(value=0)
                elif reduction == ReductionType.MIN:
                    output_slice[seg_idx].fill_(value=paddle.inf)
                elif reduction == ReductionType.PROD:
                    output_slice[seg_idx].fill_(value=1)
                continue
            if reduction == ReductionType.SUM:
                result = seg_data.sum(axis=0)
            elif reduction == ReductionType.MEAN:
                result = seg_data.mean(axis=0)
            elif reduction == ReductionType.MAX:
                result = (seg_data.max(axis=0), seg_data.argmax(axis=0))[0]
            elif reduction == ReductionType.MIN:
                result = (seg_data.min(axis=0), seg_data.argmin(axis=0))[0]
            elif reduction == ReductionType.PROD:
                result = seg_data.prod(axis=0)
            output_slice[seg_idx] = result
    return output.view(output_shape)


class SegmentReduce(paddle.autograd.PyLayer):
    """Autograd function for segment reduce with backward pass."""
    @staticmethod
    def forward(
        ctx,
        data: paddle.Tensor,
        reduce: str,
        lengths: Optional[paddle.Tensor],
        offsets: Optional[paddle.Tensor],
        axis: int,
        unsafe: bool,
        initial: Optional[Union[int, float]],
    ) -> paddle.Tensor:
        ctx.reduce = reduce
        ctx.axis = axis
        ctx.initial = initial
        ctx.save_for_backward(data, lengths, offsets)
        return segment_reduce(data, reduce, lengths, offsets, axis, unsafe,
                              initial)

    @staticmethod
    def backward(ctx, grad_output):
        reduce = ctx.reduce
        axis = ctx.axis
        initial = ctx.initial
        data, lengths, offsets = ctx.saved_tensor()
        grad_input = _segment_reduce_backward(
            grad_output.contiguous(),
            data.contiguous(),
            reduce,
            lengths,
            offsets,
            axis,
            initial,
        )
        return grad_input, None, None, None, None, None, None


def _segment_reduce_backward(
    grad_output: paddle.Tensor,
    data: paddle.Tensor,
    reduce: str,
    lengths: Optional[paddle.Tensor],
    offsets: Optional[paddle.Tensor],
    axis: int,
    initial: Optional[Union[int, float]],
) -> paddle.Tensor:
    """Backward pass for segment reduce operation."""
    reduction = get_reduction_enum(reduce)
    if lengths is not None:
        return _segment_reduce_lengths_backward(grad_output, data, reduction,
                                                lengths, axis, initial)
    else:
        return _segment_reduce_offsets_backward(grad_output, data, reduction,
                                                offsets, axis, initial)


def _segment_reduce_lengths_backward(
    grad_output: paddle.Tensor,
    data: paddle.Tensor,
    reduction: ReductionType,
    lengths: paddle.Tensor,
    axis: int,
    initial: Optional[Union[int, float]],
) -> paddle.Tensor:
    """Backward pass for lengths-based segment reduction."""
    grad_input = paddle.zeros_like(x=data)
    outer_dims = tuple(data.shape)[:axis]
    inner_dims = tuple(data.shape)[axis + 1:]
    outer_offset = (paddle.prod(
        x=paddle.to_tensor(data=outer_dims, place=data.place))
                    if outer_dims else 1)
    inner_offset = (paddle.prod(
        x=paddle.to_tensor(data=inner_dims, place=data.place))
                    if inner_dims else 1)
    data_flat = data.view(-1, tuple(data.shape)[axis], inner_offset)
    grad_input_flat = grad_input.view(-1,
                                      tuple(data.shape)[axis], inner_offset)
    grad_output_flat = grad_output.view(-1,
                                        tuple(grad_output.shape)[axis],
                                        inner_offset)
    lengths_flat = lengths.view(-1, tuple(lengths.shape)[-1])
    for outer_idx in range(outer_offset):
        lengths_vec = lengths_flat[outer_idx % lengths_flat.shape[0]]
        data_slice = data_flat[outer_idx]
        grad_input_slice = grad_input_flat[outer_idx]
        grad_output_slice = grad_output_flat[outer_idx]
        start_idx = 0
        for seg_idx, seg_length in enumerate(lengths_vec):
            if seg_length == 0:
                continue
            seg_grad = grad_output_slice[seg_idx]
            seg_data = data_slice[start_idx:start_idx + seg_length]
            if reduction == ReductionType.SUM:
                grad_input_slice[start_idx:start_idx +
                                 seg_length] += seg_grad.unsqueeze(axis=0)
            elif reduction == ReductionType.MEAN:
                grad_val = seg_grad / seg_length
                grad_input_slice[start_idx:start_idx +
                                 seg_length] += grad_val.unsqueeze(axis=0)
            elif reduction in [ReductionType.MAX, ReductionType.MIN]:
                if reduction == ReductionType.MAX:
                    max_vals, _ = seg_data.max(axis=0), seg_data.argmax(axis=0)
                else:
                    max_vals, _ = seg_data.min(axis=0), seg_data.argmin(axis=0)
                for inner_idx in range(inner_offset):
                    mask = seg_data[:, inner_idx] == max_vals[inner_idx]
                    count = mask.sum()
                    if count > 0:
                        grad_val = seg_grad[inner_idx] / count
                        positions = paddle.where(condition=mask)[0] + start_idx
                        grad_input_slice[positions, inner_idx] += grad_val
            elif reduction == ReductionType.PROD:
                for j in range(seg_length):
                    other_indices = [k for k in range(seg_length) if k != j]
                    if other_indices:
                        other_prod = seg_data[other_indices].prod(axis=0)
                    else:
                        other_prod = paddle.ones_like(x=seg_grad) * (
                            initial if initial is not None else 1.0)
                    grad_input_slice[start_idx + j] += seg_grad * other_prod
            start_idx += seg_length
    return grad_input


def _segment_reduce_offsets_backward(
    grad_output: paddle.Tensor,
    data: paddle.Tensor,
    reduction: ReductionType,
    offsets: paddle.Tensor,
    axis: int,
    initial: Optional[Union[int, float]],
) -> paddle.Tensor:
    """Backward pass for offsets-based segment reduction."""
    grad_input = paddle.zeros_like(x=data)
    outer_dims = tuple(data.shape)[:axis]
    inner_dims = tuple(data.shape)[axis + 1:]
    outer_offset = (paddle.prod(
        x=paddle.to_tensor(data=outer_dims, place=data.place))
                    if outer_dims else 1)
    inner_offset = (paddle.prod(
        x=paddle.to_tensor(data=inner_dims, place=data.place))
                    if inner_dims else 1)
    data_flat = data.view(-1, tuple(data.shape)[axis], inner_offset)
    grad_input_flat = grad_input.view(-1,
                                      tuple(data.shape)[axis], inner_offset)
    grad_output_flat = grad_output.view(-1,
                                        tuple(grad_output.shape)[axis],
                                        inner_offset)
    offsets_flat = offsets.view(-1, tuple(offsets.shape)[-1])
    for outer_idx in range(outer_offset):
        offsets_vec = offsets_flat[outer_idx % offsets_flat.shape[0]]
        data_slice = data_flat[outer_idx]
        grad_input_slice = grad_input_flat[outer_idx]
        grad_output_slice = grad_output_flat[outer_idx]
        for seg_idx in range(len(offsets_vec) - 1):
            start_idx = offsets_vec[seg_idx]
            end_idx = offsets_vec[seg_idx + 1]
            seg_length = end_idx - start_idx
            if seg_length == 0:
                continue
            seg_grad = grad_output_slice[seg_idx]
            seg_data = data_slice[start_idx:end_idx]
            if reduction == ReductionType.SUM:
                grad_input_slice[start_idx:end_idx] += seg_grad.unsqueeze(
                    axis=0)
            elif reduction == ReductionType.MEAN:
                grad_val = seg_grad / seg_length
                grad_input_slice[start_idx:end_idx] += grad_val.unsqueeze(
                    axis=0)
            elif reduction in [ReductionType.MAX, ReductionType.MIN]:
                if reduction == ReductionType.MAX:
                    max_vals, _ = seg_data.max(axis=0), seg_data.argmax(axis=0)
                else:
                    max_vals, _ = seg_data.min(axis=0), seg_data.argmin(axis=0)
                for inner_idx in range(inner_offset):
                    mask = seg_data[:, inner_idx] == max_vals[inner_idx]
                    count = mask.sum()
                    if count > 0:
                        grad_val = seg_grad[inner_idx] / count
                        positions = paddle.where(condition=mask)[0] + start_idx
                        grad_input_slice[positions, inner_idx] += grad_val
            elif reduction == ReductionType.PROD:
                for j in range(seg_length):
                    other_indices = [k for k in range(seg_length) if k != j]
                    if other_indices:
                        other_prod = seg_data[other_indices].prod(axis=0)
                    else:
                        other_prod = paddle.ones_like(x=seg_grad) * (
                            initial if initial is not None else 1.0)
                    grad_input_slice[start_idx + j] += seg_grad * other_prod
    return grad_input


def segment_reduce_autograd(
    data: paddle.Tensor,
    reduce: str,
    lengths: Optional[paddle.Tensor] = None,
    offsets: Optional[paddle.Tensor] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial: Optional[Union[int, float]] = None,
) -> paddle.Tensor:
    """Segment reduce with autograd support."""
    return SegmentReduce.apply(data, reduce, lengths, offsets, axis, unsafe,
                               initial)
