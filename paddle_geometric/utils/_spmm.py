import warnings

import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import is_paddle_sparse_tensor, scatter


def _dense_from_sparse(src: Adj) -> Tensor:
    if hasattr(src, 'to_dense'):
        return src.to_dense()
    if isinstance(src, SparseTensor):
        csr = src.to_paddle_sparse_csr_tensor()
        return csr.to_dense()
    raise NotImplementedError("Dense fallback is not available for this type")


def _spmm_dense_fallback(src: Adj, other: Tensor, reduce: str) -> Tensor:
    dense = _dense_from_sparse(src).astype(other.dtype)
    out = paddle.matmul(dense, other)
    if reduce == "mean":
        deg = dense.sum(axis=1, keepdim=True)
        out = out / deg.clip(min=1)
    return out


def spmm(
    src: Adj,
    other: Tensor,
    reduce: str = 'sum',
) -> Tensor:
    r"""Matrix product of sparse matrix with dense matrix.

    Args:
        src (paddle.Tensor or paddle_sparse.SparseTensor or EdgeIndex):
            The input sparse matrix which can be a
            :pyg:`paddle_geometric` :class:`paddle_sparse.SparseTensor`,
            a :paddle:`Paddle` :class:`paddle.sparse.Tensor` or
            a :pyg:`paddle_geometric` :class:`EdgeIndex`.
        other (paddle.Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = "sum" if reduce == "add" else reduce

    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")

    if isinstance(src, EdgeIndex):
        return src.matmul(other=other, reduce=reduce)

    if isinstance(src, SparseTensor):
        if src.nnz() == 0:
            return paddle.zeros([src.shape[0], other.shape[1]], dtype=other.dtype)

        if (other.ndim == 2 and not src.is_cuda()
                and not src.requires_grad):
            try:
                csr = src.to_paddle_sparse_csr_tensor().to(other.dtype)
                return paddle.sparse.matmul(csr, other)
            except Exception:
                return _spmm_dense_fallback(src, other, reduce)

        try:
            csr = src.to_paddle_sparse_csr_tensor().to(other.dtype)
            return paddle.sparse.matmul(csr, other)
        except Exception:
            return _spmm_dense_fallback(src, other, reduce)

    if not is_paddle_sparse_tensor(src):
        raise ValueError("'src' must be a 'paddle_sparse.SparseTensor' or a "
                         "'paddle.sparse.Tensor'")

    if src.place.is_gpu_place() and (reduce == 'min' or reduce == 'max'):
        raise NotImplementedError(f"`{reduce}` reduction is not yet "
                                  f"supported for 'paddle.sparse.Tensor' "
                                  f"on device '{src.place}'")

    if src.is_sparse_coo():
        warnings.warn(f"Converting sparse tensor to CSR format for more "
                      f"efficient processing. Consider converting your "
                      f"sparse tensor to CSR format beforehand to avoid "
                      f"repeated conversion (got '{src.layout}')")
        src = src.to_sparse_csr()

    if hasattr(src, 'is_sparse_csc') and src.is_sparse_csc():
        warnings.warn(f"Converting sparse tensor to CSR format for more "
                      f"efficient processing. Consider converting your "
                      f"sparse tensor to CSR format beforehand to avoid "
                      f"repeated conversion (got '{src.layout}')")
        src = src.to_sparse_csr()

    if reduce == 'sum':
        try:
            return paddle.sparse.matmul(src, other)
        except Exception:
            if src.place.is_cpu_place():
                return _spmm_dense_fallback(src, other, reduce)
            raise

    if src.is_sparse_csr() and not src.place.is_gpu_place():
        try:
            return paddle.sparse.matmul(src, other)
        except Exception:
            return _spmm_dense_fallback(src, other, reduce)

    if reduce == 'mean':
        if src.is_sparse_csr():
            ptr = src.crows()
            deg = ptr[1:] - ptr[:-1]
        else:
            raise NotImplementedError()

        try:
            out = paddle.sparse.matmul(src, other.astype(src.dtype))
            return out / deg.reshape([-1, 1]).astype(src.dtype).clip(min=1)
        except Exception:
            return _spmm_dense_fallback(src, other.astype(src.dtype), reduce)

    try:
        return paddle.sparse.matmul(src, other)
    except Exception:
        return _spmm_dense_fallback(src, other, reduce)
