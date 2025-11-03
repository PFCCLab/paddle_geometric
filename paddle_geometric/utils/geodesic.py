import multiprocessing as mp
import warnings
from typing import Optional

import numpy as np
import paddle


def geodesic_distance(pos: paddle.Tensor, face: paddle.Tensor,
                      src: Optional[paddle.Tensor] = None,
                      dst: Optional[paddle.Tensor] = None, norm: bool = True,
                      max_distance: Optional[float] = None,
                      num_workers: int = 0,
                      **kwargs: Optional[paddle.Tensor]) -> paddle.Tensor:
    r"""Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
    and :obj:`face`. If :obj:`src` and :obj:`dst` are given, this method only
    computes the geodesic distances for the respective source and target
    node-pairs.

    .. note::

        This function requires the :obj:`gdist` package.
        To install, run :obj:`pip install cython && pip install gdist`.

    Args:
        pos (torch.Tensor): The node positions.
        face (torch.Tensor): The face indices.
        src (torch.Tensor, optional): If given, only compute geodesic distances
            for the specified source indices. (default: :obj:`None`)
        dst (torch.Tensor, optional): If given, only compute geodesic distances
            for the specified target indices. (default: :obj:`None`)
        norm (bool, optional): Normalizes geodesic distances by
            :math:`\\sqrt{\\textrm{area}(\\mathcal{M})}`. (default: :obj:`True`)
        max_distance (float, optional): If given, only yields results for
            geodesic distances less than :obj:`max_distance`. This will speed
            up runtime dramatically. (default: :obj:`None`)
        num_workers (int, optional): How many subprocesses to use for
            calculating geodesic distances.
            :obj:`0` means that computation takes place in the main process.
            :obj:`-1` means that the available amount of CPU cores is used.
            (default: :obj:`0`)
        kwargs: desc

    :rtype: :class:`Tensor`

    Example:
        >>> pos = torch.tensor([[0.0, 0.0, 0.0],
        ...                     [2.0, 0.0, 0.0],
        ...                     [0.0, 2.0, 0.0],
        ...                     [2.0, 2.0, 0.0]])
        >>> face = torch.tensor([[0, 0],
        ...                      [1, 2],
        ...                      [3, 3]])
        >>> geodesic_distance(pos, face)
        [[0, 1, 1, 1.4142135623730951],
        [1, 0, 1.4142135623730951, 1],
        [1, 1.4142135623730951, 0, 1],
        [1.4142135623730951, 1, 1, 0]]
    """
    import gdist

    if "dest" in kwargs:
        dst = kwargs["dest"]
        warnings.warn(
            "'dest' attribute in 'geodesic_distance' is deprecated and will be"
            " removed in a future release. Use the 'dst' argument instead.")
    max_distance = float("inf") if max_distance is None else max_distance
    if norm:
        area = (pos[face[1]] - pos[face[0]]).cross(
            y=pos[face[2]] - pos[face[0]], axis=1)
        scale = float((area.norm(p=2, dim=1) / 2).sum().sqrt())
    else:
        scale = 1.0
    dtype = pos.dtype
    pos_np = pos.detach().cpu().to("float64").numpy()
    face_np = face.detach().t().cpu().to("int32").numpy()
    if src is None and dst is None:
        out = (gdist.local_gdist_matrix(
            pos_np, face_np, max_distance * scale).toarray() / scale)
        return paddle.to_tensor(data=out).to(dtype)
    if src is None:
        src_np = paddle.arange(dtype="int32", end=pos.shape[0]).numpy()
    else:
        src_np = src.detach().cpu().to("int32").numpy()
    dst_np = None if dst is None else dst.detach().cpu().to("int32").numpy()

    def _parallel_loop(
        pos_np: np.ndarray,
        face_np: np.ndarray,
        src_np: np.ndarray,
        dst_np: Optional[np.ndarray],
        max_distance: float,
        scale: float,
        i: int,
        dtype: paddle.dtype,
    ) -> paddle.Tensor:
        s = src_np[i:i + 1]
        d = None if dst_np is None else dst_np[i:i + 1]
        out = gdist.compute_gdist(pos_np, face_np, s, d, max_distance * scale)
        out = out / scale
        return paddle.to_tensor(data=out).to(dtype)

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            data = [(pos_np, face_np, src_np, dst_np, max_distance, scale, i,
                     dtype) for i in range(len(src_np))]
            outs = pool.starmap(_parallel_loop, data)
    else:
        outs = [
            _parallel_loop(pos_np, face_np, src_np, dst_np, max_distance,
                           scale, i, dtype) for i in range(len(src_np))
        ]
    out = paddle.concat(x=outs, axis=0)
    if dst is None:
        out = out.view(-1, pos.shape[0])
    return out
