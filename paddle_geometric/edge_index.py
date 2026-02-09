import functools
import warnings
from enum import Enum
import numpy as np
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import Index
from paddle_geometric.index import index2ptr, ptr2index
from paddle_geometric.typing import INDEX_DTYPES, SparseTensor

from .dispatcher import BaseTensorSubclass, register_for

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}

ReduceType = Literal['sum', 'mean', 'amin', 'amax', 'add', 'min', 'max']
PYG_REDUCE: Dict[ReduceType, ReduceType] = {
    'add': 'sum',
    'amin': 'min',
    'amax': 'max'
}
TORCH_REDUCE: Dict[ReduceType, ReduceType] = {
    'add': 'sum',
    'min': 'amin',
    'max': 'amax'
}


class SortOrder(Enum):
    ROW = 'row'
    COL = 'col'


class CatMetadata(NamedTuple):
    nnz: List[int]
    sparse_size: List[Tuple[Optional[int], Optional[int]]]
    sort_order: List[Optional[SortOrder]]
    is_undirected: List[bool]


def set_tuple_item(
    values: Tuple[Any, ...],
    dim: int,
    value: Any,
) -> Tuple[Any, ...]:
    if dim < -len(values) or dim >= len(values):
        raise IndexError("tuple index out of range")

    dim = dim + len(values) if dim < 0 else dim
    return values[:dim] + (value, ) + values[dim + 1:]


def maybe_add(
    value: Sequence[Optional[int]],
    other: Union[int, Sequence[Optional[int]]],
    alpha: int = 1,
) -> Tuple[Optional[int], ...]:

    if isinstance(other, int):
        return tuple(v + alpha * other if v is not None else None
                     for v in value)

    assert len(value) == len(other)
    return tuple(v + alpha * o if v is not None and o is not None else None
                 for v, o in zip(value, other))


def maybe_sub(
    value: Sequence[Optional[int]],
    other: Union[int, Sequence[Optional[int]]],
    alpha: int = 1,
) -> Tuple[Optional[int], ...]:

    if isinstance(other, int):
        return tuple(v - alpha * other if v is not None else None
                     for v in value)

    assert len(value) == len(other)
    return tuple(v - alpha * o if v is not None and o is not None else None
                 for v, o in zip(value, other))


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in INDEX_DTYPES:
        raise ValueError(f"'EdgeIndex' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{INDEX_DTYPES})")


def assert_two_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"'EdgeIndex' needs to be two-dimensional "
                         f"(got {tensor.dim()} dimensions)")
    if paddle.in_dynamic_mode() and tensor.shape[0] != 2:
        raise ValueError(f"'EdgeIndex' needs to have a shape of "
                         f"[2, *] (got {list(tensor.shape)})")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor[0].is_contiguous() or not tensor[1].is_contiguous():
        raise ValueError("'EdgeIndex' needs to be contiguous. Please call "
                         "`edge_index.contiguous()` before proceeding.")


def assert_symmetric(size: Tuple[Optional[int], Optional[int]]) -> None:
    if (paddle.in_dynamic_mode() and size[0] is not None
            and size[1] is not None and size[0] != size[1]):
        raise ValueError(f"'EdgeIndex' is undirected but received a "
                         f"non-symmetric size (got {list(size)})")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'EdgeIndex', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort_by(...)` first.")
        return func(self, *args, **kwargs)

    return wrapper


class EdgeIndex(BaseTensorSubclass):
    r"""A COO :obj:`edge_index` tensor with additional (meta)data attached.

    :class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
    an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
    Edges are given as pairwise source and destination node indices in sparse
    COO format.

    While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
    :class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

    * :obj:`sparse_size`: The underlying sparse matrix size
    * :obj:`sort_order`: The sort order (if present), either by row or column.
    * :obj:`is_undirected`: Whether edges are bidirectional.

    Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
    in case its representation is sorted, such as its :obj:`rowptr` or
    :obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
    versa.
    Caches are filled based on demand (*e.g.*, when calling
    :meth:`EdgeIndex.sort_by`), or when explicitly requested via
    :meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
    lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

    This representation ensures optimal computation in GNN message passing
    schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
    workflows.

    .. code-block:: python

        from torch_geometric import EdgeIndex

        edge_index = EdgeIndex(
            [[0, 1, 1, 2],
             [1, 0, 2, 1]]
            sparse_size=(3, 3),
            sort_order='row',
            is_undirected=True,
            device='cpu',
        )
        >>> EdgeIndex([[0, 1, 1, 2],
        ...            [1, 0, 2, 1]])
        assert edge_index.is_sorted_by_row
        assert edge_index.is_undirected

        # Flipping order:
        edge_index = edge_index.flip(0)
        >>> EdgeIndex([[1, 0, 2, 1],
        ...            [0, 1, 1, 2]])
        assert edge_index.is_sorted_by_col
        assert edge_index.is_undirected

        # Filtering:
        mask = torch.tensor([True, True, True, False])
        edge_index = edge_index[:, mask]
        >>> EdgeIndex([[1, 0, 2],
        ...            [0, 1, 1]])
        assert edge_index.is_sorted_by_col
        assert not edge_index.is_undirected

        # Sparse-Dense Matrix Multiplication:
        out = edge_index.flip(0) @ torch.randn(3, 16)
        assert out.size() == (3, 16)
    """
    # See "https://pytorch.org/docs/stable/notes/extending.html"
    # for a basic tutorial on how to subclass `torch.Tensor`.

    # The underlying tensor representation:
    _data: Tensor

    # The size of the underlying sparse matrix:
    _sparse_size: Tuple[Optional[int], Optional[int]] = (None, None)

    # Whether the `edge_index` representation is non-sorted (`None`), or sorted
    # based on row or column values.
    _sort_order: Optional[SortOrder] = None

    # Whether the `edge_index` is undirected:
    # NOTE `is_undirected` allows us to assume symmetric adjacency matrix size
    # and to share compressed pointer representations, however, it does not
    # allow us get rid of CSR/CSC permutation vectors since ordering within
    # neighborhoods is not necessarily deterministic.
    _is_undirected: bool = False

    # A cache for its compressed representation:
    _indptr: Optional[Tensor] = None

    # A cache for its transposed representation:
    _T_perm: Optional[Tensor] = None
    _T_index: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)
    _T_indptr: Optional[Tensor] = None

    # A cached "1"-value vector for `torch.sparse` matrix multiplication:
    _value: Optional[Tensor] = None

    # Whenever we perform a concatenation of edge indices, we cache the
    # original metadata to be able to reconstruct individual edge indices:
    _cat_metadata: Optional[CatMetadata] = None

    def __init__(
        self: Type,
        data: Any,
        *args: Any,
        sparse_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
        sort_order: Optional[Union[str, SortOrder]] = None,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> 'EdgeIndex':
        if 'device' in kwargs:
            kwargs['place'] = kwargs['device']
            del kwargs['device']

        if not isinstance(data, Tensor):
            data = paddle.to_tensor(data, *args, **kwargs)
        elif len(args) > 0:
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"(Tensor, {', '.join(str(type(arg)) for arg in args)})")
        elif len(kwargs) > 0:
            raise TypeError(f"new() received invalid keyword arguments - got "
                            f"{set(kwargs.keys())})")

        assert isinstance(data, Tensor)

        indptr: Optional[Tensor] = None

        if isinstance(data,
                      EdgeIndex):  # If passed `EdgeIndex`, inherit metadata:
            indptr = data._indptr
            sparse_size = sparse_size or data.sparse_size()
            sort_order = sort_order or data.sort_order
            is_undirected = is_undirected or data.is_undirected

        # Convert `paddle.sparse` tensors to `EdgeIndex` representation:
        if data.is_sparse_coo():
            sort_order = SortOrder.ROW
            sparse_size = sparse_size or (data.shape[0], data.shape[1]
                                          )  # .shape in Paddle
            data = data.indices()

        if data.is_sparse_csr():
            indptr = data.crows()  # CSR row pointers
            col = data.cols()  # CSR column indices

            assert isinstance(indptr, Tensor)
            row = ptr2index(indptr, output_size=col.numel())

            sort_order = SortOrder.ROW
            sparse_size = sparse_size or (data.shape[0], data.shape[0])
            if sparse_size[0] is not None and sparse_size[0] != data.shape[0]:
                indptr = None
            data = paddle.stack([row, col])  # Equivalent to torch.stack

        # Paddle doesn't support the CSC format.
        assert_valid_dtype(data)
        assert_two_dimensional(data)
        assert_contiguous(data)

        if sparse_size is None:
            sparse_size = (None, None)

        if is_undirected:
            assert_symmetric(sparse_size)
            if sparse_size[0] is not None and sparse_size[1] is None:
                sparse_size = (sparse_size[0], sparse_size[0])
            elif sparse_size[0] is None and sparse_size[1] is not None:
                sparse_size = (sparse_size[1], sparse_size[1])

        self.data = data.data

        # Attach metadata:
        self._sparse_size = sparse_size
        self._sort_order = None if sort_order is None else SortOrder(
            sort_order)
        self._is_undirected = is_undirected
        self._indptr = indptr

        if isinstance(data,
                      EdgeIndex):  # If passed `EdgeIndex`, inherit metadata:
            self.data = data._data
            self._T_perm = data._T_perm
            self._T_index = data._T_index
            self._T_indptr = data._T_indptr
            self._value = data._value

            # Reset metadata if cache is invalidated:
            num_rows = sparse_size[0]
            if num_rows is not None and num_rows != data.sparse_size(0):
                self._indptr = None

            num_cols = sparse_size[1]
            if num_cols is not None and num_cols != data.sparse_size(1):
                self._T_indptr = None

    @property
    def _data(self) -> Tensor:
        return self.data

    # Validation ##############################################################

    def validate(self) -> 'EdgeIndex':
        r"""Validates the :class:`EdgeIndex` representation.

        In particular, it ensures that

        * it only holds valid indices.
        * the sort order is correctly set.
        * indices are bidirectional in case it is specified as undirected.
        """
        assert_valid_dtype(self._data)
        assert_two_dimensional(self._data)
        assert_contiguous(self._data)
        if self.is_undirected:
            assert_symmetric(self.sparse_size())

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self.num_rows is not None
                and self._data[0].max() >= self.num_rows):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of rows "
                             f"(got {int(self._data[0].max())}, but expected "
                             f"values smaller than {self.num_rows})")

        if (self.numel() > 0 and self.num_cols is not None
                and self._data[1].max() >= self.num_cols):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its number of columns "
                             f"(got {int(self._data[1].max())}, but expected "
                             f"values smaller than {self.num_cols})")

        if self.is_sorted_by_row and (self._data[0].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"row indices")

        if self.is_sorted_by_col and (self._data[1].diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"column indices")

        if self.is_undirected:
            flat_index1 = self._data[0] * self.get_num_rows() + self._data[1]
            flat_index1 = flat_index1.sort()
            flat_index2 = self._data[1] * self.get_num_cols() + self._data[0]
            flat_index2 = flat_index2.sort()
            if not paddle.equal_all(flat_index1, flat_index2):
                raise ValueError(f"'{self.__class__.__name__}' is not "
                                 f"undirected")

        return self

    # Properties ##############################################################

    @overload
    def sparse_size(self) -> Tuple[Optional[int], Optional[int]]:
        pass

    @overload
    def sparse_size(self, dim: int) -> Optional[int]:
        pass

    def sparse_size(
        self,
        dim: Optional[int] = None,
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""The size of the underlying sparse matrix.
        If :obj:`dim` is specified, returns an integer holding the size of that
        sparse dimension.

        Args:
            dim (int, optional): The dimension for which to retrieve the size.
                (default: :obj:`None`)
        """
        if dim is not None:
            return self._sparse_size[dim]
        return self._sparse_size

    @property
    def num_rows(self) -> Optional[int]:
        r"""The number of rows of the underlying sparse matrix."""
        return self._sparse_size[0]

    @property
    def num_cols(self) -> Optional[int]:
        r"""The number of columns of the underlying sparse matrix."""
        return self._sparse_size[1]

    @property
    def sort_order(self) -> Optional[str]:
        r"""The sort order of indices, either :obj:`"row"`, :obj:`"col"` or
        :obj:`None`.
        """
        return None if self._sort_order is None else self._sort_order.value

    @property
    def is_sorted(self) -> bool:
        r"""Returns whether indices are either sorted by rows or columns."""
        return self._sort_order is not None

    @property
    def is_sorted_by_row(self) -> bool:
        r"""Returns whether indices are sorted by rows."""
        return self._sort_order == SortOrder.ROW

    @property
    def is_sorted_by_col(self) -> bool:
        r"""Returns whether indices are sorted by columns."""
        return self._sort_order == SortOrder.COL

    @property
    def is_undirected(self) -> bool:
        r"""Returns whether indices are bidirectional."""
        return self._is_undirected

    @property
    def dtype(self) -> 'paddle.dtype':  # Return type as paddle.dtype
        # TODO Remove once Paddle does not override `dtype` in DataLoader.
        return self._data.dtype  # Accessing dtype directly from the tensor

    # Cache Interface #########################################################

    def get_sparse_size(
        self,
        dim: Optional[int] = None,
    ) -> Union[paddle.shape, int]:
        r"""The size of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        If :obj:`dim` is specified, returns an integer holding the size of that
        sparse dimension.

        Args:
            dim (int, optional): The dimension for which to retrieve the size.
                (default: :obj:`None`)
        """
        if dim is not None:
            size = self._sparse_size[dim]
            if size is not None:
                return size

            if self.is_undirected:
                size = int(self._data.max()) + 1 if self.numel() > 0 else 0
                self._sparse_size = (size, size)
                return size

            size = int(self._data[dim].max()) + 1 if self.numel() > 0 else 0
            self._sparse_size = set_tuple_item(self._sparse_size, dim, size)
            return size

        return (self.get_sparse_size(0), self.get_sparse_size(1))

    def sparse_resize_(  # type: ignore
        self,
        num_rows: Optional[int],
        num_cols: Optional[int],
    ) -> 'EdgeIndex':
        r"""Assigns or re-assigns the size of the underlying sparse matrix.

        Args:
            num_rows (int, optional): The number of rows.
            num_cols (int, optional): The number of columns.
        """
        if self.is_undirected:
            if num_rows is not None and num_cols is None:
                num_cols = num_rows
            elif num_cols is not None and num_rows is None:
                num_rows = num_cols

            if num_rows is not None and num_rows != num_cols:
                raise ValueError(f"'EdgeIndex' is undirected but received a "
                                 f"non-symmetric size "
                                 f"(got [{num_rows}, {num_cols}])")

        def _modify_ptr(
            ptr: Optional[Tensor],
            size: Optional[int],
        ) -> Optional[Tensor]:

            if ptr is None or size is None:
                return None

            if ptr.numel() - 1 >= size:
                return ptr[:size + 1]

            fill_value = ptr.new_full(
                (size - ptr.numel() + 1, ),
                fill_value=ptr[-1],  # type: ignore
            )
            return paddle.concat([ptr, fill_value], axis=0)

        if self.is_sorted_by_row:
            self._indptr = _modify_ptr(self._indptr, num_rows)
            self._T_indptr = _modify_ptr(self._T_indptr, num_cols)

        if self.is_sorted_by_col:
            self._indptr = _modify_ptr(self._indptr, num_cols)
            self._T_indptr = _modify_ptr(self._T_indptr, num_rows)

        self._sparse_size = (num_rows, num_cols)

        return self

    def get_num_rows(self) -> int:
        r"""The number of rows of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        return self.get_sparse_size(0)

    def get_num_cols(self) -> int:
        r"""The number of columns of the underlying sparse matrix.
        Automatically computed and cached when not explicitly set.
        """
        return self.get_sparse_size(1)

    @assert_sorted
    def get_indptr(self) -> Tensor:
        r"""Returns the compressed index representation in case
        :class:`EdgeIndex` is sorted.
        """
        if self._indptr is not None:
            return self._indptr

        if self.is_undirected and self._T_indptr is not None:
            return self._T_indptr

        dim = 0 if self.is_sorted_by_row else 1
        self._indptr = index2ptr(self._data[dim], self.get_sparse_size(dim))

        return self._indptr

    @assert_sorted
    def _sort_by_transpose(self) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        dim = 1 if self.is_sorted_by_row else 0

        if self._T_perm is None:
            self.get_sparse_size(dim)
            index, perm = self._data[dim].sort(), self._data[dim].argsort()
            self._T_index = set_tuple_item(self._T_index, dim, index)
            self._T_perm = perm.to(self.dtype)

        if self._T_index[1 - dim] is None:
            self._T_index = set_tuple_item(  #
                self._T_index, 1 - dim, self._data[1 - dim][self._T_perm])

        row, col = self._T_index
        assert row is not None and col is not None

        return (row, col), self._T_perm

    @assert_sorted
    def get_csr(self) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        r"""Returns the compressed CSR representation
        :obj:`(rowptr, col), perm` in case :class:`EdgeIndex` is sorted.
        """
        if self.is_sorted_by_row:
            return (self.get_indptr(), self._data[1]), None

        assert self.is_sorted_by_col
        (row, col), perm = self._sort_by_transpose()

        if self._T_indptr is not None:
            rowptr = self._T_indptr
        elif self.is_undirected and self._indptr is not None:
            rowptr = self._indptr
        else:
            rowptr = self._T_indptr = index2ptr(row, self.get_num_rows())

        return (rowptr, col), perm

    @assert_sorted
    def get_csc(self) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        r"""Returns the compressed CSC representation
        :obj:`(colptr, row), perm` in case :class:`EdgeIndex` is sorted.
        """
        if self.is_sorted_by_col:
            # Verify that the data is actually sorted by column
            indices_np = np.lexsort([self._data[0].numpy(), self._data[1].numpy()])
            # Create the sorted version
            sorted_indices = paddle.to_tensor(indices_np, dtype=self.dtype)
            sorted_data = self._data[:, sorted_indices]
            # Check if current data matches sorted data
            if paddle.equal(self._data, sorted_data).all():
                # Data is properly sorted, use cached indptr
                return (self.get_indptr(), self._data[0]), None

        # Data is not properly sorted, need to transpose
        (row, col), perm = self._sort_by_transpose()

        if self._T_indptr is not None:
            colptr = self._T_indptr
        elif self.is_undirected and self._indptr is not None:
            colptr = self._indptr
        else:
            colptr = self._T_indptr = index2ptr(col, self.get_num_cols())

        return (colptr, row), perm

    def _get_value(self,
                   dtype: Optional[paddle.dtype] = None) -> paddle.Tensor:
        if self._value is not None:
            if (dtype or paddle.get_default_dtype()) == self._value.dtype:
                return self._value

        # Expanded tensors are not yet supported in all Paddle code paths :(
        # value = paddle.ones([1], dtype=dtype, place=self.place)
        # value = value.expand(self.size(1))
        self._value = paddle.ones([self.shape[1]], dtype=dtype,
                                  device=self.place)
        return self._value

    def fill_cache_(self, no_transpose: bool = False) -> 'EdgeIndex':
        r"""Fills the cache with (meta)data information.

        Args:
            no_transpose (bool, optional): If set to :obj:`True`, will not fill
                the cache with information about the transposed
                :class:`EdgeIndex`. (default: :obj:`False`)
        """
        self.get_sparse_size()

        if self.is_sorted_by_row:
            self.get_csr()
            if not no_transpose:
                self.get_csc()
        elif self.is_sorted_by_col:
            self.get_csc()
            if not no_transpose:
                self.get_csr()

        return self

    # Methods #################################################################

    def share_memory_(self) -> 'EdgeIndex':
        """"""  # noqa: D419
        warnings.warn("share_memory_ is not supported for Index, ignore it")
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        warnings.warn("is_shared is not supported for Index, ignore it")
        return False

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`EdgeIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data

    def sort_by(
        self,
        sort_order: Union[str, SortOrder],
        stable: bool = False,
    ) -> 'SortReturnType':
        r"""Sorts the elements by row or column indices.

        Args:
            sort_order (str): The sort order, either :obj:`"row"` or
                :obj:`"col"`.
            stable (bool, optional): Makes the sorting routine stable, which
                guarantees that the order of equivalent elements is preserved.
                (default: :obj:`False`)
        """
        sort_order = SortOrder(sort_order)

        if self._sort_order == sort_order:  # Nothing to do.
            return SortReturnType(self, None)

        if self.is_sorted:
            (row, col), perm = self._sort_by_transpose()
            edge_index = paddle.stack([row, col], axis=0)

        # Otherwise, perform sorting:
        elif sort_order == SortOrder.ROW:
            row, perm = self._data[0].sort(
                stable=stable), self._data[0].argsort(stable=stable)
            edge_index = paddle.stack([row, self._data[1][perm]], axis=0)

        else:
            col, perm = self._data[1].sort(
                stable=stable), self._data[1].argsort(stable=stable)

            edge_index = paddle.stack([self._data[0][perm], col], axis=0)

        out = self.__class__(edge_index)

        # We can inherit metadata and (mostly) cache:
        out._sparse_size = self.sparse_size()
        out._sort_order = sort_order
        out._is_undirected = self.is_undirected

        out._indptr = self._indptr
        out._T_indptr = self._T_indptr

        # NOTE We cannot copy CSR<>CSC permutations since we don't require that
        # local neighborhoods are sorted, and thus they may run out of sync.

        out._value = self._value

        return SortReturnType(out, perm)

    def to_dense(  # type: ignore
        self,
        value: Optional[Tensor] = None,
        fill_value: float = 0.0,
        dtype: Optional[paddle.dtype] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a dense :class:`torch.Tensor`.

        .. warning::

            In case of duplicated edges, the behavior is non-deterministic (one
            of the values from :obj:`value` will be picked arbitrarily). For
            deterministic behavior, consider calling
            :meth:`~torch_geometric.utils.coalesce` beforehand.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            fill_value (float, optional): The fill value for remaining elements
                in the dense matrix. (default: :obj:`0.0`)
            dtype (torch.dtype, optional): The data type of the returned
                tensor. (default: :obj:`None`)
        """
        dtype = value.dtype if value is not None else dtype

        size = self.get_sparse_size()
        if value is not None and value.dim() > 1:
            size = list(size) + value.shape[1:]  # type: ignore

        out = paddle.full(size, fill_value, dtype=dtype, device=self.device)
        out[self._data[0], self._data[1]] = value if value is not None else 1

        return out

    def to_sparse_coo(self, value: Optional[Tensor] = None) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_coo_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        value = self._get_value() if value is None else value

        out = paddle.sparse.sparse_coo_tensor(
            indices=self._data,
            values=value,
            shape=self.get_sparse_size(),
            place=self.device,
            stop_gradient=value.stop_gradient,
        )
        if self.is_sorted_by_row:
            out = out.coalesce()
        return out

    def to_sparse_csr(  # type: ignore
            self,
            value: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_csr_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        (rowptr, col), perm = self.get_csr()
        if value is not None and perm is not None:
            value = value[perm]
        elif value is None:
            value = self._get_value()

        return paddle.sparse.sparse_csr_tensor(
            crows=rowptr,
            cols=col,
            values=value,
            shape=self.get_sparse_size(),
            place=self.device,
            stop_gradient=value.stop_gradient,
        )

    def to_sparse_csc(  # type: ignore
            self,
            value: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a :pytorch:`null`
        :class:`torch.sparse_csc_tensor`.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        if (not hasattr(paddle, 'sparse_csc')
                or not hasattr(paddle.sparse, 'sparse_csc_tensor')):
            raise NotImplementedError("Paddle don't support this method yet")

        (colptr, row), perm = self.get_csc()
        if value is not None and perm is not None:
            value = value[perm]
        elif value is None:
            value = self._get_value()

        return paddle.sparse.sparse_csc_tensor(
            ccol_indices=colptr,
            row_indices=row,
            values=value,
            shape=self.get_sparse_size(),
            place=self.device,
            stop_gradient=value.stop_gradient,
        )

    def to_sparse(
        self,
        *,
        layout: str = 'coo',
        value: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        r"""Converts :class:`EdgeIndex` into a
        :paddle:`null` :class:`paddle.sparse` tensor.

        Args:
            layout (str, optional): The layout of the sparse tensor.
            value (paddle.Tensor, optional): The values for non-zero elements.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
        """
        # Create sparse COO tensor
        if layout is None or layout == 'coo':
            return self.to_sparse_coo(
                value)  # You can keep other formats like CSR, CSC if necessary
        elif layout == 'csr':
            return self.to_sparse_csr(value)
        elif layout == 'csc':
            if (hasattr(paddle, 'sparse_csc')
                    and hasattr(paddle.sparse, 'sparse_csc_tensor')):
                return self.to_sparse_csc(value)
            return self.to_sparse_csr(value)
        else:
            raise ValueError(f"Unknown layout {layout}.")

    def to_sparse_tensor(
        self,
        value: Optional[Tensor] = None,
    ) -> SparseTensor:
        r"""Converts :class:`EdgeIndex` into a
        :class:`torch_sparse.SparseTensor`.
        Requires that :obj:`torch-sparse` is installed.

        Args:
            value (torch.Tensor, optional): The values for non-zero elements.
                (default: :obj:`None`)
        """
        return SparseTensor(
            row=self._data[0],
            col=self._data[1],
            rowptr=self._indptr if self.is_sorted_by_row else None,
            value=value,
            sparse_sizes=self.get_sparse_size(),
            is_sorted=self.is_sorted_by_row,
            trust_data=True,
        )

    def clone(self):

        data = self.data.clone()

        if data.dtype not in INDEX_DTYPES:
            return data

        if self._data.data_ptr() != data.data_ptr():
            out = EdgeIndex(data)
        else:  # In-place:
            self.data = data
            out = self

        # Copy metadata:
        out._sparse_size = self._sparse_size
        out._sort_order = self._sort_order
        out._is_undirected = self._is_undirected
        out._cat_metadata = self._cat_metadata

        # Convert cache (but do not consider `_value`):
        if self._indptr is not None:
            out._indptr = self._indptr.clone()

        if self._T_perm is not None:
            out._T_perm = self._T_perm.clone()

        _T_row, _T_col = self._T_index
        if _T_row is not None:
            _T_row = _T_row.clone()
        if _T_col is not None:
            _T_col = _T_col.clone()
        out._T_index = (_T_row, _T_col)

        if self._T_indptr is not None:
            out._T_indptr = self._T_indptr.clone()

        return out

    def to(self, *args, **kwargs):

        data = self._data.to(*args, **kwargs)

        if data.dtype not in INDEX_DTYPES:
            return data

        if self._data.data_ptr() != data.data_ptr():
            out = EdgeIndex(data)
        else:  # In-place:
            self.data = data
            out = self

        # Copy metadata:
        out._sparse_size = self._sparse_size
        out._sort_order = self._sort_order
        out._is_undirected = self._is_undirected
        out._cat_metadata = self._cat_metadata

        # Convert cache (but do not consider `_value`):
        if self._indptr is not None:
            out._indptr = self._indptr.to(*args, **kwargs)

        if self._T_perm is not None:
            out._T_perm = self._T_perm.to(*args, **kwargs)

        _T_row, _T_col = self._T_index
        if _T_row is not None:
            _T_row = _T_row.to(*args, **kwargs)
        if _T_col is not None:
            _T_col = _T_col.to(*args, **kwargs)
        out._T_index = (_T_row, _T_col)

        if self._T_indptr is not None:
            out._T_indptr = self._T_indptr.to(*args, **kwargs)

        return out

    def cuda(self, ):
        if self.place.is_gpu_place():
            return self
        index = self.to(device='gpu')
        return index

    def cpu(self, ):
        if self.place.is_cpu_place():
            return self
        index = self.to(device='cpu')
        return index

    def astype(self, dtype):
        index = self.to(dtype=dtype)
        return index

    @property
    def is_cpu(self, ):
        return self.place.is_cpu_place()

    @property
    def is_cuda(self, ):
        return self.place.is_gpu_place()

    # TODO Investigate how to avoid overlapping return types here.
    @overload
    def matmul(  # type: ignore
        self,
        other: 'EdgeIndex',
        input_value: Optional[Tensor] = None,
        other_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Tuple['EdgeIndex', Tensor]:
        pass

    @overload
    def matmul(
        self,
        other: Tensor,
        input_value: Optional[Tensor] = None,
        other_value: None = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Tensor:
        pass

    def matmul(
        self,
        other: Union[Tensor, 'EdgeIndex'],
        input_value: Optional[Tensor] = None,
        other_value: Optional[Tensor] = None,
        reduce: ReduceType = 'sum',
        transpose: bool = False,
    ) -> Union[Tensor, Tuple['EdgeIndex', Tensor]]:
        r"""Performs a matrix multiplication of the matrices :obj:`input` and
        :obj:`other`.
        If :obj:`input` is a :math:`(n \times m)` matrix and :obj:`other` is a
        :math:`(m \times p)` tensor, then the output will be a
        :math:`(n \times p)` tensor.
        See :meth:`torch.matmul` for more information.

        :obj:`input` is a sparse matrix as denoted by the indices in
        :class:`EdgeIndex`, and :obj:`input_value` corresponds to the values
        of non-zero elements in :obj:`input`.
        If not specified, non-zero elements will be assigned a value of
        :obj:`1.0`.

        :obj:`other` can either be a dense :class:`torch.Tensor` or a sparse
        :class:`EdgeIndex`.
        if :obj:`other` is a sparse :class:`EdgeIndex`, then :obj:`other_value`
        corresponds to the values of its non-zero elements.

        This function additionally accepts an optional :obj:`reduce` argument
        that allows specification of an optional reduction operation.
        See :meth:`torch.sparse.mm` for more information.

        Lastly, the :obj:`transpose` option allows to perform matrix
        multiplication where :obj:`input` will be first transposed, *i.e.*:

        .. math::

            \textrm{input}^{\top} \cdot \textrm{other}

        Args:
            other (torch.Tensor or EdgeIndex): The second matrix to be
                multiplied, which can be sparse or dense.
            input_value (torch.Tensor, optional): The values for non-zero
                elements of :obj:`input`.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            other_value (torch.Tensor, optional): The values for non-zero
                elements of :obj:`other` in case it is sparse.
                If not specified, non-zero elements will be assigned a value of
                :obj:`1.0`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation, one of
                :obj:`"sum"`/:obj:`"add"`, :obj:`"mean"`,
                :obj:`"min"`/:obj:`amin` or :obj:`"max"`/:obj:`amax`.
                (default: :obj:`"sum"`)
            transpose (bool, optional): If set to :obj:`True`, will perform
                matrix multiplication based on the transposed :obj:`input`.
                (default: :obj:`False`)
        """
        return matmul(self, other, input_value, other_value, reduce, transpose)

    def sparse_narrow(
        self,
        dim: int,
        start: Union[int, Tensor],
        length: int,
    ) -> 'EdgeIndex':
        r"""Returns a new :class:`EdgeIndex` that is a narrowed version of
        itself. Narrowing is performed by interpreting :class:`EdgeIndex` as a
        sparse matrix of shape :obj:`(num_rows, num_cols)`.

        In contrast to :meth:`torch.narrow`, the returned tensor does not share
        the same underlying storage anymore.

        Args:
            dim (int): The dimension along which to narrow.
            start (int or torch.Tensor): Index of the element to start the
                narrowed dimension from.
            length (int): Length of the narrowed dimension.
        """
        dim = dim + 2 if dim < 0 else dim
        if dim != 0 and dim != 1:
            raise ValueError(f"Expected dimension to be 0 or 1 (got {dim})")

        if start < 0:
            raise ValueError(f"Expected 'start' value to be positive "
                             f"(got {start})")

        if dim == 0:
            if self.is_sorted_by_row:
                (rowptr, col), _ = self.get_csr()
                rowptr = rowptr.narrow(0, start, length + 1)

                if rowptr.numel() < 2:
                    row, col = self._data[0, :0], self._data[1, :0]
                    rowptr = None
                    num_rows = 0
                else:
                    col = col[rowptr[0]:rowptr[-1]]
                    rowptr = rowptr - rowptr[0]
                    num_rows = rowptr.numel() - 1

                    row = paddle.arange(
                        num_rows,
                        dtype=col.dtype,
                        device=col.place,
                    ).repeat_interleave(
                        paddle.diff(rowptr),
                        output_size=col.numel(),
                    )

                edge_index = EdgeIndex(
                    paddle.stack([row, col], axis=0),
                    sparse_size=(num_rows, self.sparse_size(1)),
                    sort_order='row',
                )
                edge_index._indptr = rowptr
                return edge_index

            else:
                mask = self._data[0] >= start
                mask &= self._data[0] < (start + length)
                offset = paddle.to_tensor([[start], [0]], place=self.device)
                edge_index = paddle.subtract(self[:, mask],
                                             offset)  # type: ignore
                edge_index._sparse_size = (length, edge_index._sparse_size[1])
                return edge_index

        else:
            assert dim == 1

            if self.is_sorted_by_col:
                (colptr, row), _ = self.get_csc()
                colptr = colptr.narrow(0, start, length + 1)

                if colptr.numel() < 2:
                    row, col = self._data[0, :0], self._data[1, :0]
                    colptr = None
                    num_cols = 0
                else:
                    row = row[colptr[0]:colptr[-1]]
                    colptr = colptr - colptr[0]
                    num_cols = colptr.numel() - 1

                    col = paddle.arange(
                        num_cols,
                        dtype=row.dtype,
                        device=row.place,
                    ).repeat_interleave(
                        colptr.diff(),
                        output_size=row.numel(),
                    )

                edge_index = EdgeIndex(
                    paddle.stack([row, col], axis=0),
                    sparse_size=(self.sparse_size(0), num_cols),
                    sort_order='col',
                )
                edge_index._indptr = colptr
                return edge_index

            else:
                mask = self._data[1] >= start
                mask &= self._data[1] < (start + length)
                offset = paddle.to_tensor([[0], [start]], place=self.device)
                edge_index = paddle.subtract(self[:, mask],
                                             offset)  # type: ignore
                edge_index._sparse_size = (edge_index._sparse_size[0], length)
                return edge_index

    def to_vector(self) -> Tensor:
        r"""Converts :class:`EdgeIndex` into a one-dimensional index
        vector representation.
        """
        num_rows, num_cols = self.get_sparse_size()

        if num_rows * num_cols > paddle_geometric.typing.MAX_INT64:
            raise ValueError("'to_vector()' will result in an overflow")

        return self._data[0] * num_rows + self._data[1]

    def to_dict(self):
        return {
            'data': self._data,
            '_sparse_size': self._sparse_size,
            '_sort_order': self._sort_order,
            '_is_undirected': self._is_undirected,
        }

    @classmethod
    def from_dict(cls, data_dict) -> 'Index':
        r"""Creates a new :class:`Index` instance from a dictionary."""
        out = cls(
            data_dict['data'],
            sparse_size=data_dict['_sparse_size'],
            sort_order=data_dict['_sort_order'],
            is_undirected=data_dict['_is_undirected'],
        )
        out._indptr = data_dict.get('_indptr', None)
        return out

    @property
    def device(self, ):
        place = self.place
        if place.is_cpu_place():
            device = 'cpu'
        elif place.is_gpu_place():
            device_id = place.gpu_device_id()
            device = 'gpu:' + str(device_id)
        elif place.is_xpu_place():
            raise ValueError("xpu is not supported")
        else:
            raise ValueError(f"The device specification {place} is invalid")

        return device

    def __repr__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        tensor_str = paddle.tensor.to_string._format_tensor(self._data, '')

        suffixes = []
        num_rows, num_cols = self.sparse_size()
        if num_rows is not None or num_cols is not None:
            size_repr = f"({num_rows or '?'}, {num_cols or '?'})"
            suffixes.append(f'sparse_size={size_repr}')
        suffixes.append(f'nnz={self._data.shape[1]}')  # 使用Paddle的shape属性
        if self.device != paddle.get_device():
            suffixes.append(f"device='{self.device}'")
        if self.dtype != paddle.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append(f'sort_order={self.sort_order}')
        if self.is_undirected:
            suffixes.append('is_undirected=True')

        return f"{prefix}{tensor_str}, {', '.join(suffixes)})"

    def __str__(self) -> str:
        return self.__repr__()

    # Helpers #################################################################

    def _shallow_copy(self) -> 'EdgeIndex':
        out = EdgeIndex(self._data)
        out._sparse_size = self._sparse_size
        out._sort_order = self._sort_order
        out._is_undirected = self._is_undirected
        out._indptr = self._indptr
        out._T_perm = self._T_perm
        out._T_index = self._T_index
        out._T_indptr = self._T_indptr
        out._value = self._value
        out._cat_metadata = self._cat_metadata
        return out

    def _clear_metadata(self) -> 'EdgeIndex':
        self._sparse_size = (None, None)
        self._sort_order = None
        self._is_undirected = False
        self._indptr = None
        self._T_perm = None
        self._T_index = (None, None)
        self._T_indptr = None
        self._value = None
        self._cat_metadata = None
        return self

    def __getitem__(self, index):
        # Step 1: Call the underlying Tensor's __getitem__
        out = self._data.__getitem__(index)
        # Step 2: Check if the output shape is still (2, n)
        if isinstance(out, Tensor) and out.ndim == 2 and out.shape[0] == 2:
            out = out.contiguous()
            new = EdgeIndex(out)
            # Step 3: Preserve some metadata
            # By default, do not inherit cache,
            # because these caches are invalid after indexing
            new._sparse_size = self._sparse_size
            new._indptr = None
            new._T_perm = None
            new._T_index = (None, None)
            new._T_indptr = None
            new._value = None
            new._cat_metadata = None

            # Only pure slices (like 1::2 or slice(None)) preserve sorting
            idx_tuple = index if isinstance(index, tuple) else (index, )

            # Check if it is an identity slice
            is_identity = False
            if index is Ellipsis:
                is_identity = True
            elif isinstance(index, tuple):
                is_identity = all(
                    isinstance(i, slice) and i.start is None and i.stop is None
                    and i.step is None for i in index)
            # Check if it is basic indexing or a boolean mask
            is_basic_or_bool_mask = all(
                isinstance(i, (slice, type(Ellipsis))) or i is None or (
                    isinstance(i, Tensor) and i.dtype == paddle.bool)
                for i in idx_tuple)

            if is_identity:
                # Fully inherit metadata
                new._sort_order = self._sort_order
                new._is_undirected = self._is_undirected
            else:
                # Partial indexing → sort_order may
                # be preserved, is_undirected=False
                if is_basic_or_bool_mask:
                    new._sort_order = self._sort_order
                else:
                    new._sort_order = None
                new._is_undirected = False
            return new

        # Index 情况：一维结果
        if isinstance(out, Tensor) and out.ndim == 1:
            out = out.contiguous()
            # 判断是否完整选取行/列
            row_index = index[0] if isinstance(index, tuple) else index
            col_index = index[1] if isinstance(
                index, tuple) and len(index) > 1 else slice(None)

            row_index = row_index + self._data.shape[0] if isinstance(
                row_index, int) and row_index < 0 else row_index
            col_index = col_index + self._data.shape[1] if isinstance(
                col_index, int) and col_index < 0 else col_index

            def is_single_index(s):
                if isinstance(s, int):
                    return True
                elif isinstance(s, slice):
                    # 如果 slice.start == slice.stop 或
                    # step =1 并且 stop-start=1 时可以视作单行
                    # 但 stop=None 或 start=None 时无法计算长度 → 当作非单行
                    return s.start is not None and s.stop is not None and (
                        s.stop - s.start) == 1
                return False

            is_single_row = is_single_index(row_index)
            is_single_col = is_single_index(col_index)

            # 仅当单行/单列选择时返回 Index
            if not (is_single_row or is_single_col):
                return out  # 部分选择返回 Tensor

            # 返回 Index
            dim_size = self._sparse_size[
                0] if row_index == 0 else self._sparse_size[1]
            # 构造 is_sorted
            if isinstance(
                    row_index, int
            ) and row_index == 0 and self._sort_order == SortOrder.ROW:
                is_sorted = True
            elif isinstance(
                    row_index, int
            ) and row_index == 1 and self._sort_order == SortOrder.COL:
                is_sorted = True
            else:
                is_sorted = False

            def is_full_col(col_index):
                # Return True if all columns are
                # selected (for _indptr preservation).
                n_cols = self._data.shape[1]
                # Case 1: 全选
                if col_index is None:
                    return True
                # Case 2: 单整数索引 -> 一定不是全选
                if isinstance(col_index, int):
                    return False

                # Case 3: 切片情况
                if isinstance(col_index, slice):
                    if col_index.start is not None:
                        start = col_index.start
                    else:
                        start = 0
                    if col_index.stop is not None:
                        stop = col_index.stop
                    else:
                        stop = n_cols

                    step = col_index.step if col_index.step is not None else 1

                    # 负数修正
                    if start < 0:
                        start = n_cols + start
                    if stop < 0:
                        stop = n_cols + stop

                    # 满足起点、终点、步长条件时表示完整列选择
                    return start == 0 and stop == n_cols and step == 1

                # Case 4: list 或 tensor 选择
                if isinstance(col_index, (list, paddle.Tensor)):
                    # 只在这些条件下才视为“完整列选择”：
                    # - 元素个数与列数一致
                    # - 值集合 == 全列集合
                    if len(col_index) == n_cols:
                        col_values = (col_index.tolist() if isinstance(
                            col_index, paddle.Tensor) else col_index)
                        col_values = [int(x) % n_cols for x in col_values]
                        return sorted(set(col_values)) == list(range(n_cols))
                    return False

                return False

            if ((row_index == 0 and self._sort_order == SortOrder.ROW) or
                (row_index == 1 and self._sort_order == SortOrder.COL)) and \
                    is_full_col(col_index):
                indptr = self._indptr
            else:
                indptr = None

            new = Index(out)
            new._dim_size = dim_size
            new._is_sorted = is_sorted
            new._indptr = indptr
            return new
        # Other cases → return the Tensor result
        return out

    def flip(self, axis, **kwargs):
        return paddle.flip(x=self, axis=axis, **kwargs)

    def index_select(self, index, axis=0, name=None, **kwargs):
        return paddle.index_select(self, index, axis=axis, name=name, **kwargs)

    def narrow(self, dim, start, length):
        return paddle.narrow(self, dim=dim, start=start, length=length)

    def __iter__(self):
        row = self[0]
        col = self[1]
        return iter((row, col))

    def __add__(self, other):
        return paddle.add(self, other)

    def __radd__(self, other):
        return paddle.add(other, self)

    def __sub__(self, other):
        return paddle.subtract(self, other)

    def __rsub__(self, other):
        return paddle.subtract(other, self)

    def __matmul__(self, other):
        return paddle.matmul(self, other)

    def __rmatmul__(self, other):
        return paddle.matmul(other, self)


@register_for("concat")(EdgeIndex)
def concat_edgeindex(x, axis=0, name=None):

    data_list = [x_i.data for x_i in x]
    concat_paddle = paddle.concat(data_list, axis=axis)

    if axis != 1 and axis != -1:  # No valid `EdgeIndex` anymore.
        return concat_paddle

    if any([not isinstance(tensor, EdgeIndex) for tensor in x]):
        return concat_paddle

    out = EdgeIndex(concat_paddle)

    nnz_list = [t.shape[1] for t in x]
    sparse_size_list = [t.sparse_size() for t in x]  # type: ignore
    sort_order_list = [t._sort_order for t in x]  # type: ignore
    is_undirected_list = [t.is_undirected for t in x]  # type: ignore

    # Post-process `sparse_size`:
    total_num_rows: Optional[int] = 0
    for num_rows, _ in sparse_size_list:
        if num_rows is None:
            total_num_rows = None
            break
        assert isinstance(total_num_rows, int)
        total_num_rows = max(num_rows, total_num_rows)

    total_num_cols: Optional[int] = 0
    for _, num_cols in sparse_size_list:
        if num_cols is None:
            total_num_cols = None
            break
        assert isinstance(total_num_cols, int)
        total_num_cols = max(num_cols, total_num_cols)

    out._sparse_size = (total_num_rows, total_num_cols)

    # Post-process `is_undirected`:
    out._is_undirected = all(is_undirected_list)

    out._cat_metadata = CatMetadata(
        nnz=nnz_list,
        sparse_size=sparse_size_list,
        sort_order=sort_order_list,
        is_undirected=is_undirected_list,
    )
    return out


@register_for("flip")(EdgeIndex)
def flip_edgeindex(x: EdgeIndex, axis: Union[List[int], Tuple[int, ...]],
                   **kwargs) -> EdgeIndex:

    data = paddle.flip(x.data, axis=axis)
    out = EdgeIndex(data)

    out._value = x._value
    out._is_undirected = x.is_undirected

    if isinstance(axis, int):
        axis = [
            axis,
        ]

    # Flip metadata and cache:
    if 0 in axis or -2 in axis:
        out._sparse_size = x.sparse_size()[::-1]

    if len(axis) == 1 and (axis[0] == 0 or axis[0] == -2):
        if x.is_sorted_by_row:
            out._sort_order = SortOrder.COL
        elif x.is_sorted_by_col:
            out._sort_order = SortOrder.ROW

        out._indptr = x._T_indptr
        out._T_perm = x._T_perm
        out._T_index = x._T_index[::-1]
        out._T_indptr = x._indptr

    return out


@register_for('index_select')(EdgeIndex)
def index_select_edgeindex(x: EdgeIndex, index: Tensor, axis: int = 0,
                           name=None, *, out=None) -> Union[EdgeIndex, Tensor]:

    result = paddle.index_select(x.data, index, axis, name, out=out)
    if (axis == 1 or axis == -1) and out is None:
        result = EdgeIndex(result)
        result._sparse_size = x.sparse_size()

    return result


@register_for('narrow')(EdgeIndex)
def narrow_edgeindex(input: EdgeIndex, dim: int, start,
                     length) -> Union[EdgeIndex, Tensor]:

    end = start + length
    if start <= 0 and end > input.data.shape[dim]:
        return input._shallow_copy()

    out = paddle.narrow(input.data, dim, start, length)
    if dim == 1 or dim == -1:

        out = EdgeIndex(out)
        out._sparse_size = input.sparse_size()
        # NOTE We could potentially maintain `rowptr`/`colptr` attributes here,
        # but it is not really clear if this is worth it. The most important
        # information, the sort order, needs to be maintained though:
        out._sort_order = input._sort_order
    return out


@register_for('unbind')(EdgeIndex)
def unbind_edgeindex(input: EdgeIndex, axis=0,
                     **kwargs) -> Union[EdgeIndex, Tensor]:

    if axis == 0 or axis == -2:
        row = input[0]
        assert isinstance(row, Index)
        col = input[1]
        assert isinstance(col, Index)
        return [row, col]
    return paddle.unbind(input.data, axis=axis)


@register_for("add")(EdgeIndex)
def add_edgeindex(x, y, **kwargs):
    # 提取底层 Tensor
    x_data = x.data if isinstance(x, EdgeIndex) else x
    y_data = y.data if isinstance(y, EdgeIndex) else y

    alpha = kwargs.get('alpha', 1)
    y_data = y_data * alpha

    # 调用 Paddle 的原始加法
    out_data = x_data + y_data
    if out_data.dtype not in INDEX_DTYPES:
        return out_data
    if out_data.dim() != 2 or out_data.shape[0] != 2:
        return out_data

    out = EdgeIndex(out_data)

    if isinstance(y, Tensor) and y.numel() <= 1:
        y = int(y)

    if isinstance(y, int):
        size = maybe_add(x._sparse_size, y, alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = x._sort_order
        out._is_undirected = x.is_undirected
        out._T_perm = x._T_perm

    elif isinstance(y, Tensor) and tuple(y.shape) == (2, 1):
        size = maybe_add(x._sparse_size, y.view(-1).tolist(), alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = x._sort_order
        if paddle.equal(y[0], y[1]):
            out._is_undirected = x.is_undirected
        out._T_perm = x._T_perm

    elif isinstance(y, EdgeIndex):
        size = maybe_add(x._sparse_size, y._sparse_size, alpha)
        assert len(size) == 2
        out._sparse_size = size
    return out


@register_for("subtract")(EdgeIndex)
def subtract_edgeindex(x, y, **kwargs):
    x_data = x.data if isinstance(x, EdgeIndex) else x
    y_data = y.data if isinstance(y, EdgeIndex) else y

    alpha = kwargs.get('alpha', 1)
    y_data = y_data * alpha

    if isinstance(x_data, Tensor) and isinstance(y_data, Tensor):
        if x_data.dtype == paddle.int32 and y_data.dtype == paddle.int64:
            x_data = x_data.astype(paddle.int64)

        if x_data.dtype == paddle.int64 and y_data.dtype == paddle.int32:
            y_data = y_data.astype(paddle.int64)

    out = x_data - y_data

    if out.dtype not in INDEX_DTYPES:
        return out
    if out.dim() != 2 or out.shape[0] != 2:
        return out

    out = EdgeIndex(out)

    if isinstance(y, Tensor) and y.numel() <= 1:
        y = int(y)

    if isinstance(y, int):
        size = maybe_sub(x._sparse_size, y, alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = x._sort_order
        out._is_undirected = x.is_undirected
        out._T_perm = x._T_perm

    elif isinstance(y, Tensor) and tuple(y.shape) == (2, 1):
        size = maybe_sub(x._sparse_size, y.view(-1).tolist(), alpha)
        assert len(size) == 2
        out._sparse_size = size
        out._sort_order = x._sort_order
        if paddle.equal(y[0], y[1]):
            out._is_undirected = x.is_undirected
        out._T_perm = x._T_perm
    return out


@register_for("matmul")(EdgeIndex)
def matmul_edgeindex(x, y, **kwargs):
    return matmul(x, y)


@register_for("mm")(EdgeIndex)
def mm_edgeindex(x, y, **kwargs):
    return matmul(x, y)


def matmul(
    input: EdgeIndex,
    other: Union[Tensor, EdgeIndex],
    input_value: Optional[Tensor] = None,
    other_value: Optional[Tensor] = None,
    reduce: ReduceType = 'sum',
    transpose: bool = False,
) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:

    if not isinstance(other, EdgeIndex):
        if other_value is not None:
            raise ValueError("'other_value' not supported for sparse-dense "
                             "matrix multiplication")
        # return _spmm(input, other, input_value, reduce, transpose)
        if input_value is not None:
            dense = input.to_dense(input_value)
        else:
            dense = input.to_dense()

        if transpose:
            output = dense.t() @ other
        else:
            output = dense @ other
        return output

    if reduce not in ['sum', 'add']:
        raise NotImplementedError(f"`reduce='{reduce}'` not yet supported for "
                                  f"sparse-sparse matrix multiplication")

    transpose &= not input.is_undirected or input_value is not None

    if paddle_geometric.typing.NO_MKL:  # pragma: no cover
        sparse_input = input.to_sparse_coo(input_value)
    elif input.is_sorted_by_col:
        sparse_input = input.to_sparse_csc(input_value)
    else:
        sparse_input = input.to_sparse_csr(input_value)

    if transpose:
        sparse_input = sparse_input.t()

    if paddle_geometric.typing.NO_MKL:  # pragma: no cover
        other = other.to_sparse_coo(other_value)
    elif other.is_sorted_by_col:
        other = other.to_sparse_csc(other_value)
    else:
        other = other.to_sparse_csr(other_value)

    out = paddle.matmul(sparse_input, other)

    rowptr: Optional[Tensor] = None
    if out.is_sparse_csr():
        rowptr = out.crows().to(input.dtype)
        col = out.cols().to(input.dtype)
        edge_index = csr_to_coo_paddle(rowptr, col, out_int32=rowptr.dtype
                                       != paddle.int64)
    elif out.is_sparse_coo():  # pragma: no cover
        out = out.coalesce()
        edge_index = out.indices()

    else:
        raise NotImplementedError

    edge_index = EdgeIndex(edge_index)
    edge_index._sort_order = SortOrder.ROW
    edge_index._sparse_size = (out.shape[0], out.shape[1])
    edge_index._indptr = rowptr

    return edge_index, out.values()


def csr_to_coo_paddle(crow_indices, col_indices, out_int32=False):
    num_rows = crow_indices.shape[0] - 1
    row_counts = crow_indices[1:] - crow_indices[:-1]
    row_indices = paddle.repeat_interleave(paddle.arange(num_rows), row_counts)
    tensor = paddle.sparse.sparse_coo_tensor(
        paddle.stack([row_indices, col_indices], axis=0),
        paddle.ones((row_indices.shape[0], ), dtype=paddle.float32),
        shape=(num_rows, num_rows),
    )
    if out_int32:
        tensor = tensor.to(paddle.int32)
    return tensor


class SortReturnType(NamedTuple):
    values: EdgeIndex
    indices: Optional[Tensor]


# Sparse-Dense Matrix Multiplication ##########################################

# def _paddle_sparse_spmm(
#     input: EdgeIndex,
#     other: paddle.Tensor,
#     value: Optional[paddle.Tensor] = None,
#     reduce: ReduceType = 'sum',
#     transpose: bool = False,
# ) -> paddle.Tensor:
#     # Paddle does not have a direct equivalent
#     # to `torch-sparse`, so we assume
#     # custom implementation for sparse matrix multiplication.
#     assert paddle_geometric.typing.WITH_PADDLE_SPARSE
#     reduce = PYG_REDUCE[reduce] if reduce in PYG_REDUCE else reduce

#     # Optional arguments for backpropagation:
#     colptr: Optional[paddle.Tensor] = None
#     perm: Optional[paddle.Tensor] = None

#     if not transpose:
#         assert input.is_sorted_by_row
#         (rowptr, col), _ = input.get_csr()
#         row = input._data[0]
#         if other.requires_grad and reduce in ['sum', 'mean']:
#             (colptr, _), perm = input.get_csc()
#     else:
#         assert input.is_sorted_by_col
#         (rowptr, col), _ = input.get_csc()
#         row = input._data[1]
#         if other.requires_grad and reduce in ['sum', 'mean']:
#             (colptr, _), perm = input.get_csr()

#     if reduce == 'sum':
#         return paddle.sparse.spmv(  # Sparse matrix-vector multiplication
#             row, rowptr, col, value, colptr, perm, other
#         )

#     if reduce == 'mean':
#         rowcount = paddle.diff(rowptr) if other.requires_grad else None
# Sparse matrix-vector multiplication with mean
#         return paddle.sparse.spmv(
#             row, rowptr, col, value, rowcount, colptr, perm, other
#         )

#     if reduce == 'min':
#         return paddle.sparse.spmv_min(rowptr, col, value, other)[0]

#     if reduce == 'max':
#         return paddle.sparse.spmv_max(rowptr, col, value, other)[0]

#     raise NotImplementedError

# class _PaddleSPMM(PyLayer):
#     @staticmethod
#     def forward(
#       ctx: Any,
#       input: EdgeIndex,
#       other: Tensor,
#       value: Optional[Tensor] = None,
#       reduce: ReduceType = 'sum',
#       transpose: bool = False) -> Tensor:
#         reduce = PYG_REDUCE[reduce] if reduce in PYG_REDUCE else reduce

#         value = value.detach() if value is not None else value
#         if other.requires_grad:
#             other = other.detach()
#             ctx.save_for_backward(input, value)
#             ctx.reduce = reduce
#             ctx.transpose = transpose

#         if not transpose:
#             assert input.is_sorted_by_row
#             adj = input.to_sparse_csr(value)
#         else:
#             assert input.is_sorted_by_col
#             adj = input.to_sparse_csc(value).t()

#         if paddle_geometric.typing.WITH_PT20 and not other.is_gpu():
#             return paddle.sparse.mm(adj, other, reduce)
#         else:  # pragma: no cover
#             assert reduce == 'sum'
#             return adj @ other

#     @staticmethod
#     def backward(ctx: Any, *grad_outputs: Any):
#         grad_out, = grad_outputs

#         other_grad: Optional[Tensor] = None
#         if ctx.needs_input_grad[1]:
#             input, value = ctx.saved_tensors
#             assert ctx.reduce == 'sum'

#             if not ctx.transpose:
#                 if value is None and input.is_undirected:
#                     adj = input.to_sparse_csr(value)
#                 else:
#                     (colptr, row), perm = input.get_csc()
#                     if value is not None and perm is not None:
#                         value = value[perm]
#                     else:
#                         value = input._get_value()
#                     adj = paddle.sparse_csr_tensor(
#                         crow_indices=colptr,
#                         col_indices=row,
#                         values=value,
#                         shape=input.get_sparse_size()[::-1],
#                         device=input.device,
#                     )
#             else:
#                 if value is None and input.is_undirected:
#                     adj = input.to_sparse_csc(value).t()
#                 else:
#                     (rowptr, col), perm = input.get_csr()
#                     if value is not None and perm is not None:
#                         value = value[perm]
#                     else:
#                         value = input._get_value()
#                     adj = paddle.sparse_csr_tensor(
#                         crow_indices=rowptr,
#                         col_indices=col,
#                         values=value,
#                         shape=input.get_sparse_size()[::-1],
#                         device=input.device,
#                     )

#             other_grad = adj @ grad_out

#         if ctx.needs_input_grad[2]:
#             raise NotImplementedError(
#                   "Gradient computation for 'value' not yet supported
#           ")

#         return None, other_grad, None, None, None

# def _scatter_spmm(
#     input: EdgeIndex,
#     other: Tensor,
#     value: Optional[Tensor] = None,
#     reduce: ReduceType = 'sum',
#     transpose: bool = False,
# ) -> Tensor:
#     from torch_geometric.utils import scatter

#     if not transpose:
#         other_j = other[input._data[1]]
#         index = input._data[0]
#         dim_size = input.get_sparse_size(0)
#     else:
#         other_j = other[input._data[0]]
#         index = input._data[1]
#         dim_size = input.get_sparse_size(1)

#     other_j = other_j * value.view(-1, 1) if value is not None else other_j
#     return scatter(other_j, index, 0, dim_size=dim_size, reduce=reduce)

# def _spmm(
#     input: EdgeIndex,
#     other: Tensor,
#     value: Optional[Tensor] = None,
#     reduce: ReduceType = 'sum',
#     transpose: bool = False,
# ) -> Tensor:

#     if reduce not in ['sum', 'mean', 'amin', 'amax', 'add', 'min', 'max']:
#         raise ValueError(f"`reduce='{reduce}'` is not a valid reduction")

#     if not transpose and not input.is_sorted_by_row:
#         cls_name = input.__class__.__name__
#         raise ValueError(f"'matmul(..., transpose=False)' requires "
#                          f"'{cls_name}' to be sorted by rows")

#     if transpose and not input.is_sorted_by_col:
#         cls_name = input.__class__.__name__
#         raise ValueError(f"'matmul(..., transpose=True)' requires "
#                          f"'{cls_name}' to be sorted by columns")

#     if paddle_geometric.typing.WITH_TORCH_SPARSE:
#         return _paddle_sparse_spmm(input, other, value, reduce, transpose)

#     if value is not None and value.requires_grad:
#         return _scatter_spmm(input, other, value, reduce, transpose)

#     # If no gradient, perform regular matrix multiplication
#     if reduce == 'sum':
#         return paddle.sparse.matmul(input, other)

#     if reduce == 'mean':
#         out = paddle.sparse.matmul(input, other)
#         count = input.get_indptr().diff()
#         return out / count.clamp_(min=1).to(out.dtype).reshape([-1, 1])

#     if reduce == 'max':
#         return paddle.sparse.spmm_max(input, other)

#     raise NotImplementedError

# def matmul(
#     input: EdgeIndex,
#     other: Union[Tensor, EdgeIndex],
#     input_value: Optional[Tensor] = None,
#     other_value: Optional[Tensor] = None,
#     reduce: ReduceType = 'sum',
#     transpose: bool = False,
# ) -> Union[Tensor, Tuple[EdgeIndex, Tensor]]:

#     if not isinstance(other, EdgeIndex):
#         if other_value is not None:
#             raise ValueError("'other_value' not supported for sparse-dense "
#                              "matrix multiplication")
#         return _spmm(input, other, input_value, reduce, transpose)

#     if reduce not in ['sum', 'add']:
#         raise NotImplementedError(
#               f"`reduce='{reduce}'` not yet supported for "
#               f"sparse-sparse matrix multiplication")

#     transpose &= not input.is_undirected or input_value is not None

#     if input.is_sorted_by_col:
#         sparse_input = input.to_sparse_csc(input_value)
#     else:
#         sparse_input = input.to_sparse_csr(input_value)

#     if transpose:
#         sparse_input = sparse_input.t()

#     if other.is_sorted_by_col:
#         other = other.to_sparse_csc(other_value)
#     else:
#         other = other.to_sparse_csr(other_value)

#     out = paddle.sparse.matmul(sparse_input, other)

#     rowptr: Optional[Tensor] = None
#     if out.layout == paddle.sparse_csr:
#         rowptr = out.crow_indices().to(input.dtype)
#         col = out.col_indices().to(input.dtype)
#         edge_index = paddle.convert_indices_from_csr_to_coo(
#             rowptr, col, out_int32=rowptr.dtype != paddle.int64)

#     elif out.layout == paddle.sparse.coo:
#         edge_index = out.indices()

#     edge_index = EdgeIndex(edge_index)
#     edge_index._sort_order = SortOrder.ROW
#     edge_index._sparse_size = (out.shape[0], out.shape[1])
#     edge_index._indptr = rowptr

#     return edge_index, out.values()

# # Implements the matrix multiplication (mm) for EdgeIndex tensors
# def _mm(
#     input: EdgeIndex,
#     other: Union[paddle.Tensor, EdgeIndex],
# ) -> Union[paddle.Tensor, Tuple[EdgeIndex, paddle.Tensor]]:
#     return matmul(input, other)

# # Implements the sparse matrix multiplication with
# addition (addmm) for EdgeIndex tensors
# def _addmm(
#     input: paddle.Tensor,
#     mat1: EdgeIndex,
#     mat2: paddle.Tensor,
#     beta: float = 1.0,
#     alpha: float = 1.0,
# ) -> paddle.Tensor:
#     assert paddle.abs(input).sum() == 0.0  # Ensure the input tensor is zero
#     out = matmul(mat1, mat2)
#     assert isinstance(out, paddle.Tensor)
#     return alpha * out if alpha != 1.0 else out

# # Implements the sparse matrix multiplication with
# reduction (mm_reduce) for EdgeIndex tensors
# def _mm_reduce(
#     mat1: EdgeIndex,
#     mat2: paddle.Tensor,
#     reduce: ReduceType = 'sum',
# ) -> Tuple[paddle.Tensor, paddle.Tensor]:
#     out = matmul(mat1, mat2, reduce=reduce)
#     assert isinstance(out, paddle.Tensor)
#     return out, out  # We return a dummy tensor for `argout` for now.
