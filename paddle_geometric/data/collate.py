from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import EdgeIndex, Index
from paddle_geometric.data.data import BaseData
from paddle_geometric.data.storage import BaseStorage, NodeStorage
from paddle_geometric.edge_index import SortOrder
from paddle_geometric.typing import (
    SparseTensor,
    TensorFrame,
    paddle_frame,
    paddle_sparse,
)
from paddle_geometric.utils import cumsum, is_paddle_sparse_tensor, is_sparse
from paddle_geometric.utils.sparse import cat

T = TypeVar('T')
SliceDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]
IncDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    if not isinstance(data_list, (list, tuple)):
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # type: ignore
    else:
        out = cls()

    out.stores_as(data_list[0])  # type: ignore

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    device: Optional[paddle.CPUPlace] = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:  # type: ignore
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:
                continue

            values = [store[attr] for store in stores]

            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            if attr == 'ptr':
                continue

            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            if isinstance(value,
                          Tensor) and paddle.device.get_device() != 'cpu':
                device = value.place

            out_store[attr] = value

            if key is not None:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f'{attr}_batch'] = batch
                out_store[f'{attr}_ptr'] = ptr

        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(paddle.to_tensor(repeats))

    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:
    elem = values[0]

    if isinstance(elem, Tensor) and not is_sparse(elem):
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.ndim == 0:
            values = [value.unsqueeze(0) for value in values]
        sizes = paddle.to_tensor(
            [value.shape[cat_dim or 0] for value in values])
        slices = cumsum(sizes)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.ndim > 1 or int(incs[-1]) != 0:
                values = [value + inc.to(value.device) if hasattr(inc, 'to') else value + inc
                          for value, inc in zip(values, incs)]
        else:
            incs = None

        if getattr(elem, 'is_nested', False):
            tensors = []
            for nested_tensor in values:
                tensors.extend(paddle.unbind(nested_tensor, axis=0))
            value = paddle.nn.utils.rnn.pad_sequence(tensors)
            return value, slices, incs

        value = paddle.concat(values, axis=cat_dim or 0)
        if increment and isinstance(value, Index) and values[0].is_sorted:
            # Check whether the whole `Index` is sorted:
            if (value.diff() >= 0).all():
                value._is_sorted = True

        if increment and isinstance(value, EdgeIndex) and values[0].is_sorted:
            # Check whether the whole `EdgeIndex` is sorted by row:
            if values[0].is_sorted_by_row and (value[0].diff() >= 0).all():
                value._sort_order = SortOrder.ROW
            # Check whether the whole `EdgeIndex` is sorted by column:
            elif values[0].is_sorted_by_col and (value[1].diff() >= 0).all():
                value._sort_order = SortOrder.COL
        return value, slices, incs
    elif isinstance(elem, TensorFrame):
        key = str(key)
        sizes = paddle.to_tensor([value.num_rows for value in values])
        slices = cumsum(sizes)
        value = paddle_frame.cat(values, dim=0)
        return value, slices, None

    elif is_sparse(elem) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = []
        for value in values:
            tmp = []
            for dim in cat_dims:
                tmp.append(value.shape[dim]
                           if isinstance(value, Tensor) else value.size(dim))
            repeats.append(tmp)

        slices = cumsum(paddle.to_tensor(repeats))
        if is_paddle_sparse_tensor(elem):
            value = cat(values, dim=cat_dim)
        else:
            value = paddle_sparse.cat(values, dim=cat_dim)
        return value, slices, None
    elif isinstance(elem, (int, float)):
        value = paddle.to_tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value += incs
        else:
            incs = None
        slices = paddle.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for k in elem.keys():
            value_dict[k], slice_dict[k], inc_dict[k] = _collate(
                k, [v[k] for v in values], data_list, stores, increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and len(elem) > 0 and isinstance(elem[0], (Tensor, SparseTensor))):
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        slices = paddle.arange(len(values) + 1)
        return values, slices, None


def _batch_and_ptr(
    slices: Any,
    device: Optional[paddle.CPUPlace] = None,
) -> Tuple[Any, Any]:
    if isinstance(slices, Tensor) and slices.ndim == 1:
        repeats = slices[1:] - slices[:-1]
        batch = repeat_interleave(repeats.tolist(), device=device)
        ptr = cumsum(repeats.to(device=device))
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v, device)
        return batch, ptr

    elif (isinstance(slices, Sequence) and not isinstance(slices, str)
          and isinstance(slices[0], Tensor)):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s, device)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr
    else:
        return None, None


def repeat_interleave(
    repeats: List[int],
    device: Optional[paddle.CPUPlace] = None,
) -> Tensor:
    outs = [paddle.full([n], i, device=device) for i, n in enumerate(repeats)]
    return paddle.concat(outs, axis=0)


def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], Tensor):
        repeats = paddle.stack(repeats, axis=0)
    else:
        repeats = paddle.to_tensor(repeats)
    return cumsum(repeats[:-1])
