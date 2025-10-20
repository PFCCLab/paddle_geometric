import os
from typing import List

import paddle
import pytest

import paddle_geometric.typing
from paddle_geometric import Index, place2devicestr
from paddle_geometric.io import fs
from paddle_geometric.paddle_utils import *  # noqa
from paddle_geometric.testing import onlyCUDA, withCUDA
from paddle_geometric.typing import INDEX_DTYPES

DTYPES = [pytest.param(dtype, id=str(dtype)[6:]) for dtype in INDEX_DTYPES]


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_basic(dtype, device):
    kwargs = dict(dtype=dtype, place=device, dim_size=3)
    index = Index([0, 1, 1, 2], **kwargs)
    index.validate()
    assert isinstance(index, Index)

    assert str(index).startswith("Index([0, 1, 1, 2], ")
    assert "dim_size=3" in str(index)
    assert (f"device='{device}'" in str(index)) == index.place.is_cpu_place()
    assert (f"dtype={dtype}" in str(index)) == (dtype != paddle.int64)

    assert index.dtype == dtype
    assert index.device == device
    assert index.dim_size == 3
    assert not index.is_sorted

    out = index.as_tensor()
    assert not isinstance(out, Index)
    assert out.dtype == dtype
    assert place2devicestr(out.place) == device

    out = index * 1
    assert not isinstance(out, Index)
    assert out.dtype == dtype
    assert place2devicestr(out.place) == device


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_identity(dtype, device):
    kwargs = dict(dtype=dtype, place=device, dim_size=3, is_sorted=True)
    index = Index([0, 1, 1, 2], **kwargs)

    out = Index(index)
    assert not isinstance(out.as_tensor(), Index)
    assert out.data_ptr() == index.data_ptr()
    assert out.dtype == index.dtype
    assert out.place == index.place
    assert out.dim_size == index.dim_size
    assert out.is_sorted == index.is_sorted

    out = Index(index, dim_size=4, is_sorted=False)
    assert out.dim_size == 4
    assert out.is_sorted == index.is_sorted


def test_validate():
    with pytest.raises(ValueError, match="unsupported data type"):
        Index([0.0, 1.0])
    with pytest.raises(ValueError, match="needs to be one-dimensional"):
        Index([[0], [1]])
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        Index(paddle.to_tensor(data=[0, 1]), "int64")
    with pytest.raises(TypeError, match="invalid keyword arguments"):
        Index(paddle.to_tensor(data=[0, 1]), dtype="int64")
    with pytest.raises(ValueError, match="contains negative indices"):
        Index([-1, 0]).validate()
    with pytest.raises(ValueError, match="than its registered size"):
        Index([0, 10], dim_size=2).validate()
    with pytest.raises(ValueError, match="not sorted"):
        Index([1, 0], is_sorted=True).validate()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_cache_(dtype, device):
    kwargs = dict(dtype=dtype, place=device)
    index = Index([0, 1, 1, 2], is_sorted=True, **kwargs)
    index.validate().fill_cache_()
    assert index.dim_size == 3
    assert index._indptr.dtype == dtype
    assert index._indptr.equal(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype)).all()
    index = Index([1, 0, 2, 1], **kwargs)
    index.validate().fill_cache_()
    assert index.dim_size == 3
    assert index._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_dim_resize(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], is_sorted=True, **kwargs).fill_cache_()

    assert index.dim_size == 3
    assert index._indptr.equal(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype)).all()

    out = index.dim_resize_(4)
    assert out.dim_size == 4
    assert out._indptr.equal(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device,
                         dtype=dtype)).all()

    out = index.dim_resize_(3)
    assert out.dim_size == 3
    assert out._indptr.equal(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype)).all()

    out = index.dim_resize_(None)
    assert out.dim_size is None
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_clone(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], is_sorted=True, dim_size=3, **kwargs)

    out = index.clone()
    assert isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device
    assert out.dim_size == 3
    assert out.is_sorted

    out = paddle.clone(x=index)
    assert isinstance(out, Index)
    assert out.dtype == dtype
    assert out.device == device
    assert out.dim_size == 3
    assert out.is_sorted


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_function(dtype, device):
    kwargs = dict(dtype=dtype)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()

    index = index.to(device)
    assert isinstance(index, Index)
    assert place2devicestr(index.place) == device
    assert index._indptr.dtype == dtype
    assert place2devicestr(index._indptr.place) == device

    out = index.cpu()
    assert isinstance(out, Index)
    assert place2devicestr(out.place) == "cpu"

    out = index.to("int32")
    assert out.dtype == paddle.int32
    if paddle_geometric.typing.WITH_PP32:
        assert isinstance(out, Index)
        assert out._indptr.dtype == paddle.int32
    else:
        assert not isinstance(out, Index)

    out = index.to("float32")
    assert not isinstance(out, Index)
    assert out.dtype == paddle.float32

    out = index.long()
    assert isinstance(out, Index)
    assert out.dtype == paddle.int64

    out = index.int()
    assert out.dtype == paddle.int32
    if paddle_geometric.typing.WITH_PP32:
        assert isinstance(out, Index)
    else:
        assert not isinstance(out, Index)


@onlyCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_cuda(dtype):
    kwargs = dict(dtype=dtype)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    assert index.place.is_gpu_place()

    out = index.cuda()
    assert isinstance(out, Index)
    assert out.place.is_gpu_place()

    out = out.cpu()
    assert isinstance(out, Index)
    assert out.place.is_cpu_place()


@pytest.mark.skip(reason='PaddlePaddle not support share_memory')
@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_share_memory(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()
    out = index.share_memory_()
    assert isinstance(out, Index)
    assert out.is_shared()
    assert out._data.is_shared()
    assert out._indptr.is_shared()
    assert out.data_ptr() == index.data_ptr()


@pytest.mark.skip(reason='PaddlePaddle not support pin_memory')
@onlyCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_pin_memory(dtype):
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, dtype=dtype)
    assert "pinned" not in str(index.place)
    out = index.pin_memory()
    assert "pinned" in str(out.place)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_contiguous(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    assert index.is_contiguous()
    out = index.contiguous()
    assert isinstance(out, Index)
    assert out.is_contiguous()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sort(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([1, 0, 2, 1], dim_size=3, **kwargs)

    index, _ = index.sort()
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], dtype=dtype,
                                              place=device)).item()
    assert index.dim_size == 3
    assert index.is_sorted

    out, perm = index.sort()
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert perm.equal_all(
        y=paddle.to_tensor(data=[0, 1, 2, 3], place=device)).item()
    assert out.dim_size == 3

    index, _ = index.sort(descending=True)
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[2, 1, 1, 0], dtype=dtype,
                                              place=device)).item()
    assert index.dim_size == 3
    assert not index.is_sorted


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_stable(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([1, 0, 2, 1], dim_size=3, **kwargs)

    index, perm = index.sort(stable=True)
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], dtype=dtype,
                                              place=device)).item()
    assert perm.equal_all(
        y=paddle.to_tensor(data=[1, 0, 3, 2], place=device)).item()
    assert index.dim_size == 3
    assert index.is_sorted

    out, perm = index.sort(stable=True)
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert perm.equal_all(
        y=paddle.to_tensor(data=[0, 1, 2, 3], place=device)).item()
    assert out.dim_size == 3

    index, perm = index.sort(descending=True, stable=True)
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[2, 1, 1, 0], dtype=dtype,
                                              place=device)).item()
    assert perm.equal_all(
        y=paddle.to_tensor(data=[3, 1, 2, 0], place=device)).item()
    assert index.dim_size == 3
    assert not index.is_sorted


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_cat(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index1 = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index2 = Index([1, 2, 2, 3], dim_size=4, is_sorted=True, **kwargs)
    index3 = Index([1, 2, 2, 3], **kwargs)

    out = paddle.concat(x=[index1, index2])
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2, 1, 2, 2, 3],
                                            place=device, dtype=dtype)).item()
    assert tuple(out.shape) == (8, )
    assert isinstance(out, Index)
    assert out.dim_size == 4
    assert not out.is_sorted
    assert out._cat_metadata.nnz == [4, 4]
    assert out._cat_metadata.dim_size == [3, 4]
    assert out._cat_metadata.is_sorted == [True, True]
    out = paddle.concat(x=[index1, index2, index3])
    assert tuple(out.shape) == (12, )
    assert isinstance(out, Index)
    assert out.dim_size is None
    assert not out.is_sorted
    out = paddle.concat(x=[index1, index2.as_tensor()])
    assert tuple(out.shape) == (8, )
    assert not isinstance(out, Index)
    inplace = paddle.empty(shape=[8], dtype=dtype)
    out = paddle.assign(paddle.concat(x=[index1, index2]), output=inplace)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2, 1, 2, 2, 3],
                                            place=device, dtype=dtype)).item()
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, Index)
    assert not isinstance(inplace, Index)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_flip(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    out = index.flip(axis=0)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[2, 1, 1, 0], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 3
    assert not out.is_sorted


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_index_select(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    i = paddle.to_tensor(data=[1, 3], place=device)

    out = index.index_select(axis=0, index=i)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert isinstance(out, Index)
    assert out.dim_size == 3
    assert not out.is_sorted
    inplace = paddle.empty(shape=[2], dtype=dtype)
    out = paddle.assign(paddle.index_select(x=index, axis=0, index=i),
                        output=inplace)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, Index)
    assert not isinstance(inplace, Index)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_narrow(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)

    out = index.narrow(0, start=1, length=2)
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 1], place=device, dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_getitem(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)

    out = index[:]
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index[paddle.to_tensor(data=[False, True, False, True],
                                 place=device)]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index[paddle.to_tensor(data=[1, 3], place=device)]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert out.dim_size == 3
    assert not out.is_sorted

    out = index[1:3]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 1], place=device, dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index[...]
    assert isinstance(out, Index)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index[..., 1:3]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 1], place=device, dtype=dtype)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index[None, 1:3]
    assert not isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[1, 1]], place=device, dtype=dtype)).item()

    out = index[1:3, None]
    assert not isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[1], [1]], place=device, dtype=dtype)).item()

    out = index[0]
    assert not isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=0, place=device, dtype=dtype)).item()

    tmp = paddle.randn(shape=[3])
    out = tmp[index]
    assert not isinstance(out, Index)
    assert out.equal_all(y=tmp[index.as_tensor()]).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_add(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)

    out = paddle.add(x=index, y=2, alpha=2)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[4, 5, 5, 6], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 7
    assert out.is_sorted

    out = index + paddle.to_tensor(data=[2], dtype=dtype, place=device)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[2, 3, 3, 4], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 5
    assert out.is_sorted

    out = paddle.to_tensor(data=[2], dtype=dtype, place=device) + index
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[2, 3, 3, 4], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 5
    assert out.is_sorted

    out = index.add(index)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 2, 2, 4], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 6
    assert not out.is_sorted

    index += 2
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[2, 3, 3, 4], dtype=dtype,
                                              place=device)).item()
    assert index.dim_size == 5
    assert index.is_sorted

    # with pytest.raises(RuntimeError, match="DID NOT RAISE"):
    #     index += 2.5


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sub(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([4, 5, 5, 6], dim_size=7, is_sorted=True, **kwargs)

    out = paddle.subtract(x=index, y=2, alpha=2)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 3
    assert out.is_sorted

    out = index - paddle.to_tensor(data=[2], dtype=dtype, place=device)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[2, 3, 3, 4], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size == 5
    assert out.is_sorted

    out = paddle.to_tensor(data=[6], dtype=dtype, place=device) - index
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[2, 1, 1, 0], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size is None
    assert not out.is_sorted

    out = index.sub(index)
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 0, 0, 0], dtype=dtype,
                                            place=device)).item()
    assert out.dim_size is None
    assert not out.is_sorted

    index -= 2
    assert isinstance(index, Index)
    assert index.equal_all(y=paddle.to_tensor(data=[2, 3, 3, 4], dtype=dtype,
                                              place=device)).item()
    assert index.dim_size == 5
    assert not out.is_sorted
    # with pytest.raises(RuntimeError, match="can't be cast"):
    #     index -= 2.5


@pytest.mark.skip(reason="Index.tolist() not support")
def test_to_list():
    index = Index([0, 1, 1, 2])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        index.tolist()


@pytest.mark.skip(reason="Index.numpy() not support")
def test_numpy():
    index = Index([0, 1, 1, 2])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        index.numpy()


@pytest.mark.skip(reason="Index.save() not support")
@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_save_and_load(dtype, device, tmp_path):
    kwargs = dict(dtype=dtype, device=device)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()

    path = os.path.join(tmp_path, "edge_index.pt")
    paddle.save(obj=index, path=path)

    out = fs.torch_load(path)
    assert isinstance(out, Index)
    assert out.equal_all(y=index).item()
    assert out.dim_size == 3
    assert out.is_sorted
    assert out._indptr.equal(index._indptr)


def _collate_fn(indices: List[Index]) -> List[Index]:
    return indices


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_workers", [
    0,
])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_data_loader(dtype, num_workers, pin_memory):
    kwargs = dict(dtype=dtype)
    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True, **kwargs)
    index.fill_cache_()

    index = index.to_dict()

    loader = paddle.io.DataLoader(dataset=[index] * 4, batch_size=2,
                                  num_workers=num_workers,
                                  collate_fn=_collate_fn, drop_last=True)
    assert len(loader) == 2
    for batch in loader:
        assert isinstance(batch, list)
        assert len(batch) == 2
        for index in batch:
            index = Index.from_dict(index)
            assert isinstance(index, Index)
            assert index.dtype == dtype
