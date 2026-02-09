import os

import numpy as np
import paddle


def _set_num_threads(int):
    os.environ['CPU_NUM'] = str(int)


def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type


class Embedding(paddle.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_idx = self._padding_idx


def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype=list(kwargs.values())[0])


paddle.Tensor.view = _Tensor_view


def _Tensor_reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


paddle.Tensor.reshape = _Tensor_reshape


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


# def _Tensor_split(self, split_size, dim=0):
#     if isinstance(split_size, int):
#         return paddle.split(self, self.shape[dim] // split_size, dim)
#     else:
#         return paddle.split(self, split_size, dim)

# paddle.Tensor.split = _Tensor_split


def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args,
                             **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


paddle.Tensor._max = _Tensor_max


def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)


def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args,
                             **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


paddle.Tensor._min = _Tensor_min


def paddle_min(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.minimum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.min(*args,
                                 **kwargs), paddle.argmin(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.min(*args,
                                 **kwargs), paddle.argmin(*args, **kwargs)
                return ret
        else:
            ret = paddle.min(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def _Tensor_add(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)
    return paddle.add(self, y)


paddle.Tensor.add = _Tensor_add


def _Tensor_sub(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)
    return paddle.subtract(self, y)


paddle.Tensor.sub = _Tensor_sub
paddle.Tensor.subtract = _Tensor_sub


def paddle_split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def paddle_max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args,
                                 **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args,
                                 **kwargs), paddle.argmax(*args, **kwargs)
                return ret
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def _Tensor_round(self, decimals=None):
    if decimals:
        x = paddle.abs(self) // (10**-decimals) * (10**-decimals)
        return paddle.where(self < 0, -x, x)
    return paddle.round(self)


paddle.Tensor.round = _Tensor_round


class GRUCell(paddle.nn.GRUCell):
    def forward(self, inputs, states=None):
        return super().forward(inputs, states)[0]


def _Tensor_index_copy_(self, dim, index, source):
    if dim == 0:
        return self.scatter_(index, source)

    shape = self.shape

    new_index = []
    for i in range(0, int(np.prod(shape[:dim]))):
        new_index.append(index + i * len(index))
    new_index = paddle.concat(new_index)
    new_self = self.reshape_([-1] + shape[dim + 1:])
    new_source = source.reshape([-1] + shape[dim + 1:])

    return new_self.scatter_(new_index, new_source).reshape_(shape)


paddle.Tensor.index_copy_ = _Tensor_index_copy_


def _Tensor_div(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    res = paddle.divide(self, y)

    if "rounding_mode" in kwargs:
        rounding_mode = kwargs["rounding_mode"]
        if rounding_mode == "trunc":
            res = paddle.trunc(res)
        elif rounding_mode == "floor":
            res = paddle.floor(res)

    return res


paddle.Tensor.div = _Tensor_div
paddle.Tensor.divide = _Tensor_div


def _Tensor_cpu(self):
    if self.is_sparse_coo():
        if not self.place.is_cpu_place():
            res = paddle.sparse.sparse_coo_tensor(
                indices=self.indices(), values=self.values(), shape=self.shape,
                dtype=self.dtype, stop_gradient=self.stop_gradient,
                place="cpu")
        else:
            res = self
    elif self.is_sparse_csr():
        if not self.place.is_cpu_place():
            res = paddle.sparse.sparse_csr_tensor(
                crows=self.crows(),
                cols=self.cols(),
                values=self.values(),
                shape=self.shape,
                dtype=self.dtype,
                place="cpu",
                stop_gradient=self.stop_gradient,
            )
        else:
            res = self
    else:
        res = self.to("cpu")
    return res


paddle.Tensor.cpu = _Tensor_cpu


def _Tensor_cuda(self, device_id=None, blocking=True):
    def get_res_place(device_id):

        if device_id is None:
            res_place = paddle.base.framework._current_expected_place()
            if not isinstance(res_place, paddle.core.CUDAPlace):
                res_place = paddle.core.CUDAPlace(0)
        elif isinstance(device_id, int):
            res_place = paddle.core.CUDAPlace(device_id)
        else:
            raise ValueError("device_id must be int|None")
        return res_place

    res_place = get_res_place(device_id=device_id)
    if self.place._equals(res_place):
        return self
    if self.is_sparse_coo():
        res = paddle.sparse.sparse_coo_tensor(indices=self.indices(),
                                              values=self.values(),
                                              shape=self.shape,
                                              dtype=self.dtype,
                                              stop_gradient=self.stop_gradient,
                                              place=res_place)
    elif self.is_sparse_csr():
        res = paddle.sparse.sparse_csr_tensor(
            crows=self.crows(),
            cols=self.cols(),
            values=self.values(),
            shape=self.shape,
            dtype=self.dtype,
            place=res_place,
            stop_gradient=self.stop_gradient,
        )
    else:
        res = self.to(device=res_place, blocking=blocking)
    return res


paddle.Tensor.cuda = _Tensor_cuda
