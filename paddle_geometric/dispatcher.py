import functools
from typing import Any, Callable, Dict, Type

import paddle
from paddle import Tensor

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}
# General Registry Master Table: A table recording the registry for each type of operation
# {
#     "concat": {
#         "EdgeIndex": concat_edgeindex,
#         "Index": concat_index,
#     },
#     "add": {
#         "EdgeIndex": add_edgeindex,
#         "Index": add_index,
#     },
# }
REGISTRIES: Dict[str, Dict[Type, Any]] = {}


def register_for(op_name: str):
    """Generic registration decorator factory.
    Usage example:
        @register_for("concat")(MyClass)
        def concat_myclass(...): ...
    """
    def inner(cls: Type):
        registry = REGISTRIES.setdefault(op_name, {})

        def decorator(func: Callable):
            registry[cls] = func
            return func

        return decorator

    return inner


def set_registries(op_name):
    REGISTRIES.setdefault(op_name, {})


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __array_function__.

    __array_function__ implementations should be called in order on the return
    values from this function.
    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        if arg_type not in overloaded_types:
            try:
                arg_type.__array_function__  # noqa
            except AttributeError:
                continue

            overloaded_types.append(arg_type)

            # if array_function is not Tensor.__array_function__:
            index = len(overloaded_args)
            for i, old_arg in enumerate(overloaded_args):
                if issubclass(arg_type, type(old_arg)):
                    index = i
                    break
            overloaded_args.insert(index, arg)

    return overloaded_types, overloaded_args


def full_name(obj):
    return f'{obj.__module__}.{obj.__qualname__}'


def attempt_augmented_error_message(error, append_message):
    """Attempt to recreate an error with an appended message."""
    try:
        return type(error)(error.args[0] + append_message, *error.args[1:])
    except Exception:
        return error


def try_array_function_override(func, relevant_arguments, args, kwargs):
    # TODO: consider simplifying the interface, to only require either `types`
    # (by calling __array_function__ a classmethod) or `overloaded_args` (by
    # dropping `types` from the signature of __array_function__)
    types, overloaded_args = get_overloaded_types_and_args(relevant_arguments)
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __array_function__ implementation might reasonably need to
        # check all argument types.
        try:
            result = overloaded_arg.__array_function__(func, types, args,
                                                       kwargs)
        except Exception as error:
            # Ensure the type of the overloaded argument ends up in the
            # traceback
            message = (" [while calling {!r} implementation of {!r}]".format(
                full_name(type(overloaded_arg)), full_name(func)))
            new_error = attempt_augmented_error_message(error, message)
            # Would probably need to use six to do this sanely on Python 2:
            # https://stackoverflow.com/questions/9157210/
            raise new_error.with_traceback(error.__traceback__) from None

        if result is not NotImplemented:
            return True, result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'.format(
                        func, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Wrap a function for dispatch with the __array_function__ protocol."""
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            relevant_arguments = dispatcher(*args, **kwargs)
            success, value = try_array_function_override(
                new_func, relevant_arguments, args, kwargs)
            if success:
                return value
            return func(*args, **kwargs)

        return new_func

    return decorator


class BaseTensorSubclass(paddle.Tensor):
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if func in [paddle.concat]:
            if not all(issubclass(t, Tensor) for t in types):
                return NotImplemented
        else:
            if not any(issubclass(t, Tensor) for t in types):
                return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def _flip_dispatcher(x, axis, **kwargs):
    return (x, )


def _binary_dispatcher(x, y, **kwargs):
    return (x, y)


def _list_dispatcher(x, axis=None, name=None):
    yield from x


def _index_select_dispatcher(x, index, axis=0, name=None, *, out=None):
    return (x, )


def _narrow_dispatcher(input, dim, start, length):
    return (input, )


default_concat = paddle.concat
paddle.concat = array_function_dispatch(_list_dispatcher)(paddle.concat)
default_add = paddle.add
paddle.add = array_function_dispatch(_binary_dispatcher)(paddle.add)
default_subtract = paddle.subtract
paddle.subtract = array_function_dispatch(_binary_dispatcher)(paddle.subtract)
default_flip = paddle.flip
paddle.flip = array_function_dispatch(_flip_dispatcher)(paddle.flip)
default_index_select = paddle.index_select
paddle.index_select = array_function_dispatch(_index_select_dispatcher)(
    paddle.index_select)
def _narrow(input, dim, start, length):
    """Implements torch.narrow functionality using paddle.slice."""
    end = start + length
    return paddle.slice(input, axes=[dim], starts=[start], ends=[end])

default_narrow = _narrow
paddle.narrow = array_function_dispatch(_narrow_dispatcher)(_narrow)
default_unbind = paddle.unbind
paddle.unbind = array_function_dispatch(_list_dispatcher)(paddle.unbind)
default_matmul = paddle.matmul
paddle.matmul = array_function_dispatch(_binary_dispatcher)(paddle.matmul)
default_mm = paddle.mm
paddle.mm = array_function_dispatch(_binary_dispatcher)(paddle.mm)


@implements(paddle.concat)
def concat(x, axis=0, name=None, *, out=None):
    op_name = 'concat'
    set_registries(op_name)
    types = [type(x_) for x_ in x]
    for typ in types:
        if typ in REGISTRIES[op_name]:
            return REGISTRIES[op_name][typ](x, axis=axis, name=name)
    return default_concat(x, axis=axis, name=name, out=out)


@implements(paddle.add)
def add(x, y, name=None, *, alpha=1, out=None):
    op_name = 'add'
    set_registries(op_name)
    types = [type(x), type(y)]
    for typ in types:
        if typ in REGISTRIES[op_name]:
            return REGISTRIES[op_name][typ](x, y, name=name, alpha=alpha,
                                            out=out)
    return default_add(x, y, name=name, alpha=alpha, out=out)


@implements(paddle.subtract)
def subtract(x, y, name=None, *, alpha=1, out=None):
    op_name = 'subtract'
    set_registries(op_name)
    types = [type(x), type(y)]
    for typ in types:
        if typ in REGISTRIES[op_name]:
            return REGISTRIES[op_name][typ](x, y, name=name, alpha=alpha,
                                            out=out)
    return default_subtract(x, y, name=name, alpha=alpha, out=out)


@implements(paddle.flip)
def flip(x, axis, name=None):
    op_name = 'flip'
    set_registries(op_name)
    typ = type(x)
    if typ in REGISTRIES[op_name]:
        return REGISTRIES[op_name][typ](x, axis, name=name)
    return default_flip(x, axis, name)


@implements(paddle.index_select)
def index_select(x, index, axis=0, name=None, *, out=None):
    op_name = 'index_select'
    set_registries(op_name)
    typ = type(x)
    if typ in REGISTRIES[op_name]:
        return REGISTRIES[op_name][typ](x, index, axis, name=name, out=out)
    return default_index_select(x, index, axis, name=name, out=out)


@implements(paddle.narrow)
def narrow(input, dim, start, length):
    op_name = 'narrow'
    set_registries(op_name)
    typ = type(input)
    if typ in REGISTRIES[op_name]:
        return REGISTRIES[op_name][typ](input, dim, start, length)
    return default_narrow(input, dim, start, length)


@implements(paddle.unbind)
def unbind(input, axis=0):
    op_name = 'unbind'
    set_registries(op_name)
    typ = type(input)
    if typ in REGISTRIES[op_name]:
        return REGISTRIES[op_name][typ](input, axis)
    return default_unbind(input, axis)


@implements(paddle.matmul)
def matmul(x, y, transpose_x=False, transpose_y=False, name=None, *, out=None):
    op_name = 'matmul'
    set_registries(op_name)
    types = [type(x), type(y)]
    for typ in types:
        if typ in REGISTRIES[op_name]:
            return REGISTRIES[op_name][typ](x, y, transpose_x=transpose_x,
                                            transpose_y=transpose_y, name=name,
                                            out=out)
    return default_matmul(x, y, transpose_x=transpose_x,
                          transpose_y=transpose_y, name=name, out=out)


@implements(paddle.mm)
def mm(input, mat2, name=None):
    op_name = 'mm'
    set_registries(op_name)
    types = [type(input), type(mat2)]
    for typ in types:
        if typ in REGISTRIES[op_name]:
            return REGISTRIES[op_name][typ](input, mat2, name=name)
    return default_mm(input, mat2, name=name)
