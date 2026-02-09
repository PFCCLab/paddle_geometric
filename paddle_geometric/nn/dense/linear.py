import copy
import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.index import index2ptr
from paddle_geometric.nn import inits
from paddle_geometric.utils import index_sort


def is_uninitialized_parameter(x: Any) -> bool:
    # Check if the parameter is uninitialized
    # return isinstance(x, paddle.nn.UninitializedParameter)
    shape = x.shape
    for shape_i in shape:
        if shape_i <= 0:
            return True
    return False


def reset_weight_(weight: paddle.Tensor, in_channels: int,
                  initializer: Optional[str] = None) -> paddle.Tensor:
    if in_channels <= 0:
        pass
    elif initializer == "glorot":
        inits.glorot(weight)
    elif initializer == "uniform":
        bound = 1.0 / math.sqrt(in_channels)
        paddle.nn.init.uniform_(weight, -bound, bound)
    elif initializer == "kaiming_uniform":
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    elif initializer is None:
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")
    return weight


def reset_bias_(bias: Optional[paddle.Tensor], in_channels: int,
                initializer: Optional[str] = None) -> Optional[paddle.Tensor]:
    if bias is None or in_channels <= 0:
        pass
    elif initializer == "zeros":
        inits.zeros(bias)
    elif initializer is None:
        inits.uniform(in_channels, bias)
    else:
        raise RuntimeError(f"Bias initializer '{initializer}' not supported")
    return bias


class Linear(Layer):
    r"""Applies a linear transformation to the incoming data.

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    In contrast to :class:`torch.nn.Linear`, it supports lazy initialization
    and customizable weight and bias initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)

    Shapes:
        - **input:** features :math:`(*, F_{in})`
        - **output:** features :math:`(*, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: Optional[str] = None,
        bias_initializer: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # Initialize weight if in_channels is specified, otherwise leave it uninitialized
        if in_channels > 0:
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[in_channels, out_channels]))
        else:
            # self.weight = paddle.nn.UninitializedParameter()
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[0, 0]))
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)
            # raise NotImplementedError("paddle.nn.UninitializedParameter is not implemented yet.")

        # Initialize bias if specified
        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[out_channels]))
        else:
            self.bias = None

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(
            self.in_channels,
            self.out_channels,
            self.bias is not None,
            self.weight_initializer,
            self.bias_initializer,
        ).to(self.weight.place)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset_weight_(self.weight, self.in_channels, self.weight_initializer)
        reset_bias_(self.bias, self.in_channels, self.bias_initializer)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): The input features.
        """
        return paddle.nn.functional.linear(x=x, weight=self.weight,
                                           bias=self.bias)

    @paddle.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].shape[-1]
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=[self.in_channels, self.out_channels]))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, "_hook")

    def _state_dict_impl(
        self,
        destination=None,
        include_sublayers: bool = True,
        structured_name_prefix: str = "",
        include_non_persistable_buffer: bool = False,
        use_hook: bool = True,
        keep_vars: bool = True,
    ):
        if destination is None:
            destination = OrderedDict()
        if (is_uninitialized_parameter(self.weight) or keep_vars):
            destination[structured_name_prefix + "weight"] = self.weight
        else:
            destination[structured_name_prefix +
                        "weight"] = self.weight.detach()
        if self.bias is not None:
            if keep_vars:
                destination[structured_name_prefix + "bias"] = self.bias
            else:
                destination[structured_name_prefix +
                            "bias"] = self.bias.detach()
        return destination

    def set_state_dict(self, state_dict, *args, **kwargs):
        weight = state_dict.get("weight", None)
        if weight is not None and is_uninitialized_parameter(weight):
            self.in_channels = -1
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[0, 0]))
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)
        elif weight is not None and is_uninitialized_parameter(self.weight):
            self.in_channels = weight.size(0)
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=[self.in_channels, self.out_channels]))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')
        super().set_state_dict(state_dict, *args, **kwargs)

    load_state_dict = set_state_dict

    def __repr__(self) -> str:
        # Custom string representation for Linear layer
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')


class HeteroLinear(Layer):
    r"""Applies separate linear transformations to the incoming data according
    to types.

    For type :math:`\kappa`, it computes

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}.

    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        num_types (int): The number of types.
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`type_vec` is sorted. This avoids internal re-sorting of the
            data and can improve runtime and memory efficiency.
            (default: :obj:`False`)
    """
    _timing_cache: Dict[int, Tuple[float, float]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_types: int,
        is_sorted: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = num_types
        self.is_sorted = is_sorted
        self.kwargs = kwargs

        if self.in_channels == -1:
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[0, 0, 0]))
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)
        else:
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=[num_types, in_channels, out_channels]))

        if kwargs.get('bias', True):
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[num_types, out_channels]))
        else:
            self.add_parameter(name="bias", parameter=None)

        self._timing_cache: Dict[int, Tuple[float, float]] = {}

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset_weight_(self.weight, self.in_channels,
                      self.kwargs.get('weight_initializer', None))
        reset_bias_(self.bias, self.in_channels,
                    self.kwargs.get('bias_initializer', None))

    def forward_naive(self, x: Tensor, type_ptr: Tensor) -> Tensor:
        out = paddle.empty(shape=[x.shape[0], self.out_channels],
                           dtype=x.dtype)
        for i, (start, end) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
            out[start:end] = x[start:end] @ self.weight[i]
        return out

    def forward(self, x: paddle.Tensor,
                type_vec: paddle.Tensor) -> paddle.Tensor:
        """The forward pass.

        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        perm: Optional[paddle.Tensor] = None
        if not self.is_sorted and (type_vec[1:] < type_vec[:-1]).any():
            type_vec, perm = index_sort(type_vec, self.num_types)
            x = x[perm]
        type_ptr = index2ptr(type_vec, self.num_types)
        out = self.forward_naive(x, type_ptr)
        if self.bias is not None:
            out += self.bias[type_vec]
        if perm is not None:
            out_unsorted = paddle.empty_like(out)
            out_unsorted[perm] = out
            out = out_unsorted
        return out

    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].shape[-1]
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=[
                    self.num_types, self.in_channels, self.out_channels
                ]))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_types={self.num_types}, '
                f'bias={self.kwargs.get("bias", True)})')


class HeteroDictLinear(Layer):
    r"""Applies separate linear transformations to the incoming data
    dictionary.

    For key :math:`\kappa`, it computes

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}.

    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int or Dict[Any, int]): Size of each input sample. If
            passed an integer, :obj:`types` will be a mandatory argument.
            initialized lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        types (List[Any], optional): The keys of the input dictionary.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.Linear`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[Any, int]],
        out_channels: int,
        types: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(in_channels, dict):
            self.types = list(in_channels.keys())

            if any([i == -1 for i in in_channels.values()]):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

            if types is not None and set(self.types) != set(types):
                raise ValueError("The provided 'types' do not match with the "
                                 "keys in the 'in_channels' dictionary")

        else:
            if types is None:
                raise ValueError("Please provide a list of 'types' if passing "
                                 "'in_channels' as an integer")

            if in_channels == -1:
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

            self.types = types
            in_channels = {node_type: in_channels for node_type in types}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

        self.lins = paddle.nn.LayerDict({
            key:
            Linear(channels, self.out_channels, **kwargs)
            for key, channels in self.in_channels.items()
        })

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins.values():
            lin.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        r"""Forward pass.

        Args:
            x_dict (Dict[Any, Tensor]): A dictionary holding input
                features for each individual type.
        """
        out_dict = {}
        for key, lin in self.lins.items():
            if key in x_dict:
                out_dict[key] = lin(x_dict[key])
        return out_dict

    @paddle.no_grad()
    def initialize_parameters(self, module, input):
        for key, x in input[0].items():
            lin = self.lins[key]
            if is_uninitialized_parameter(lin.weight):
                self.lins[key].initialize_parameters(None, x)
                self.lins[key].reset_parameters()
        self._hook.remove()
        self.in_channels = {key: x.shape[-1] for key, x in input[0].items()}
        delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.kwargs.get("bias", True)})')
