import copy
import os.path as osp
from typing import Optional, Tuple, Union

import pytest
import paddle
from paddle import Tensor
from paddle.nn import Linear

import paddle_geometric.typing
from paddle_geometric import EdgeIndex
from paddle_geometric.nn import GATConv, MessagePassing, aggr
from paddle_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from paddle_geometric.utils import (
    add_self_loops,
    scatter,
    spmm,
    to_paddle_csc_tensor,
)



def skip_if_cuda_error():
    try:
        import paddle
        paddle.set_device('cpu')
        return False
    except Exception as e:
        if 'cuda' in str(e).lower() or 'CUDA' in str(e):
            pytest.skip(f"CUDA 兼容性问题: {e}")
        return False


@pytest.fixture(autouse=True)
def setup_cpu_mode():
    """自动设置 CPU 模式"""
    skip_if_cuda_error()


class MyConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggr: str = 'add'):
        super().__init__(aggr=aggr)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels)
        self.lin_r = Linear(in_channels[1], out_channels)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        return x_j if edge_weight is None else edge_weight.reshape([-1, 1]) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class MyConvWithSelfLoops(MessagePassing):
    def __init__(self, aggr: str = 'add'):
        super().__init__(aggr=aggr)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index)

        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x)


def test_my_conv_basic():
    x1 = paddle.randn([4, 8])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = MyConv(8, 32)
    out = conv(x1, edge_index, value)
    assert out.shape == [4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, (4, 4)), out, atol=1e-6)
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    # Bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, value, size=(4, 2))

    conv = MyConv((8, 16), 32)
    out1 = conv((x1, x2), edge_index, value)
    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out1.shape == [2, 32]
    assert out2.shape == [2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)
    assert paddle.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-6)
    assert paddle.allclose(conv((x1, None), adj1.t()), out2, atol=1e-6)


def test_my_conv_save(tmp_path):
    conv = MyConv(8, 32)
    assert conv._jinja_propagate is not None
    assert conv.__class__._jinja_propagate is not None
    assert conv._orig_propagate is not None
    assert conv.__class__._orig_propagate is not None

    path = osp.join(tmp_path, 'model.pdparams')
    paddle.save(conv.state_dict(), path)


def test_my_conv_edge_index():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_index = EdgeIndex(edge_index, sparse_size=(4, 4), sort_order='col')

    conv = MyConv(8, 32)

    out = conv(x, edge_index)
    assert out.shape == [4, 32]


class MyCommentedConv(MessagePassing):
    r"""This layer calls `self.propagate()` internally."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # `self.propagate()` is used here to propagate messages.
        return self.propagate(edge_index, x=x)


def test_my_commented_conv():
    # Check that `self.propagate` occurences in comments are correctly ignored.
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyCommentedConv()
    conv(x, edge_index)


class MyKwargsConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(x=x, edge_index=edge_index)


def test_my_kwargs_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyKwargsConv()
    conv(x, edge_index)


def test_my_conv_out_of_bounds():
    x = paddle.randn([3, 8])
    value = paddle.randn([4])

    conv = MyConv(8, 32)

    with pytest.raises(IndexError, match="valid indices"):
        edge_index = paddle.to_tensor([[-1, 1, 2, 2], [0, 0, 1, 1]])
        conv(x, edge_index, value)

    with pytest.raises(IndexError, match="valid indices"):
        edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        conv(x, edge_index, value)


@pytest.mark.parametrize('aggr', ['add', 'sum', 'mean', 'min', 'max', 'mul'])
def test_my_conv_aggr(aggr):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = paddle.randn([edge_index.shape[1]])

    conv = MyConv(8, 32, aggr=aggr)
    out = conv(x, edge_index, edge_weight)
    assert out.shape == [4, 32]


def test_my_static_graph_conv():
    x1 = paddle.randn([3, 4, 8])
    x2 = paddle.randn([3, 2, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn([edge_index.shape[1]])

    conv = MyConv(8, 32)
    out = conv(x1, edge_index, value)
    assert out.shape == [3, 4, 32]
    assert paddle.allclose(conv(x1, edge_index, value, (4, 4)), out)

    conv = MyConv((8, 16), 32)
    out1 = conv((x1, x2), edge_index, value)
    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out1.shape == [3, 2, 32]
    assert out2.shape == [3, 2, 32]
    assert paddle.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)


class MyMultipleAggrConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr=['add', 'mean', 'max'], **kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x)


@pytest.mark.parametrize('multi_aggr_tuple', [
    (dict(mode='cat'), 3),
    (dict(mode='proj', mode_kwargs=dict(in_channels=16, out_channels=16)), 1)
])
def test_my_multiple_aggr_conv(multi_aggr_tuple):
    # The 'cat' combine mode will expand the output dimensions by
    # the number of aggregators which is 3 here, while the 'proj'
    # mode keeps output dimensions unchanged.
    aggr_kwargs, expand = multi_aggr_tuple
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = MyMultipleAggrConv(aggr_kwargs=aggr_kwargs)
    out = conv(x, edge_index)
    assert out.shape == [4, 16 * expand]
    # Note: Sparse tensor results may have NaN values for 'proj' mode due to max aggregation
    # on nodes with no neighbors. We use equal_nan=True to handle this.
    assert paddle.allclose(conv(x, adj1.t()), out, equal_nan=True)


def test_my_multiple_aggr_conv_jit():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyMultipleAggrConv()
    out = conv(x, edge_index)


def test_copy():
    conv = MyConv(8, 32)
    conv2 = copy.copy(conv)

    assert conv != conv2
    # Note: Paddle's equal returns a tensor, not a boolean
    assert paddle.equal(conv.lin_l.weight, conv2.lin_l.weight).all()
    assert paddle.equal(conv.lin_r.weight, conv2.lin_r.weight).all()
    # Note: Paddle doesn't have the same data_ptr concept as PyTorch


class MyEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # edge_updater_type: (x: Tensor)
        edge_attr = self.edge_updater(edge_index, x=x)

        # propagate_type: (edge_attr: Tensor)
        return self.propagate(edge_index, edge_attr=edge_attr,
                              size=(x.shape[0], x.shape[0]))

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        return x_j - x_i

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr


def test_my_edge_conv():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    row, col = edge_index[0], edge_index[1]
    expected = scatter(x[row] - x[col], col, dim=0, dim_size=4, reduce='sum')

    conv = MyEdgeConv()
    out = conv(x, edge_index)
    assert out.shape == [4, 16]
    assert paddle.allclose(out, expected)
    assert paddle.allclose(conv(x, adj1.t()), out)


def test_my_edge_conv_jit():
    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyEdgeConv()
    out = conv(x, edge_index)


num_pre_hook_calls = 0
num_hook_calls = 0


def test_message_passing_hooks():
    conv = MyConv(8, 32)

    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = paddle.randn([edge_index.shape[1]])
    adj = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    def pre_hook(module, inputs):
        assert module == conv
        global num_pre_hook_calls
        num_pre_hook_calls += 1
        return inputs

    def hook(module, inputs, output):
        assert module == conv
        global num_hook_calls
        num_hook_calls += 1
        return output

    handle1 = conv.register_propagate_forward_pre_hook(pre_hook)
    assert len(conv._propagate_forward_pre_hooks) == 1
    handle2 = conv.register_propagate_forward_hook(hook)
    assert len(conv._propagate_forward_hooks) == 1

    handle3 = conv.register_message_forward_pre_hook(pre_hook)
    assert len(conv._message_forward_pre_hooks) == 1
    handle4 = conv.register_message_forward_hook(hook)
    assert len(conv._message_forward_hooks) == 1

    handle5 = conv.register_aggregate_forward_pre_hook(pre_hook)
    assert len(conv._aggregate_forward_pre_hooks) == 1
    handle6 = conv.register_aggregate_forward_hook(hook)
    assert len(conv._aggregate_forward_hooks) == 1

    handle7 = conv.register_message_and_aggregate_forward_pre_hook(pre_hook)
    assert len(conv._message_and_aggregate_forward_pre_hooks) == 1
    handle8 = conv.register_message_and_aggregate_forward_hook(hook)
    assert len(conv._message_and_aggregate_forward_hooks) == 1

    out1 = conv(x, edge_index, value)
    assert num_pre_hook_calls == 3
    assert num_hook_calls == 3
    out2 = conv(x, adj.t())
    assert num_pre_hook_calls == 5
    assert num_hook_calls == 5
    assert paddle.allclose(out1, out2, atol=1e-6)

    handle1.remove()
    assert len(conv._propagate_forward_pre_hooks) == 0
    handle2.remove()
    assert len(conv._propagate_forward_hooks) == 0

    handle3.remove()
    assert len(conv._message_forward_pre_hooks) == 0
    handle4.remove()
    assert len(conv._message_forward_hooks) == 0

    handle5.remove()
    assert len(conv._aggregate_forward_pre_hooks) == 0
    handle6.remove()
    assert len(conv._aggregate_forward_hooks) == 0

    handle7.remove()
    assert len(conv._message_and_aggregate_forward_pre_hooks) == 0
    handle8.remove()
    assert len(conv._message_and_aggregate_forward_hooks) == 0

    conv = MyEdgeConv()

    handle1 = conv.register_edge_update_forward_pre_hook(pre_hook)
    assert len(conv._edge_update_forward_pre_hooks) == 1
    handle2 = conv.register_edge_update_forward_hook(hook)
    assert len(conv._edge_update_forward_hooks) == 1

    out1 = conv(x, edge_index)
    assert num_pre_hook_calls == 6
    assert num_hook_calls == 6
    out2 = conv(x, adj.t())
    assert num_pre_hook_calls == 7
    assert num_hook_calls == 7
    assert paddle.allclose(out1, out2, atol=1e-6)

    handle1.remove()
    assert len(conv._propagate_forward_pre_hooks) == 0
    handle2.remove()
    assert len(conv._propagate_forward_hooks) == 0


def test_modified_message_passing_hook():
    conv = MyConv(8, 32)

    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = paddle.randn([edge_index.shape[1]])

    out1 = conv(x, edge_index, edge_weight)

    def hook(module, inputs, output):
        assert len(inputs) == 1
        assert len(inputs[-1]) == 2
        assert 'x_j' in inputs[-1]
        assert 'edge_weight' in inputs[-1]
        return output + 1.

    conv.register_message_forward_hook(hook)

    out2 = conv(x, edge_index, edge_weight)
    assert not paddle.allclose(out1, out2, atol=1e-6)


class MyDefaultArgConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    # propagate_type: (x: Tensor)
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j, zeros: bool = True):
        return x_j * 0 if zeros else x_j


def test_my_default_arg_conv():
    x = paddle.randn([4, 1])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = MyDefaultArgConv()
    assert conv(x, edge_index).reshape([-1]).tolist() == [0, 0, 0, 0]
    assert conv(x, adj1.t()).reshape([-1]).tolist() == [0, 0, 0, 0]


class MyMultipleOutputConv(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tuple[Tensor, Tensor]:
        return x_j, x_j

    def aggregate(self, inputs: Tuple[Tensor, Tensor],
                  index: Tensor) -> Tuple[Tensor, Tensor]:
        return (scatter(inputs[0], index, dim=0, reduce='sum'),
                scatter(inputs[0], index, dim=0, reduce='mean'))

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


def test_tuple_output():
    conv = MyMultipleOutputConv()

    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv(x, edge_index)
    assert isinstance(out1, tuple) and len(out1) == 2


class MyExplainConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index, x=x)


def test_explain_message():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyExplainConv()
    conv.explain = True
    assert conv.propagate.__module__.endswith('message_passing')

    with pytest.raises(ValueError, match="pre-defined 'edge_mask'"):
        conv(x, edge_index)

    conv._edge_mask = paddle.to_tensor([0.0, 0.0, 0.0, 0.0])
    conv._apply_sigmoid = False
    assert conv(x, edge_index).abs().sum() == 0.

    conv._edge_mask = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
    conv._apply_sigmoid = False
    out1 = conv(x, edge_index)

    conv.explain = False
    assert conv.propagate.__module__.endswith('MyExplainConv_propagate')
    out2 = conv(x, edge_index)
    assert paddle.allclose(out1, out2)


class MyAggregatorConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x)


@pytest.mark.parametrize('aggr_module', [
    aggr.MeanAggregation(),
    aggr.SumAggregation(),
    aggr.MaxAggregation(),
    aggr.SoftmaxAggregation(),
    aggr.PowerMeanAggregation(),
    aggr.MultiAggregation(['mean', 'max'])
])
def test_message_passing_with_aggr_module(aggr_module):
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index[0], edge_index[1]
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = MyAggregatorConv(aggr=aggr_module)
    assert isinstance(conv.aggr_module, aggr.Aggregation)
    out = conv(x, edge_index)
    assert out.shape[0] == 4 and out.shape[1] in {8, 16}
    assert paddle.allclose(conv(x, adj1.t()), out)


class MyOptionalEdgeAttrConv(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        return x_j if edge_attr is None else x_j * edge_attr.reshape([-1, 1])


def test_my_optional_edge_attr_conv():
    conv = MyOptionalEdgeAttrConv()

    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    out = conv(x, edge_index)
    assert out.shape == [4, 8]