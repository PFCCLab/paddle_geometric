import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.nn.inits import kaiming_uniform, uniform
from paddle_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor


class Linear(Layer):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = self.create_parameter(
            shape=[groups, in_channels // groups, out_channels // groups])

        if bias:
            self.bias = self.create_parameter(shape=[out_channels])
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.shape[1], a=math.sqrt(5))
        uniform(self.weight.shape[1], self.bias)

    def forward(self, src):
        if self.groups > 1:
            size = list(src.shape[:-1])
            src = src.reshape(
                [-1, self.groups, self.in_channels // self.groups])
            src = src.transpose([1, 0, 2])
            out = paddle.matmul(src, self.weight)
            out = out.transpose([1, 0, 2])
            out = out.reshape(size + [self.out_channels])
        else:
            out = paddle.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, groups={self.groups})')


def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    src_max = paddle.clip(paddle.max(src, axis=dim, keepdim=True), min=0.)
    out = paddle.exp(src - src_max)
    out = out / (paddle.sum(out, axis=dim, keepdim=True) +
                 paddle.exp(margin - src_max))
    return out


class Attention(Layer):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
        assert query.ndim == key.ndim == value.ndim >= 2
        assert query.shape[-1] == key.shape[-1]
        assert key.shape[-2] == value.shape[-2]

        perm = list(range(key.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        score = paddle.matmul(query, key.transpose(perm))
        score = score / math.sqrt(key.shape[-1])
        score = restricted_softmax(score, dim=-1)
        score = F.dropout(score, p=self.dropout, training=self.training)

        return paddle.matmul(score, value)

    def __repr__(self) -> str:  # pragma: no cover
        return f'{self.__class__.__name__}(dropout={self.dropout})'


class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0,
                 bias=True):
        super().__init__(dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.heads) % min(groups, self.heads) == 0

        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):
        assert query.ndim == key.ndim == value.ndim >= 2
        assert query.shape[-1] == key.shape[-1] == value.shape[-1]
        assert key.shape[-2] == value.shape[-2]

        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        size = list(query.shape[:-2])
        out_channels_per_head = self.out_channels // self.heads

        query_size = size + [query.shape[-2], self.heads, out_channels_per_head]
        query = query.reshape(query_size)
        perm = list(range(query.ndim))
        perm[-3], perm[-2] = perm[-2], perm[-3]
        query = query.transpose(perm)

        key_size = size + [key.shape[-2], self.heads, out_channels_per_head]
        key = key.reshape(key_size)
        perm = list(range(key.ndim))
        perm[-3], perm[-2] = perm[-2], perm[-3]
        key = key.transpose(perm)

        value_size = size + [value.shape[-2], self.heads,
                             out_channels_per_head]
        value = value.reshape(value_size)
        perm = list(range(value.ndim))
        perm[-3], perm[-2] = perm[-2], perm[-3]
        value = value.transpose(perm)

        out = self.compute_attention(query, key, value)

        perm = list(range(out.ndim))
        perm[-3], perm[-2] = perm[-2], perm[-3]
        out = out.transpose(perm)
        out = out.reshape(size + [query.shape[-2], self.out_channels])

        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'groups={self.groups}, dropout={self.dropout}, '
                f'bias={self.bias})')


class DNAConv(MessagePassing):
    r"""The dynamic neighborhood aggregation operator from the `"Just Jump:
    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
    <https://arxiv.org/abs/1904.04849>`_ paper.

    .. math::
        \mathbf{x}_v^{(t)} = h_{\mathbf{\Theta}}^{(t)} \left( \mathbf{x}_{v
        \leftarrow v}^{(t)}, \left\{ \mathbf{x}_{v \leftarrow w}^{(t)} : w \in
        \mathcal{N}(v) \right\} \right)

    based on (multi-head) dot-product attention

    .. math::
        \mathbf{x}_{v \leftarrow w}^{(t)} = \textrm{Attention} \left(
        \mathbf{x}^{(t-1)}_v \, \mathbf{\Theta}_Q^{(t)}, [\mathbf{x}_w^{(1)},
        \ldots, \mathbf{x}_w^{(t-1)}] \, \mathbf{\Theta}_K^{(t)}, \,
        [\mathbf{x}_w^{(1)}, \ldots, \mathbf{x}_w^{(t-1)}] \,
        \mathbf{\Theta}_V^{(t)} \right)

    with :math:`\mathbf{\Theta}_Q^{(t)}, \mathbf{\Theta}_K^{(t)},
    \mathbf{\Theta}_V^{(t)}` denoting (grouped) projection matrices for query,
    key and value information, respectively.
    :math:`h^{(t)}_{\mathbf{\Theta}}` is implemented as a non-trainable
    version of :class:`paddle_geometric.nn.conv.GCNConv`.

    .. note::
        In contrast to other layers, this operator expects node features as
        shape :obj:`[num_nodes, num_layers, channels]`.

    Args:
        channels (int): Size of each input/output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        groups (int, optional): Number of groups to use for all linear
            projections. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of attention
            coefficients. (default: :obj:`0.`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, L, F)` where :math:`L` is the
          number of layers,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, heads: int = 1, groups: int = 1,
                 dropout: float = 0., cached: bool = False,
                 normalize: bool = True, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.bias = bias
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.multi_head = MultiHead(channels, channels, heads, groups, dropout,
                                    bias)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.multi_head.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        if x.ndim != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, '
                             'channels].')

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.shape[self.node_dim], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.shape[self.node_dim], False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        x_i = x_i[:, -1:]  # [num_edges, 1, channels]
        out = self.multi_head(x_i, x_j, x_j)  # [num_edges, 1, channels]
        return edge_weight.reshape([-1, 1]) * out.squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.multi_head.in_channels}, '
                f'heads={self.multi_head.heads}, '
                f'groups={self.multi_head.groups})')
