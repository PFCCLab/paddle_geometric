from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import ReLU

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.nn.inits import glorot, ones, zeros
from paddle_geometric.typing import Adj, OptTensor, Size, SparseTensor
from paddle_geometric.utils import is_paddle_sparse_tensor, scatter, softmax
from paddle_geometric.utils.sparse import set_sparse_value


class RGATConv(MessagePassing):
    r"""The relational graph attentional operator from the
    "Relational Graph Attention Networks" paper.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        mod: Optional[str] = None,
        attention_mechanism: str = "across-relation",
        attention_mode: str = "additive-self-attention",
        heads: int = 1,
        dim: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.mod = mod
        self.activation = ReLU()
        self.concat = concat
        self.attention_mode = attention_mode
        self.attention_mechanism = attention_mechanism
        self.dim = dim
        self.edge_dim = edge_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        mod_types = ['additive', 'scaled', 'f-additive', 'f-scaled']

        if (self.attention_mechanism != "within-relation"
                and self.attention_mechanism != "across-relation"):
            raise ValueError('attention mechanism must either be '
                             '"within-relation" or "across-relation"')

        if (self.attention_mode != "additive-self-attention"
                and self.attention_mode != "multiplicative-self-attention"):
            raise ValueError('attention mode must either be '
                             '"additive-self-attention" or '
                             '"multiplicative-self-attention"')

        if self.attention_mode == "additive-self-attention" and self.dim > 1:
            raise ValueError('"additive-self-attention" mode cannot be '
                             'applied when value of d is greater than 1. '
                             'Use "multiplicative-self-attention" instead.')

        if self.dropout > 0.0 and self.mod in mod_types:
            raise ValueError('mod must be None with dropout value greater '
                             'than 0 in order to sample attention '
                             'coefficients stochastically')

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.q = self.create_parameter(
            shape=[self.heads * self.out_channels, self.heads * self.dim])
        self.k = self.create_parameter(
            shape=[self.heads * self.out_channels, self.heads * self.dim])

        if bias and concat:
            self.bias = self.create_parameter(
                shape=[self.heads * self.dim * self.out_channels])
        elif bias and not concat:
            self.bias = self.create_parameter(
                shape=[self.dim * self.out_channels])
        else:
            self.bias = None

        if edge_dim is not None:
            self.lin_edge = Linear(self.edge_dim,
                                   self.heads * self.out_channels, bias=False,
                                   weight_initializer='glorot')
            self.e = self.create_parameter(
                shape=[self.heads * self.out_channels, self.heads * self.dim])
        else:
            self.lin_edge = None
            self.e = None

        if num_bases is not None:
            self.att = self.create_parameter(
                shape=[self.num_relations, self.num_bases])
            self.basis = self.create_parameter(
                shape=[self.num_bases, self.in_channels,
                       self.heads * self.out_channels])
        elif num_blocks is not None:
            if (self.in_channels % self.num_blocks != 0
                    or (self.heads * self.out_channels) % self.num_blocks != 0):
                raise ValueError(
                    "both 'in_channels' and 'heads * out_channels' must be "
                    "multiple of 'num_blocks' used")
            self.weight = self.create_parameter(
                shape=[
                    self.num_relations,
                    self.num_blocks,
                    self.in_channels // self.num_blocks,
                    (self.heads * self.out_channels) // self.num_blocks,
                ])
        else:
            self.weight = self.create_parameter(
                shape=[self.num_relations, self.in_channels,
                       self.heads * self.out_channels])

        self.w = self.create_parameter(shape=[self.out_channels])
        self.l1 = self.create_parameter(shape=[1, self.out_channels])
        self.b1 = self.create_parameter(shape=[1, self.out_channels])
        self.l2 = self.create_parameter(
            shape=[self.out_channels, self.out_channels])
        self.b2 = self.create_parameter(shape=[1, self.out_channels])

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.num_bases is not None:
            glorot(self.basis)
            glorot(self.att)
        else:
            glorot(self.weight)
        glorot(self.q)
        glorot(self.k)
        zeros(self.bias)
        ones(self.l1)
        zeros(self.b1)
        if self.l2 is not None:
            self.l2.set_value(
                paddle.full(shape=self.l2.shape,
                            fill_value=1 / self.out_channels))
        zeros(self.b2)
        if self.lin_edge is not None:
            glorot(self.lin_edge)
            glorot(self.e)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
                             size=size, edge_attr=edge_attr)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_paddle_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                return out, (edge_index, alpha)
            if isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.num_bases is not None:
            w = paddle.matmul(self.att, self.basis.reshape([self.num_bases, -1]))
            w = w.reshape([self.num_relations, self.in_channels,
                           self.heads * self.out_channels])
        if self.num_blocks is not None:
            if (not paddle.is_floating_point(x_i)
                    and not paddle.is_floating_point(x_j)):
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')
            w = self.weight
            x_i = x_i.reshape([-1, 1, w.shape[1], w.shape[2]])
            x_j = x_j.reshape([-1, 1, w.shape[1], w.shape[2]])
            w = paddle.index_select(w, index=edge_type, axis=0)
            outi = paddle.einsum('abcd,acde->ace', x_i, w)
            outi = outi.contiguous().reshape([-1, self.heads * self.out_channels])
            outj = paddle.einsum('abcd,acde->ace', x_j, w)
            outj = outj.contiguous().reshape([-1, self.heads * self.out_channels])
        else:
            if self.num_bases is None:
                w = self.weight
            w = paddle.index_select(w, index=edge_type, axis=0)
            outi = paddle.bmm(x_i.unsqueeze(1), w).squeeze(-2)
            outj = paddle.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        qi = paddle.matmul(outi, self.q)
        kj = paddle.matmul(outj, self.k)

        alpha_edge, alpha = 0, paddle.to_tensor([0])
        if edge_attr is not None:
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.reshape([-1, 1])
            assert self.lin_edge is not None, (
                "Please set 'edge_dim = edge_attr.shape[-1]' while calling the "
                "RGATConv layer")
            edge_attributes = self.lin_edge(edge_attr).reshape(
                [-1, self.heads * self.out_channels])
            if edge_attributes.shape[0] != edge_attr.shape[0]:
                edge_attributes = paddle.index_select(edge_attributes,
                                                     index=edge_type, axis=0)
            alpha_edge = paddle.matmul(edge_attributes, self.e)

        if self.attention_mode == "additive-self-attention":
            if edge_attr is not None:
                alpha = paddle.add(qi, kj) + alpha_edge
            else:
                alpha = paddle.add(qi, kj)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        elif self.attention_mode == "multiplicative-self-attention":
            if edge_attr is not None:
                alpha = (qi * kj) * alpha_edge
            else:
                alpha = qi * kj

        if self.attention_mechanism == "within-relation":
            across_out = paddle.zeros_like(alpha)
            for r in range(self.num_relations):
                mask = edge_type == r
                if mask.astype('int64').sum() == 0:
                    continue
                across_out[mask] = softmax(alpha[mask], index[mask])
            alpha = across_out
        elif self.attention_mechanism == "across-relation":
            alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha

        if self.mod == "additive":
            if self.attention_mode == "additive-self-attention":
                ones = paddle.ones_like(alpha)
                h = (outj.reshape([-1, self.heads, self.out_channels]) *
                     ones.reshape([-1, self.heads, 1]))
                h = paddle.multiply(self.w, h)

                return (outj.reshape([-1, self.heads, self.out_channels]) *
                        alpha.reshape([-1, self.heads, 1]) + h)
            if self.attention_mode == "multiplicative-self-attention":
                ones = paddle.ones_like(alpha)
                h = (outj.reshape([-1, self.heads, 1, self.out_channels]) *
                     ones.reshape([-1, self.heads, self.dim, 1]))
                h = paddle.multiply(self.w, h)

                return (outj.reshape([-1, self.heads, 1, self.out_channels]) *
                        alpha.reshape([-1, self.heads, self.dim, 1]) + h)

        elif self.mod == "scaled":
            if self.attention_mode == "additive-self-attention":
                ones = paddle.ones(index.shape, dtype=alpha.dtype)
                degree = scatter(ones, index, dim_size=size_i,
                                 reduce='sum')[index].unsqueeze(-1)
                degree = paddle.matmul(degree, self.l1) + self.b1
                degree = self.activation(degree)
                degree = paddle.matmul(degree, self.l2) + self.b2

                return paddle.multiply(
                    outj.reshape([-1, self.heads, self.out_channels]) *
                    alpha.reshape([-1, self.heads, 1]),
                    degree.reshape([-1, 1, self.out_channels]))
            if self.attention_mode == "multiplicative-self-attention":
                ones = paddle.ones(index.shape, dtype=alpha.dtype)
                degree = scatter(ones, index, dim_size=size_i,
                                 reduce='sum')[index].unsqueeze(-1)
                degree = paddle.matmul(degree, self.l1) + self.b1
                degree = self.activation(degree)
                degree = paddle.matmul(degree, self.l2) + self.b2

                return paddle.multiply(
                    outj.reshape([-1, self.heads, 1, self.out_channels]) *
                    alpha.reshape([-1, self.heads, self.dim, 1]),
                    degree.reshape([-1, 1, 1, self.out_channels]))

        elif self.mod == "f-additive":
            alpha = paddle.where(alpha > 0, alpha + 1, alpha)

        elif self.mod == "f-scaled":
            ones = paddle.ones(index.shape, dtype=alpha.dtype)
            degree = scatter(ones, index, dim_size=size_i,
                             reduce='sum')[index].unsqueeze(-1)
            alpha = alpha * degree

        elif self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        if self.attention_mode == "additive-self-attention":
            return alpha.reshape([-1, self.heads, 1]) * outj.reshape(
                [-1, self.heads, self.out_channels])
        return (alpha.reshape([-1, self.heads, self.dim, 1]) *
                outj.reshape([-1, self.heads, 1, self.out_channels]))

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.attention_mode == "additive-self-attention":
            if self.concat is True:
                aggr_out = aggr_out.reshape(
                    [-1, self.heads * self.out_channels])
            else:
                aggr_out = aggr_out.mean(axis=1)

            if self.bias is not None:
                aggr_out = aggr_out + self.bias

            return aggr_out

        if self.concat is True:
            aggr_out = aggr_out.reshape(
                [-1, self.heads * self.dim * self.out_channels])
        else:
            aggr_out = aggr_out.mean(axis=1)
            aggr_out = aggr_out.reshape([-1, self.dim * self.out_channels])

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
