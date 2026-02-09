from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.typing import Adj, OptTensor, SparseTensor, paddle_sparse
from paddle_geometric.utils import index_sort, one_hot, scatter, spmm


def masked_edge_index(edge_index: Adj, edge_mask: Tensor) -> Adj:
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return paddle_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the "Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103> paper.

    This is a memory-efficient implementation that iterates over relations.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        if num_bases is not None:
            self.weight = self.create_parameter(
                shape=[num_bases, in_channels[0], out_channels])
            self.comp = self.create_parameter(
                shape=[num_relations, num_bases])
        elif num_blocks is not None:
            if (in_channels[0] % num_blocks != 0
                    or out_channels % num_blocks != 0):
                raise ValueError(
                    "Input and output channels must be divisible by num_blocks.")
            self.weight = self.create_parameter(
                shape=[
                    num_relations,
                    num_blocks,
                    in_channels[0] // num_blocks,
                    out_channels // num_blocks,
                ])
            self.comp = None
        else:
            self.weight = self.create_parameter(
                shape=[num_relations, in_channels[0], out_channels])
            self.comp = None

        if root_weight:
            self.root = self.create_parameter(
                shape=[in_channels[1], out_channels])
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = self.create_parameter(shape=[out_channels])
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None) -> Tensor:
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = x[0] if isinstance(x, tuple) else x
        if x_l is None:
            x_l = paddle.arange(self.in_channels_l, dtype='int64')

        x_r: Tensor = x_l if not isinstance(x, tuple) else x[1]

        size = (x_l.shape[0], x_r.shape[0])
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        out = paddle.zeros([x_r.shape[0], self.out_channels], dtype=self.weight.dtype)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition
            weight = paddle.matmul(self.comp, weight.reshape([self.num_bases, -1]))
            weight = weight.reshape(
                [self.num_relations, self.in_channels_l, self.out_channels])

        if self.num_blocks is not None:  # Block-diagonal-decomposition
            if not paddle.is_floating_point(x_r):
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.reshape([-1, weight.shape[1], weight.shape[2]])
                h = paddle.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.reshape([-1, self.out_channels])
        else:
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)

                if not paddle.is_floating_point(x_r):
                    out = out + self.propagate(
                        tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(tmp, x=x_l, size=size)
                    out = out + paddle.matmul(h, weight[i])

        if self.root is not None:
            if not paddle.is_floating_point(x_r):
                out = out + self.root[x_r]
            else:
                out = out + paddle.matmul(x_r, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')


class FastRGCNConv(RGCNConv):
    r"""See :class:`RGCNConv`."""

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None) -> Tensor:

        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = x[0] if isinstance(x, tuple) else x
        if x_l is None:
            x_l = paddle.arange(self.in_channels_l, dtype='int64')

        x_r: Tensor = x_l if not isinstance(x, tuple) else x[1]

        size = (x_l.shape[0], x_r.shape[0])

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)

        if self.root is not None:
            if not paddle.is_floating_point(x_r):
                out = out + self.root[x_r]
            else:
                out = out + paddle.matmul(x_r, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_type: Tensor,
                edge_index_j: Tensor) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = paddle.matmul(self.comp, weight.reshape([self.num_bases, -1]))
            weight = weight.reshape(
                [self.num_relations, self.in_channels_l, self.out_channels])

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if not paddle.is_floating_point(x_j):
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].reshape([-1, weight.shape[2], weight.shape[3]])
            x_j = x_j.reshape([-1, 1, weight.shape[1]])
            return paddle.bmm(x_j, weight).reshape([-1, self.out_channels])

        else:  # No regularization/Basis-decomposition ========================
            if not paddle.is_floating_point(x_j):
                weight_index = edge_type * weight.shape[1] + edge_index_j
                return weight.reshape([-1, self.out_channels])[weight_index]

            return paddle.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = one_hot(edge_type, self.num_relations, dtype=inputs.dtype)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = paddle.gather(norm, 1, edge_type.reshape([-1, 1]))
            norm = 1. / norm.clip(1.)
            inputs = norm * inputs

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)
