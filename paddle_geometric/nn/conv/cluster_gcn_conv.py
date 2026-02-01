from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.typing import Adj, OptTensor, SparseTensor, paddle_sparse
from paddle_geometric.utils import (
    add_self_loops,
    degree,
    is_paddle_sparse_tensor,
    remove_self_loops,
    spmm,
    to_edge_index,
)
from paddle_geometric.utils.sparse import set_sparse_value

class ClusterGCNConv(MessagePassing):
    r"""The ClusterGCN graph convolutional operator from the
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \left( \mathbf{\hat{A}} + \lambda \cdot
        \textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
        \mathbf{X} \mathbf{W}_2

    where :math:`\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}(\mathbf{A}
    + \mathbf{I})`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        diag_lambda (float, optional): Diagonal enhancement value
            :math:`\lambda`. (default: :obj:`0.`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 diag_lambda: float = 0., add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.add_self_loops = add_self_loops

        self.lin_out = Linear(in_channels, out_channels, bias=bias,
                              weight_initializer='glorot')
        self.lin_root = Linear(in_channels, out_channels, bias=False,
                               weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_out.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        num_nodes = x.shape[self.node_dim]
        edge_weight: OptTensor = None

        if isinstance(edge_index, SparseTensor):
            assert edge_index.size(0) == edge_index.size(1)

            if self.add_self_loops:
                edge_index = paddle_sparse.set_diag(edge_index)

            col, row, _ = edge_index.coo()
            deg_inv = 1. / paddle_sparse.sum(edge_index, dim=1).clip(min=1.)

            edge_weight = deg_inv[col]
            edge_weight = paddle.where(row == col,
                                       edge_weight + self.diag_lambda *
                                       deg_inv[row],
                                       edge_weight)
            edge_index = edge_index.set_value(edge_weight, layout='coo')

        elif isinstance(edge_index, Tensor) and is_paddle_sparse_tensor(
                edge_index):
            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            col_and_row, value = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
            deg_inv = 1. / degree(col, num_nodes=edge_index.shape[0]).clip(
                min=1.)

            edge_weight = deg_inv[col]
            edge_weight = paddle.where(row == col,
                                       edge_weight + self.diag_lambda *
                                       deg_inv[row],
                                       edge_weight)

            edge_index = set_sparse_value(edge_index, edge_weight)

        else:
            if self.add_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            row, col = edge_index[0], edge_index[1]
            deg_inv = 1. / degree(col, num_nodes=num_nodes).clip(min=1.)

            edge_weight = deg_inv[col]
            edge_weight = paddle.where(row == col,
                                       edge_weight + self.diag_lambda *
                                       deg_inv[row],
                                       edge_weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_out(out) + self.lin_root(x)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, diag_lambda={self.diag_lambda})')
