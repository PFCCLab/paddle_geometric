from typing import Optional, Tuple

import pytest
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.nn import TransformerConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('edge_dim', [None, 8])
@pytest.mark.parametrize('concat', [True, False])
def test_transformer_conv(edge_dim, concat):
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 16])
    out_channels = 32
    heads = 2
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], edge_dim]) if edge_dim else None
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = TransformerConv(8, out_channels, heads, beta=True,
                           edge_dim=edge_dim, concat=concat)
    assert str(conv) == f'TransformerConv(8, {out_channels}, heads={heads})'

    out = conv(x1, edge_index, edge_attr)
    assert tuple(out.shape) == (4, out_channels * (heads if concat else 1))
    assert paddle.allclose(conv(x1, adj1.t(), edge_attr), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr,
                                            sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
            ) -> Tensor:
                return self.conv(x, edge_index, edge_attr)

        jit = paddle.jit.to_static(MyModule())
        assert paddle.allclose(jit(x1, edge_index, edge_attr), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, edge_attr, return_attention_weights=True)
    assert paddle.allclose(result[0], out)
    assert tuple(result[1][0].shape) == (2, 4)
    assert tuple(result[1][1].shape) == (4, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        result = conv(x1, adj2.t(), return_attention_weights=True)
        assert paddle.allclose(result[0], out, atol=1e-6)
        assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 4
        assert conv._alpha is None

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: Optional[Tensor],
            ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
                return self.conv(x, edge_index, edge_attr,
                                 return_attention_weights=True)

        jit = paddle.jit.to_static(MyModule())
        result = jit(x1, edge_index, edge_attr)
        assert paddle.allclose(result[0], out)
        assert tuple(result[1][0].shape) == (2, 4)
        assert tuple(result[1][1].shape) == (4, 2)
        assert result[1][1].min() >= 0 and result[1][1].max() <= 1
        assert conv._alpha is None

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:

            class MyModule(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.conv = conv

                def forward(
                    self,
                    x: Tensor,
                    edge_index: SparseTensor,
                ) -> Tuple[Tensor, SparseTensor]:
                    return self.conv(x, edge_index,
                                     return_attention_weights=True)

            jit = paddle.jit.to_static(MyModule())
            result = jit(x1, adj2.t())
            assert paddle.allclose(result[0], out, atol=1e-6)
            assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 4
            assert conv._alpha is None

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    conv = TransformerConv((8, 16), out_channels, heads=heads, beta=True,
                           edge_dim=edge_dim, concat=concat)
    assert str(conv) == (f'TransformerConv((8, 16), {out_channels}, '
                         f'heads={heads})')

    out = conv((x1, x2), edge_index, edge_attr)
    assert tuple(out.shape) == (2, out_channels * (heads if concat else 1))
    assert paddle.allclose(conv((x1, x2), adj1.t(), edge_attr), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, edge_attr,
                                            sparse_sizes=(4, 2))
        assert paddle.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tuple[Tensor, Tensor],
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
            ) -> Tensor:
                return self.conv(x, edge_index, edge_attr)

        jit = paddle.jit.to_static(MyModule())
        assert paddle.allclose(jit((x1, x2), edge_index, edge_attr), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)
