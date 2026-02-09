from typing import Tuple

import pytest
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.nn import GATv2Conv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('residual', [False, True])
def test_gatv2_conv(residual):
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = GATv2Conv(8, 32, heads=2, residual=residual)
    assert str(conv) == 'GATv2Conv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, 64)
    assert paddle.allclose(conv(x1, edge_index), out)
    assert paddle.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
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
            ) -> Tensor:
                return self.conv(x, edge_index)

        jit = paddle.jit.to_static(MyModule())
        assert paddle.allclose(jit(x1, edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert paddle.allclose(result[0], out)
    assert tuple(result[1][0].shape)== (2, 7)
    assert tuple(result[1][1].shape)== (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1


    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        result = conv(x1, adj2.t(), return_attention_weights=True)
        assert paddle.allclose(result[0], out, atol=1e-6)
        assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
            ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
                return self.conv(x, edge_index, return_attention_weights=True)

        jit = paddle.jit.to_static(MyModule())
        result = jit(x1, edge_index)
        assert paddle.allclose(result[0], out)
        assert tuple(result[1][0].shape)== (2, 7)
        assert tuple(result[1][1].shape)== (7, 2)
        assert result[1][1].min() >= 0 and result[1][1].max() <= 1

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
            assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7

    # Test bipartite message passing:
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert tuple(out.shape)== (2, 64)
    assert paddle.allclose(conv((x1, x2), edge_index), out)
    assert paddle.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
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
            ) -> Tensor:
                return self.conv(x, edge_index)

        jit = paddle.jit.to_static(MyModule())
        assert paddle.allclose(jit((x1, x2), edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)


def test_gatv2_conv_with_edge_attr():
    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = paddle.randn(shape=[edge_index.shape[1]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 4])

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value=0.5)
    out = conv(x, edge_index, edge_weight)
    assert tuple(out.shape)== (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value='mean')
    out = conv(x, edge_index, edge_weight)
    assert tuple(out.shape)== (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value=0.5)
    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value='mean')
    out = conv(x, edge_index, edge_attr)
    assert tuple(out.shape)== (4, 64)
