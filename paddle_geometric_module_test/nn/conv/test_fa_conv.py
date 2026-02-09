from typing import Tuple

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.nn import FAConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


def test_fa_conv():
    x = paddle.randn(shape=[4, 16])
    x_0 = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = FAConv(16, eps=1.0, cached=True)
    assert str(conv) == 'FAConv(16, eps=1.0)'
    out = conv(x, x_0, edge_index)
    assert conv._cached_edge_index is not None
    assert tuple(out.shape)== (4, 16)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x, x_0, adj2.t()), out, atol=1e-6)
        assert conv._cached_adj_t is not None
        assert paddle.allclose(conv(x, x_0, adj2.t()), out, atol=1e-6)

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                x_0: Tensor,
                edge_index: Adj,
            ) -> Tensor:
                return self.conv(x, x_0, edge_index)

        jit = paddle.jit.to_static(MyModule())
        assert paddle.allclose(jit(x, x_0, edge_index), out, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, x_0, adj2.t()), out)

    conv.reset_parameters()
    assert conv._cached_edge_index is None
    assert conv._cached_adj_t is None

    # Test without caching:
    conv.cached = False
    out = conv(x, x_0, edge_index)
    assert paddle.allclose(conv(x, x_0, adj1.t()), out, atol=1e-6)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(conv(x, x_0, adj2.t()), out, atol=1e-6)

    # Test `return_attention_weights`:
    result = conv(x, x_0, edge_index, return_attention_weights=True)
    assert paddle.allclose(result[0], out, atol=1e-6)
    assert tuple(result[1][0].shape)== (2, 10)
    assert tuple(result[1][1].shape)== (10, )
    assert conv._alpha is None

    result = conv(x, x_0, adj1.t(), return_attention_weights=True)
    assert paddle.allclose(result[0], out, atol=1e-6)
    assert tuple(result[1][0].shape)== (4, 4)
    assert result[1][0].nnz() == 10
    assert conv._alpha is None

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        result = conv(x, x_0, adj2.t(), return_attention_weights=True)
        assert paddle.allclose(result[0], out, atol=1e-6)
        assert result[1].sizes() == [4, 4] and result[1].nnz() == 10
        assert conv._alpha is None

    if is_full_test():

        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(
                self,
                x: Tensor,
                x_0: Tensor,
                edge_index: Tensor,
            ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
                return self.conv(x, x_0, edge_index,
                                 return_attention_weights=True)

        jit = paddle.jit.to_static(MyModule())
        result = jit(x, x_0, edge_index)
        assert paddle.allclose(result[0], out, atol=1e-6)
        assert tuple(result[1][0].shape)== (2, 10)
        assert tuple(result[1][1].shape)== (10, )
        assert conv._alpha is None

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:

            class MyModule(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.conv = conv

                def forward(
                    self,
                    x: Tensor,
                    x_0: Tensor,
                    edge_index: SparseTensor,
                ) -> Tuple[Tensor, SparseTensor]:
                    return self.conv(x, x_0, edge_index,
                                     return_attention_weights=True)

            jit = paddle.jit.to_static(MyModule())
            result = jit(x, x_0, adj2.t())
            assert paddle.allclose(result[0], out, atol=1e-6)
            assert result[1].sizes() == [4, 4] and result[1].nnz() == 10
            assert conv._alpha is None
