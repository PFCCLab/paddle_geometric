import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import GCNConv
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_coo_tensor, to_paddle_csc_tensor


def test_gcn_conv():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = paddle.rand([edge_index.shape[1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_paddle_csc_tensor(edge_index, value, size=(4, 4))

    conv = GCNConv(16, 32)
    assert str(conv) == 'GCNConv(16, 32)'

    out1 = conv(x, edge_index)
    assert tuple(out1.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert tuple(out2.shape)== (4, 32)
    assert paddle.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert paddle.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x, edge_index), out1, atol=1e-6)
        assert paddle.allclose(jit(x, edge_index, value), out2, atol=1e-6)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj3.t()), out1, atol=1e-6)
            assert paddle.allclose(jit(x, adj4.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv._cached_edge_index is not None
    assert paddle.allclose(conv(x, edge_index), out1, atol=1e-6)
    assert paddle.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        conv(x, adj3.t())
        assert conv._cached_adj_t is not None
        assert paddle.allclose(conv(x, adj3.t()), out1, atol=1e-6)


def test_gcn_conv_with_decomposed_layers():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    def hook(module, inputs):
        assert tuple(inputs[0]['x_j'].shape)== (10, 32 // module.decomposed_layers)

    conv = GCNConv(16, 32)
    conv.register_message_forward_pre_hook(hook)
    out1 = conv(x, edge_index)

    conv.decomposed_layers = 2
    assert conv.propagate.__module__.endswith('message_passing')
    out2 = conv(x, edge_index)
    assert paddle.allclose(out1, out2)

    # TorchScript should still work since it relies on class methods
    # (but without decomposition).
    paddle.jit.to_static(conv)

    conv.decomposed_layers = 1
    assert conv.propagate.__module__.endswith('GCNConv_propagate')


def test_gcn_conv_with_sparse_input_feature():
    x = paddle.sparse.sparse_coo_tensor(
        indices=paddle.to_tensor([[0, 0], [0, 1]]),
        values=paddle.to_tensor([1., 1.]),
        shape=[4, 16],
    )
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(16, 32)
    assert tuple(conv(x, edge_index).shape)== (4, 32)


def test_static_gcn_conv():
    x = paddle.randn(shape=[3, 4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(16, 32)
    out = conv(x, edge_index)
    assert tuple(out.shape)== (3, 4, 32)


def test_gcn_conv_error():
    with pytest.raises(ValueError, match="does not support adding self-loops"):
        GCNConv(16, 32, normalize=False, add_self_loops=True)


def test_gcn_conv_flow():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 0, 0], [1, 2, 3]])

    conv = GCNConv(16, 32, flow="source_to_target")
    out1 = conv(x, edge_index)
    conv.flow = "target_to_source"
    out2 = conv(x, edge_index.flip(0))
    assert paddle.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('layout', ['coo', 'csr'])
def test_gcn_norm_gradient(requires_grad, layout):
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = paddle.ones(edge_index.shape[1], requires_grad=requires_grad)
    adj = to_paddle_coo_tensor(edge_index, edge_weight)
    if layout == 'csr':
        adj = adj.to_sparse_csr()

    assert adj.requires_grad == gcn_norm(adj)[0].requires_grad
