import paddle

from paddle_geometric.nn import FusedGATConv


def test_to_graph_format():
    edge_index = paddle.to_tensor([[1, 0, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    assert csr[0].dtype == paddle.int32
    assert paddle.allclose(csr[0], paddle.to_tensor([0, 1, 2, 3, 4], dtype=paddle.int32))
    assert csr[1].dtype == paddle.int32
    assert paddle.allclose(csr[1], paddle.to_tensor([0, 0, 1, 1], dtype=paddle.int32))
    assert csc[0].dtype == paddle.int32
    assert paddle.allclose(csc[0], paddle.to_tensor([0, 1, 2, 3], dtype=paddle.int32))
    assert csc[1].dtype == paddle.int32
    assert paddle.allclose(csc[1], paddle.to_tensor([0, 2, 4, 4, 4], dtype=paddle.int32))
    assert perm.dtype == paddle.int32
    assert paddle.allclose(perm, paddle.to_tensor([0, 1, 2, 3], dtype=paddle.int32))


def test_fused_gat_conv():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False)
    assert str(conv) == 'FusedGATConv(8, 32, heads=2)'

    out = conv(x, csr, csc, perm)
    assert out.shape == [4, 64]


def test_fused_gat_conv_concat_false():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False, concat=False)
    assert 'FusedGATConv(8, 32, heads=2)' in str(conv)

    out = conv(x, csr, csc, perm)
    assert out.shape == [4, 32]


def test_fused_gat_conv_with_dropout():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False, dropout=0.5)
    assert 'FusedGATConv(8, 32, heads=2)' in str(conv)

    conv.eval()  # Set to eval mode to disable dropout
    out = conv(x, csr, csc, perm)
    assert out.shape == [4, 64]


def test_fused_gat_conv_no_bias():
    x = paddle.randn([4, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False, bias=False)
    assert 'FusedGATConv(8, 32, heads=2)' in str(conv)

    out = conv(x, csr, csc, perm)
    assert out.shape == [4, 64]


def test_fused_gat_conv_errors():
    # Test that add_self_loops=True raises error
    try:
        from paddle_geometric.nn import FusedGATConv
        conv = FusedGATConv(8, 32, add_self_loops=True)
        assert False, "Should have raised ValueError for add_self_loops=True"
    except ValueError as e:
        assert "does not support adding self-loops" in str(e)

    # Test that edge_dim is not None raises error
    try:
        conv = FusedGATConv(8, 32, edge_dim=4, add_self_loops=False)
        assert False, "Should have raised ValueError for edge_dim is not None"
    except ValueError as e:
        assert "does not support edge features" in str(e)