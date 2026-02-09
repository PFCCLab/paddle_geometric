import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import SimpleConv
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import to_paddle_csc_tensor


@pytest.mark.parametrize('aggr, combine_root', [
    ('mean', None),
    ('sum', 'sum'),
    (['mean', 'max'], 'cat'),
    ('mean', 'self_loop'),
])
def test_simple_conv(aggr, combine_root):
    x1 = paddle.randn(shape=[4, 8])
    x2 = paddle.randn(shape=[2, 8])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    adj1 = to_paddle_csc_tensor(edge_index, size=(4, 4))

    conv = SimpleConv(aggr, combine_root)
    assert str(conv) == 'SimpleConv()'

    num_aggrs = 1 if isinstance(aggr, str) else len(aggr)
    output_size = sum([8] * num_aggrs) + (8 if combine_root == 'cat' else 0)

    out = conv(x1, edge_index)
    assert tuple(out.shape)== (4, output_size)
    assert paddle.allclose(conv(x1, edge_index, size=(4, 4)), out)
    assert paddle.allclose(conv(x1, adj1.t()), out)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        jit = paddle.jit.to_static(conv)
        assert paddle.allclose(jit(x1, edge_index), out)
        assert paddle.allclose(jit(x1, edge_index, size=(4, 4)), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x1, adj2.t()), out)

    # Test bipartite message passing:
    if combine_root != 'self_loop':
        adj1 = to_paddle_csc_tensor(edge_index, size=(4, 2))

        out = conv((x1, x2), edge_index)
        assert tuple(out.shape)== (2, output_size)
        assert paddle.allclose(conv((x1, x2), edge_index, size=(4, 2)), out)
        assert paddle.allclose(conv((x1, x2), adj1.t()), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            adj2 = SparseTensor.from_edge_index(edge_index,
                                                sparse_sizes=(4, 2))
            assert paddle.allclose(conv((x1, x2), adj2.t()), out)
