import pytest

pytest.skip("CuGraph conv tests are not required for Paddle port.",
            allow_module_level=True)

import paddle

from paddle_geometric import EdgeIndex
from paddle_geometric.nn import CuGraphGATConv, GATConv
from paddle_geometric.testing import onlyCUDA, withPackage


@onlyCUDA
@withPackage('pylibcugraphops>=23.02')
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bipartite', [True, False])
@pytest.mark.parametrize('concat', [True, False])
@pytest.mark.parametrize('heads', [1, 2, 3])
@pytest.mark.parametrize('max_num_neighbors', [8, None])
def test_gat_conv_equality(bias, bipartite, concat, heads, max_num_neighbors):
    in_channels, out_channels = 5, 2
    kwargs = dict(bias=bias, concat=concat)

    size = (10, 8) if bipartite else (10, 10)
    x = paddle.rand([size[0], in_channels]).cuda()
    edge_index = paddle.to_tensor([
        [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ]).cuda()

    conv1 = GATConv(in_channels, out_channels, heads, add_self_loops=False,
                    **kwargs).cuda()
    conv2 = CuGraphGATConv(in_channels, out_channels, heads, **kwargs).cuda()

    with paddle.no_grad():
        conv2.lin.weight.data[:, :] = conv1.lin.weight.data
        conv2.att.data[:heads * out_channels] = conv1.att_src.data.flatten()
        conv2.att.data[heads * out_channels:] = conv1.att_dst.data.flatten()

    if bipartite:
        out1 = conv1((x, x[:size[1]]), edge_index)
    else:
        out1 = conv1(x, edge_index)

    out2 = conv2(
        x,
        EdgeIndex(edge_index, sparse_size=size),
        max_num_neighbors=max_num_neighbors,
    )
    assert paddle.allclose(out1, out2, atol=1e-3)

    grad_output = paddle.rand_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert paddle.allclose(conv1.lin.weight.grad, conv2.lin.weight.grad,
                          atol=1e-3)
    assert paddle.allclose(conv1.att_src.grad.flatten(),
                          conv2.att.grad[:heads * out_channels], atol=1e-3)
    assert paddle.allclose(conv1.att_dst.grad.flatten(),
                          conv2.att.grad[heads * out_channels:], atol=1e-3)
    if bias:
        assert paddle.allclose(conv1.bias.grad, conv2.bias.grad, atol=1e-3)
