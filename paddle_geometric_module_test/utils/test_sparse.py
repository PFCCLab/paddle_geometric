import os.path as osp

import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.io import fs
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import is_full_test, withCUDA, withPackage
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import (
    dense_to_sparse,
    is_sparse,
    is_paddle_sparse_tensor,
    to_edge_index,
    to_paddle_coo_tensor,
    to_paddle_csc_tensor,
    to_paddle_csr_tensor,
    to_paddle_sparse_tensor,
)
from paddle_geometric.utils.sparse import cat

HAS_PADDLE_SPARSE_CSC = (
    hasattr(paddle, 'sparse_csc')
    and hasattr(paddle.sparse, 'sparse_csc_tensor')
    and hasattr(paddle.Tensor, 'to_sparse_csc')
)


def test_dense_to_sparse():
    adj = paddle.to_tensor([
        [3.0, 1.0],
        [2.0, 0.0],
    ])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [3, 1, 2]

    if is_full_test():
        jit = dense_to_sparse
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
        assert edge_attr.tolist() == [3, 1, 2]

    adj = paddle.to_tensor([[
        [3.0, 1.0],
        [2.0, 0.0],
    ], [
        [0.0, 1.0],
        [0.0, 2.0],
    ]])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
    assert edge_attr.tolist() == [3, 1, 2, 1, 2]

    if is_full_test():
        jit = dense_to_sparse
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
        assert edge_attr.tolist() == [3, 1, 2, 1, 2]

    adj = paddle.to_tensor([
        [
            [3.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 3.0],
            [0.0, 5.0, 0.0],
        ],
    ])
    mask = paddle.to_tensor([[True, True, False], [True, True, True]])

    edge_index, edge_attr = dense_to_sparse(adj, mask)

    assert edge_index.tolist() == [[0, 0, 1, 2, 3, 3, 4],
                                   [0, 1, 0, 3, 3, 4, 3]]
    assert edge_attr.tolist() == [3, 1, 2, 1, 2, 3, 5]

    if is_full_test():
        jit = dense_to_sparse
        edge_index, edge_attr = jit(adj, mask)
        assert edge_index.tolist() == [[0, 0, 1, 2, 3, 3, 4],
                                       [0, 1, 0, 3, 3, 4, 3]]
        assert edge_attr.tolist() == [3, 1, 2, 1, 2, 3, 5]


def test_dense_to_sparse_bipartite():
    edge_index, edge_attr = dense_to_sparse(paddle.rand(2, 10, 5))
    assert edge_index[0].max() == 19
    assert edge_index[1].max() == 9


def test_is_paddle_sparse_tensor():
    x = paddle.randn(shape=[5, 5])

    assert not is_paddle_sparse_tensor(x)
    assert is_paddle_sparse_tensor(x.to_sparse())

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert not is_paddle_sparse_tensor(SparseTensor.from_dense(x))


def test_is_sparse():
    x = paddle.randn(shape=[5, 5])

    assert not is_sparse(x)
    assert is_sparse(x.to_sparse())

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert is_sparse(SparseTensor.from_dense(x))


def test_to_paddle_coo_tensor():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 8])

    adj = to_paddle_coo_tensor(edge_index, is_coalesced=False)
    assert adj.is_coalesced()
    assert adj.shape== (4, 4)
    assert adj.layout == paddle.sparse_coo
    assert paddle.allclose(adj.indices(), edge_index)

    adj = to_paddle_coo_tensor(edge_index, is_coalesced=True)
    assert adj.is_coalesced()
    assert adj.shape== (4, 4)
    assert adj.layout == paddle.sparse_coo
    assert paddle.allclose(adj.indices(), edge_index)

    adj = to_paddle_coo_tensor(edge_index, size=6)
    assert adj.shape== (6, 6)
    assert adj.layout == paddle.sparse_coo
    assert paddle.allclose(adj.indices(), edge_index)

    adj = to_paddle_coo_tensor(edge_index, edge_attr)
    assert adj.shape== (4, 4, 8)
    assert adj.layout == paddle.sparse_coo
    assert paddle.allclose(adj.indices(), edge_index)
    assert paddle.allclose(adj.values(), edge_attr)

    if is_full_test():
        jit = to_paddle_coo_tensor
        adj = jit(edge_index, edge_attr)
        assert adj.shape== (4, 4, 8)
        assert adj.layout == paddle.sparse_coo
        assert paddle.allclose(adj.indices(), edge_index)
        assert paddle.allclose(adj.values(), edge_attr)


def test_to_paddle_csr_tensor():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    adj = to_paddle_csr_tensor(edge_index)
    assert adj.shape== (4, 4)
    assert adj.layout == paddle.sparse_csr
    assert paddle.allclose(adj.to_sparse_coo().coalesce().indices(), edge_index)

    edge_weight = paddle.randn(shape=[edge_index.shape[1]])
    adj = to_paddle_csr_tensor(edge_index, edge_weight)
    assert adj.shape== (4, 4)
    assert adj.layout == paddle.sparse_csr
    coo = adj.to_sparse_coo().coalesce()
    assert paddle.allclose(coo.indices(), edge_index)
    assert paddle.allclose(coo.values(), edge_weight)



def test_to_paddle_csc_tensor():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    adj = to_paddle_csc_tensor(edge_index)
    if HAS_PADDLE_SPARSE_CSC:
        assert adj.shape== (4, 4)
        assert adj.layout == paddle.sparse_csc
        adj_coo = adj.to_sparse_coo().coalesce()
        assert paddle.allclose(adj_coo.indices().flip([0]), edge_index)
    else:
        from paddle_geometric.typing import MockPaddleCSCTensor
        assert isinstance(adj, MockPaddleCSCTensor)
        csr = adj.t()
        edge_index_out, _ = to_edge_index(csr)
        expected = edge_index.flip([0])
        num_nodes = int(expected.max().item()) + 1
        order_out = (edge_index_out[0] * num_nodes + edge_index_out[1]).argsort()
        order_exp = (expected[0] * num_nodes + expected[1]).argsort()
        assert paddle.allclose(edge_index_out[:, order_out], expected[:, order_exp])

    edge_weight = paddle.randn(shape=[edge_index.shape[1]])
    adj = to_paddle_csc_tensor(edge_index, edge_weight)
    if HAS_PADDLE_SPARSE_CSC:
        assert adj.shape== (4, 4)
        assert adj.layout == paddle.sparse_csc
        adj_coo = adj.to_sparse_coo().coalesce()

        perm = adj_coo.indices()[0].argsort()
        assert paddle.allclose(adj_coo.indices()[:, perm], edge_index)
        assert paddle.allclose(adj_coo.values()[perm], edge_weight)
    else:
        from paddle_geometric.typing import MockPaddleCSCTensor
        assert isinstance(adj, MockPaddleCSCTensor)
        csr = adj.t()
        edge_index_out, edge_weight_out = to_edge_index(csr)
        expected = edge_index.flip([0])
        num_nodes = int(expected.max().item()) + 1
        order_out = (edge_index_out[0] * num_nodes + edge_index_out[1]).argsort()
        order_exp = (expected[0] * num_nodes + expected[1]).argsort()
        assert paddle.allclose(edge_index_out[:, order_out], expected[:, order_exp])
        assert paddle.allclose(edge_weight_out[order_out], edge_weight[order_exp])


@withPackage('paddle>=2.1.0')
def test_to_paddle_coo_tensor_save_load(tmp_path):
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    adj = to_paddle_coo_tensor(edge_index, is_coalesced=False)
    assert adj.is_coalesced()

    path = osp.join(tmp_path, 'adj.pdparams')
    paddle.save(adj, path)
    adj = fs.paddle_load(path)
    assert adj.is_coalesced()


def test_to_edge_index():
    adj = paddle.to_tensor([
        [0., 1., 0., 0.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [0., 0., 1., 0.],
    ]).to_sparse()

    edge_index, edge_attr = to_edge_index(adj)
    assert edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert edge_attr.tolist() == [1., 1., 1., 1., 1., 1.]

    if is_full_test():
        jit = to_edge_index
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
        assert edge_attr.tolist() == [1., 1., 1., 1., 1., 1.]


@withCUDA
@pytest.mark.parametrize(
    'layout',
    [paddle.sparse_coo, paddle.sparse_csr] + (
        [paddle.sparse_csc] if HAS_PADDLE_SPARSE_CSC else []
    ),
)
@pytest.mark.parametrize('dim', [0, 1, (0, 1)])
def test_cat(layout, dim, device):
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], place=device)

    edge_weight = paddle.rand(4, place=device)

    adj = to_paddle_sparse_tensor(edge_index, edge_weight, layout=layout)

    out = cat([adj, adj], dim=dim)
    edge_index, edge_weight = to_edge_index(out.to_sparse_csr())

    if dim == 0:
        assert out.shape== (6, 3)
        assert edge_index[0].tolist() == [0, 1, 1, 2, 3, 4, 4, 5]
        assert edge_index[1].tolist() == [1, 0, 2, 1, 1, 0, 2, 1]
    elif dim == 1:
        assert out.shape== (3, 6)
        assert edge_index[0].tolist() == [0, 0, 1, 1, 1, 1, 2, 2]
        assert edge_index[1].tolist() == [1, 4, 0, 2, 3, 5, 1, 4]
    else:
        assert out.shape== (6, 6)
        assert edge_index[0].tolist() == [0, 1, 1, 2, 3, 4, 4, 5]
        assert edge_index[1].tolist() == [1, 0, 2, 1, 4, 3, 5, 4]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    edge_index = paddle.randint(num_nodes, (2, num_edges), place=args.device)

    benchmark(
        funcs=[
            SparseTensor.from_edge_index, to_paddle_coo_tensor,
            to_paddle_csr_tensor, to_paddle_csc_tensor
        ],
        func_names=['SparseTensor', 'To COO', 'To CSR', 'To CSC'],
        args=(edge_index, None, (num_nodes, num_nodes)),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
    )
