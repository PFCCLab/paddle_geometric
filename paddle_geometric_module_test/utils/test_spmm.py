import itertools
import warnings

import pytest
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric import EdgeIndex
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import withCUDA, withPackage
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import spmm, to_paddle_coo_tensor

HAS_PADDLE_SPARSE_CSC = (
    hasattr(paddle, 'sparse_csc')
    and hasattr(paddle.sparse, 'sparse_csc_tensor')
    and hasattr(paddle.Tensor, 'to_sparse_csc')
)


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_spmm_basic(device, reduce):
    src = paddle.randn(shape=[5, 4, place=device])
    other = paddle.randn(shape=[4, 8, place=device])

    out1 = (src @ other) / (src.shape[1] if reduce == 'mean' else 1)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.shape== (5, 8)
    assert paddle.allclose(out1, out2, atol=1e-6)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
        assert paddle.allclose(out2, out3, atol=1e-6)

    # Test `mean` reduction with isolated nodes:
    src[0] = 0.
    out1 = (src @ other) / (4. if reduce == 'mean' else 1.)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.shape== (5, 8)
    assert paddle.allclose(out1, out2, atol=1e-6)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
        assert paddle.allclose(out2, out3, atol=1e-6)


@withCUDA
@pytest.mark.parametrize('reduce', ['min', 'max'])
def test_spmm_reduce(device, reduce):
    src = paddle.randn(shape=[5, 4, place=device])
    other = paddle.randn(shape=[4, 8, place=device])

    if src.place.is_gpu_place():
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src.to_sparse_csr(), other, reduce)
    else:
        out1 = spmm(src.to_sparse_csr(), other, reduce)
        assert out1.shape== (5, 8)
        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            out2 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
            assert paddle.allclose(out1, out2)


@withCUDA
@pytest.mark.parametrize(
    'layout',
    [paddle.sparse_coo, paddle.sparse_csr] + (
        [paddle.sparse_csc] if HAS_PADDLE_SPARSE_CSC else []
    ),
)
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_spmm_layout(device, layout, reduce):
    src = paddle.randn(shape=[5, 4, place=device])
    if layout == paddle.sparse_coo:
        src = src.to_sparse_coo()
    elif layout == paddle.sparse_csr:
        src = src.to_sparse_csr()
    else:
        assert layout == paddle.sparse_csc
        src = src.to_sparse_csc()
    other = paddle.randn(shape=[4, 8, place=device])

    if src.place.is_gpu_place() and reduce in {'min', 'max'}:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src, other, reduce=reduce)
    elif layout != paddle.sparse_csr:
        with pytest.warns(UserWarning, match="Converting sparse tensor"):
            spmm(src, other, reduce=reduce)
    else:
        spmm(src, other, reduce=reduce)


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_spmm_jit(reduce):
    def jit_paddle_sparse(src: SparseTensor, other: paddle.Tensor, reduce: str) -> paddle.Tensor:
        """
        Perform sparse matrix multiplication with a specified reduction method.
        """
        return spmm(src, other, reduce=reduce)

    def jit_paddle(src: paddle.Tensor, other: paddle.Tensor, reduce: str) -> paddle.Tensor:
        """
        Perform dense matrix multiplication with a specified reduction method.
        """
        return spmm(src, other, reduce=reduce)

    src = paddle.randn(shape=[5, 4])
    other = paddle.randn(shape=[4, 8])

    out1 = src @ other
    out2 = jit_paddle(src.to_sparse_csr(), other, reduce)
    assert out1.shape== (5, 8)
    if reduce == 'sum':
        assert paddle.allclose(out1, out2, atol=1e-6)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        out3 = jit_paddle_sparse(SparseTensor.from_dense(src), other, reduce)
        assert paddle.allclose(out2, out3, atol=1e-6)


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_spmm_edge_index(device, reduce):
    src = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(4, 3),
        sort_order='row',
        place=device,
    )
    other = paddle.rand(3, 4, place=device)
    out = spmm(src, other, reduce=reduce)
    assert out.shape== (4, 4)

    if not other.place.is_gpu_place() or reduce not in ['min', 'max']:
        out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
        assert paddle.allclose(out, out2)


if __name__ == '__main__':
    import argparse

    warnings.filterwarnings('ignore', ".*Sparse CSR tensor support.*")
    warnings.filterwarnings('ignore', ".*Converting sparse tensor to CSR.*")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = paddle.randn(shape=[num_nodes, 64, place=args.device])
    edge_index = paddle.randint(num_nodes, (2, num_edges), place=args.device)

    reductions = ['sum', 'mean']
    if not x.place.is_gpu_place():
        reductions.extend(['min', 'max'])
    layouts = [paddle.sparse_coo, paddle.sparse_csr]
    if HAS_PADDLE_SPARSE_CSC:
        layouts.append(paddle.sparse_csc)

    for reduce, layout in itertools.product(reductions, layouts):
        print(f'Aggregator: {reduce}, Layout: {layout}')

        adj = to_paddle_coo_tensor(edge_index, size=num_nodes)
        adj = adj.to_sparse(layout=layout)

        benchmark(
            funcs=[spmm],
            func_names=['spmm'],
            args=(adj, x, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
