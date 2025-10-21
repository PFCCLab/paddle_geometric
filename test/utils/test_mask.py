import paddle

from paddle_geometric.utils import index_to_mask, mask_select, mask_to_index


def test_mask_select():
    src = paddle.randn(6, 8)
    mask = paddle.to_tensor([False, True, False, True, False, True])

    out = mask_select(src, 0, mask)
    assert tuple(out.shape) == (3, 8)
    assert paddle.equal_all(src[paddle.to_tensor([1, 3, 5])], out).item()

    # jit = paddle.jit.to_static(mask_select)
    # assert paddle.equal(jit(src, 0, mask), out)


def test_index_to_mask():
    index = paddle.to_tensor([1, 3, 5])

    mask = index_to_mask(index)
    assert mask.tolist() == [False, True, False, True, False, True]

    mask = index_to_mask(index, size=7)
    assert mask.tolist() == [False, True, False, True, False, True, False]


def test_mask_to_index():
    mask = paddle.to_tensor([False, True, False, True, False, True])

    index = mask_to_index(mask)
    assert index.tolist() == [1, 3, 5]
