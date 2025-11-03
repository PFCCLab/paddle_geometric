import paddle

from paddle_geometric.utils import one_hot


def test_one_hot():
    index = paddle.to_tensor([0, 1, 2])

    out = one_hot(index)
    assert out.size() == (3, 3)
    assert out.dtype == paddle.float
    assert out.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    out = one_hot(index, num_classes=4, dtype=paddle.int64)
    assert out.size() == (3, 4)
    assert out.dtype == paddle.int64
    assert out.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
