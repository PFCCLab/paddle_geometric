import paddle

from paddle_geometric.utils import unbatch, unbatch_edge_index


def test_unbatch():
    src = paddle.arange(10)
    batch = paddle.to_tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])

    out = unbatch(src, batch)
    assert len(out) == 5
    for i in range(len(out)):
        assert paddle.equal_all(out[i], src[batch == i]).item()


def test_unbatch_edge_index():
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5],
    ])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1])

    edge_indices = unbatch_edge_index(edge_index, batch)
    assert edge_indices[0].tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert edge_indices[1].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
