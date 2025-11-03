from math import sqrt

import paddle

from paddle_geometric.testing import withPackage
from paddle_geometric.utils import geodesic_distance


@withPackage('gdist')
def test_geodesic_distance():
    pos = paddle.to_tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ])
    face = paddle.to_tensor([[0, 1, 3], [0, 2, 3]]).t()

    out = geodesic_distance(pos, face)
    expected = paddle.to_tensor([
        [0.0, 1.0, 1.0, sqrt(2)],
        [1.0, 0.0, sqrt(2), 1.0],
        [1.0, sqrt(2), 0.0, 1.0],
        [sqrt(2), 1.0, 1.0, 0.0],
    ])
    assert paddle.allclose(out, expected)
    assert paddle.allclose(out, geodesic_distance(pos, face, num_workers=-1))

    out = geodesic_distance(pos, face, norm=False)
    expected = paddle.to_tensor([
        [0, 2, 2, 2 * sqrt(2)],
        [2, 0, 2 * sqrt(2), 2],
        [2, 2 * sqrt(2), 0, 2],
        [2 * sqrt(2), 2, 2, 0],
    ])
    assert paddle.allclose(out, expected)

    src = paddle.to_tensor([0, 0, 0, 0])
    dst = paddle.to_tensor([0, 1, 2, 3])
    out = geodesic_distance(pos, face, src=src, dst=dst)
    expected = paddle.to_tensor([0.0, 1.0, 1.0, sqrt(2)])
    assert paddle.allclose(out, expected)

    out = geodesic_distance(pos, face, dst=dst)
    expected = paddle.to_tensor([0.0, 0.0, 0.0, 0.0])
    assert paddle.allclose(out, expected)
