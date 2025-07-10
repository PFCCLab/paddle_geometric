import re
from typing import List

import paddle
from paddle import Tensor
from paddle_geometric.data import Data
from paddle_geometric.io import parse_txt_array

# @finshed
def parse_off(src: List[str]) -> Data:
    if src[0] == "OFF":
        src = src[1:]
    else:
        src[0] = src[0][3:]
    num_nodes, num_faces = (int(item) for item in src[0].split()[:2])
    pos = parse_txt_array(src[1 : 1 + num_nodes])
    face = face_to_tri(src[1 + num_nodes : 1 + num_nodes + num_faces])
    data = Data(pos=pos)
    data.face = face
    return data


def face_to_tri(face: List[str]) -> paddle.Tensor:
    face_index = [[int(x) for x in line.strip().split()] for line in face]
    triangle = paddle.to_tensor(data=[line[1:] for line in face_index if line[0] == 3])
    triangle = triangle.to("int64")
    rect = paddle.to_tensor(data=[line[1:] for line in face_index if line[0] == 4])
    rect = rect.to("int64")
    if rect.size > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
        return paddle.concat(x=[triangle, first, second], axis=0).t().contiguous()
    return triangle.t().contiguous()


def read_off(path: str) -> Data:
    """Reads an OFF (Object File Format) file, returning both the position of
    nodes and their connectivity in a :class:`torch_geometric.data.Data`
    object.

    Args:
        path (str): The path to the file.
    """
    with open(path) as f:
        src = f.read().split("\n")[:-1]
    return parse_off(src)


def write_off(data: Data, path: str) -> None:
    r"""Writes a Data object to an OFF (Object File Format) file.

    Args:
        data (Data): A Data object containing node positions and connectivity.
        path (str): The path to the file.
    """
    assert data.pos is not None
    assert data.face is not None

    num_nodes, num_faces = data.pos.shape[0], data.face.shape[1]

    pos = data.pos.astype('float32')
    face = data.face.transpose([1, 0])
    num_vertices = paddle.full((num_faces, 1), face.shape[1], dtype='int64')
    face = paddle.concat([num_vertices, face], axis=-1)

    # Format positions and face data for writing
    pos_repr = re.sub(',', '', paddle.to_string(pos))
    pos_repr = '\n'.join([x.strip() for x in pos_repr.split('\n')])[:-1]

    face_repr = re.sub(',', '', paddle.to_string(face))
    face_repr = '\n'.join([x.strip() for x in face_repr.split('\n')])[:-1]

    with open(path, 'w') as f:
        f.write(f'OFF\n{num_nodes} {num_faces} 0\n')
        f.write(pos_repr)
        f.write('\n')
        f.write(face_repr)
        f.write('\n')
