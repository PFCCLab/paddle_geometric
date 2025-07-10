import paddle

from paddle_geometric.data import Data

try:
    import openmesh
except ImportError:
    openmesh = None

# @finshed
def read_ply(path: str) -> Data:
    if openmesh is None:
        raise ImportError("`read_ply` requires the `openmesh` package.")
    mesh = openmesh.read_trimesh(path)
    pos = paddle.to_tensor(data=mesh.points()).to("float32")
    face = paddle.to_tensor(data=mesh.face_vertex_indices())
    face = face.t().to("int64").contiguous()
    return Data(pos=pos, face=face)
