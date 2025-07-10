import os.path as osp
from typing import Dict, List, Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.io import fs, read_txt_array
from paddle_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes',
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]
# @finshed

def read_tu_data(
    folder: str, prefix: str
) -> Tuple[Data, Dict[str, paddle.Tensor], Dict[str, int]]:
    files = fs.glob(os.path.join(folder, f"{prefix}_*.txt"))
    names = [os.path.basename(f)[len(prefix) + 1 : -4] for f in files]
    edge_index = read_file(folder, prefix, "A", "int64").t() - 1
    batch = read_file(folder, prefix, "graph_indicator", "int64") - 1
    node_attribute = paddle.empty(shape=(batch.shape[0], 0))
    if "node_attributes" in names:
        node_attribute = read_file(folder, prefix, "node_attributes")
        if node_attribute.dim() == 1:
            node_attribute = node_attribute.unsqueeze(axis=-1)
    node_label = paddle.empty(shape=(batch.shape[0], 0))
    if "node_labels" in names:
        node_label = read_file(folder, prefix, "node_labels", "int64")
        if node_label.dim() == 1:
            node_label = node_label.unsqueeze(axis=-1)
        node_label = node_label - (node_label.min(axis=0), node_label.argmin(axis=0))[0]
        node_labels = list(node_label.unbind(axis=-1))
        node_labels = [one_hot(x) for x in node_labels]
        if len(node_labels) == 1:
            node_label = node_labels[0]
        else:
            node_label = paddle.concat(x=node_labels, axis=-1)
    edge_attribute = paddle.empty(shape=(edge_index.shape[1], 0))
    if "edge_attributes" in names:
        edge_attribute = read_file(folder, prefix, "edge_attributes")
        if edge_attribute.dim() == 1:
            edge_attribute = edge_attribute.unsqueeze(axis=-1)
    edge_label = paddle.empty(shape=(edge_index.shape[1], 0))
    if "edge_labels" in names:
        edge_label = read_file(folder, prefix, "edge_labels", "int64")
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(axis=-1)
        edge_label = edge_label - (edge_label.min(axis=0), edge_label.argmin(axis=0))[0]
        edge_labels = list(edge_label.unbind(axis=-1))
        edge_labels = [one_hot(e) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_label = edge_labels[0]
        else:
            edge_label = paddle.concat(x=edge_labels, axis=-1)
    x = cat([node_attribute, node_label])
    edge_attr = cat([edge_attribute, edge_label])
    y = None
    if "graph_attributes" in names:
        y = read_file(folder, prefix, "graph_attributes")
    elif "graph_labels" in names:
        y = read_file(folder, prefix, "graph_labels", "int64")
        _, y = y.unique(return_inverse=True)
    num_nodes = int(edge_index._max()) + 1 if x is None else x.shape[0]
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)
    sizes = {
        "num_node_attributes": node_attribute.shape[-1],
        "num_node_labels": node_label.shape[-1],
        "num_edge_attributes": edge_attribute.shape[-1],
        "num_edge_labels": edge_label.shape[-1],
    }
    return data, slices, sizes


def read_file(
    folder: str, prefix: str, name: str, dtype: Optional[paddle.dtype] = None
) -> paddle.Tensor:
    path = os.path.join(folder, f"{prefix}_{name}.txt")
    return read_txt_array(path, sep=",", dtype=dtype)


def cat(seq: List[Optional[paddle.Tensor]]) -> Optional[paddle.Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.size > 0]
    values = [(v.unsqueeze(axis=-1) if v.dim() == 1 else v) for v in values]
    return paddle.concat(x=values, axis=-1) if len(values) > 0 else None


def split(data: Data, batch: paddle.Tensor) -> Tuple[Data, Dict[str, paddle.Tensor]]:
    node_slice = cumsum(paddle.bincount(x=batch))
    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(paddle.bincount(x=batch[row]))
    data.edge_index -= node_slice[batch[row]].unsqueeze(axis=0)
    slices = {"edge_index": edge_slice}
    if data.x is not None:
        slices["x"] = node_slice
    else:
        data._num_nodes = paddle.bincount(x=batch).tolist()
        data.num_nodes = batch.size
    if data.edge_attr is not None:
        slices["edge_attr"] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, paddle.Tensor)
        if data.y.size(0) == batch.shape[0]:
            slices["y"] = node_slice
        else:
            slices["y"] = paddle.arange(start=0, end=int(batch[-1]) + 2, dtype="int64")
    return data, slices
