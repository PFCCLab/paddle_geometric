from typing import Any, Dict

import numpy as np
import paddle

from paddle_geometric.data import Data
from paddle_geometric.utils import remove_self_loops
from paddle_geometric.utils import to_undirected as to_undirected_fn

# @finshed
def read_npz(path: str, to_undirected: bool = True) -> Data:
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    import scipy.sparse as sp

    x = sp.csr_matrix(
        (f["attr_data"], f["attr_indices"], f["attr_indptr"]), f["attr_shape"]
    ).todense()
    x = paddle.to_tensor(data=x).to("float32")
    x[x > 0] = 1
    adj = sp.csr_matrix(
        (f["adj_data"], f["adj_indices"], f["adj_indptr"]), f["adj_shape"]
    ).tocoo()
    row = paddle.to_tensor(data=adj.row).to("int64")
    col = paddle.to_tensor(data=adj.col).to("int64")
    edge_index = paddle.stack(x=[row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])
    y = paddle.to_tensor(data=f["labels"]).to("int64")
    return Data(x=x, edge_index=edge_index, y=y)
