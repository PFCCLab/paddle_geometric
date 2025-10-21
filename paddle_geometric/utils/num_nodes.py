from copy import copy
from typing import Dict, Optional, Tuple, Union

import paddle
from paddle import Tensor

import paddle_geometric
from paddle_geometric import EdgeIndex
from paddle_geometric.paddle_utils import *  # noqa
from paddle_geometric.typing import EdgeType, NodeType, SparseTensor


# Function to calculate the number of nodes based on edge_index input
def maybe_num_nodes(
    edge_index: Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor],
                      SparseTensor],
    num_nodes: Optional[int] = None,
) -> int:
    """This function calculates the number of nodes in the graph based on the provided edge_index.
    The function handles different input types (Tensor or tuple) and returns the maximum node
    index, considering both rows and columns in the edge_index.

    Args:
        edge_index (Union[Tensor, Tuple[Tensor, Tensor]]): Edge index, could be a Tensor or a tuple of Tensors.
        num_nodes (Optional[int]): Optional number of nodes, if provided, will be returned directly.

    Returns:
        int: The calculated number of nodes in the graph.
    """
    # If num_nodes is explicitly provided, return it
    if num_nodes is not None:
        return num_nodes

    elif isinstance(edge_index, EdgeIndex):
        return max(edge_index.get_sparse_size())

    # If edge_index is a Tensor
    elif isinstance(edge_index, Tensor):
        # If the edge_index is sparse, the number of nodes is the maximum of row and column sizes
        if paddle_geometric.utils.is_paddle_sparse_tensor(edge_index):
            return max(edge_index.shape[0], edge_index.shape[1])
        return int(
            edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

    # If edge_index is a tuple (e.g., (row, col))
    elif isinstance(edge_index, tuple):
        return max(
            int(edge_index[0]._max()) + 1 if edge_index[0].size > 0 else 0,
            int(edge_index[1]._max()) + 1 if edge_index[1].size > 0 else 0,
        )
    elif isinstance(edge_index, SparseTensor):
        return max(edge_index.shape[0], edge_index.shape[1])
    # If edge_index is not a supported type, raise an error
    raise NotImplementedError(
        "edge_index must be a Tensor or tuple of Tensors")


# Function to calculate the number of nodes for each type in a dictionary of edge indices
def maybe_num_nodes_dict(
    edge_index_dict: Dict[EdgeType, paddle.Tensor],
    num_nodes_dict: Optional[Dict[NodeType, int]] = None,
) -> Dict[NodeType, int]:
    """This function calculates the number of nodes for each type in a dictionary of edge indices.
    It iterates over the dictionary, computes the maximum node index for each edge type, and updates
    the num_nodes_dict.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Dictionary of edge indices with keys as node types.
        num_nodes_dict (Optional[Dict[str, int]]): Optional dictionary of pre-existing node counts for each type.

    Returns:
        Dict[str, int]: Updated dictionary of node counts for each edge type.
    """
    num_nodes_dict = {} if num_nodes_dict is None else copy(num_nodes_dict)

    # List of types already present in num_nodes_dict
    found_types = list(num_nodes_dict.keys())

    # Iterate over the edge_index_dict
    for keys, edge_index in edge_index_dict.items():
        # Process the first key (node type)
        key = keys[0]
        if key not in found_types:
            # Calculate the maximum node index for the first key (node type)
            N = int(edge_index[0].max().item() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        # Process the last key (node type)
        key = keys[-1]
        if key not in found_types:
            # Calculate the maximum node index for the last key (node type)
            N = int(edge_index[1].max().item() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

    return num_nodes_dict
