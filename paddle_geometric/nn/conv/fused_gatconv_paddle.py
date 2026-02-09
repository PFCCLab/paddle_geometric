"""
Paddle implementation of FusedGATConv - Pure Python version
This provides a CPU-compatible fallback implementation for PaddlePaddle
Based on dgNN's CUDA implementation logic
"""
import paddle
from paddle import Tensor


def GATConvFuse(attn_row, attn_col, row_ptr, col_ind, col_ptr, row_ind, permute,
                 negative_slope, in_feat, attn_drop):
    """
    Pure Python implementation of GATConvFuse for PaddlePaddle.
    This implementation closely follows the dgNN CUDA kernel logic.

    Args:
        attn_row: Attention coefficients for destination nodes [num_nodes, num_heads]
        attn_col: Attention coefficients for source nodes [num_nodes, num_heads]
        row_ptr: Row pointer for CSR format [num_nodes + 1] (NOT USED, we use CSC format)
        col_ind: Column indices for CSR format [num_edges] (NOT USED, we use CSC format)
        col_ptr: Column pointer for CSC format [num_nodes + 1] (USED: this is actually rowptr for dst nodes)
        row_ind: Row indices for CSC format [num_edges] (USED: this is actually col for src nodes)
        permute: Permutation tensor [num_edges] (NOT USED in forward)
        negative_slope: Negative slope for LeakyReLU
        in_feat: Input features [num_nodes, num_heads, out_channels]
        attn_drop: Dropout probability

    Returns:
        Output features after GAT convolution [num_nodes, num_heads * out_channels]
    """
    # dgNN uses a special CSR format where rows represent destination nodes
    # and columns represent source nodes.
    # However, PyTorch Geometric's to_graph_format generates:
    # - CSR format: rows=source nodes, cols=destination nodes
    # - CSC format: rows=source nodes, cols=destination nodes (but with col_ptr for dst nodes)
    #
    # The key insight: dgNN needs destination-node-perspective CSR format,
    # which is what PyTorch Geometric provides as (col_ptr, row_ind) in the CSC format!
    #
    # So we use col_ptr as row_ptr and row_ind as col_ind

    num_nodes = col_ptr.shape[0] - 1
    num_edges = row_ind.shape[0]
    num_heads, out_channels = attn_col.shape[1], in_feat.shape[2]

    # Initialize output
    out = paddle.zeros([num_nodes, num_heads * out_channels], dtype=in_feat.dtype)

    # Process each destination node
    for dst_node in range(num_nodes):
        start, end = col_ptr[dst_node], col_ptr[dst_node + 1]

        if start >= end:
            continue  # No incoming edges

        # Get all source nodes (neighbors)
        src_nodes = row_ind[start:end]  # [num_neighbors]
        num_neighbors = src_nodes.shape[0]

        # Process each head independently
        for head in range(num_heads):
            # Get attention coefficients for this destination node
            attn_row_val = attn_row[dst_node, head]  # scalar

            # Step 1: Compute attention scores for all source nodes
            # e_ij = attn_row[dst] + attn_col[src]
            attn_col_vals = attn_col[src_nodes, head]  # [num_neighbors]
            attention_scores = attn_row_val + attn_col_vals  # [num_neighbors]

            # Step 2: Apply LeakyReLU
            attention_scores = paddle.nn.functional.leaky_relu(
                attention_scores, negative_slope)  # [num_neighbors]

            # Step 3: Compute softmax over source nodes (numerically stable)
            # Find max for numerical stability
            max_score = paddle.max(attention_scores)
            exp_scores = paddle.exp(attention_scores - max_score)  # [num_neighbors]
            sum_exp = paddle.sum(exp_scores)  # scalar

            # Step 4: Normalize to get attention weights
            attention_weights = exp_scores / sum_exp  # [num_neighbors]

            # Step 5: Apply dropout if training
            if attn_drop > 0:
                # Create dropout mask
                dropout_mask = (paddle.uniform([num_neighbors]) > attn_drop).cast('float32')
                # Rescale attention weights: w / (1 - dropout_rate)
                attention_weights = attention_weights * dropout_mask / (1.0 - attn_drop)

            # Step 6: Aggregate features using attention weights
            src_features = in_feat[src_nodes, head, :]  # [num_neighbors, out_channels]

            # Weighted sum: sum_i (alpha_i * h_i)
            weighted_sum = paddle.sum(
                attention_weights.unsqueeze(-1) * src_features, axis=0)  # [out_channels]

            # Store in output
            out[dst_node, head * out_channels:(head + 1) * out_channels] = weighted_sum

    return out


def GATConvFuse_inference(attn_row, attn_col, row_ptr, col_ind, negative_slope, in_feat):
    """
    Inference-only version of GATConvFuse without dropout.

    Args:
        attn_row: Attention coefficients for destination nodes [num_nodes, num_heads]
        attn_col: Attention coefficients for source nodes [num_nodes, num_heads]
        row_ptr: Row pointer for CSR format [num_nodes + 1]
        col_ind: Column indices for CSR format [num_edges]
        negative_slope: Negative slope for LeakyReLU
        in_feat: Input features [num_nodes, num_heads, out_channels]

    Returns:
        Output features after GAT convolution [num_nodes, num_heads * out_channels]
    """
    # Inference version without dropout
    # We need to provide dummy parameters for the unused arguments
    col_ptr = paddle.zeros([1], dtype='int32')  # dummy
    row_ind = paddle.zeros([1], dtype='int32')  # dummy
    permute = paddle.zeros([1], dtype='int32')  # dummy

    return GATConvFuse(attn_row, attn_col, row_ptr, col_ind, col_ptr, row_ind, permute,
                      negative_slope, in_feat, attn_drop=0.0)