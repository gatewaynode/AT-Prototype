"""
Attention Processing Utilities for Prompt Injection Detection.

This module provides utility functions for processing transformer attention maps
and computing focus scores. The functions transform raw attention tensors into
interpretable metrics that indicate where the model is directing its attention.

Key Concepts:
-------------
- **Attention Map**: A tensor showing how much each token attends to every other
  token. Shape varies by model but typically [batch, heads, seq_len, seq_len].
  
- **Heatmap**: A 2D array [num_layers, num_heads] where each cell contains the
  aggregated attention score from the last generated token to the instruction
  tokens for that specific layer/head combination.
  
- **Focus Score**: A scalar value representing the average attention across
  selected important heads. Higher = more attention on instruction (safe),
  Lower = attention diverted (potential injection).

Data Flow:
----------
    Raw Attention Maps → process_attn() → Heatmap → calc_attn_score() → Focus Score
    
    1. process_attn(): Extracts and normalizes attention to instruction tokens
    2. calc_attn_score(): Averages heatmap values for important heads

Token Range Concept:
--------------------
The `rng` parameter defines two ranges:
    - rng[0] = (inst_start, inst_end): Token indices of the instruction
    - rng[1] = (data_start, data_end): Token indices of the user data

These ranges are model-specific because different models have different chat
template formats with varying numbers of special tokens.

Example:
    For a Qwen model with input "System: Say hello | User: Data: test data"
    - Instruction tokens might be at indices 3-10
    - Data tokens might be at indices -12 to -5 (relative to end)
"""

import torch 
import numpy as np


def process_attn(attention, rng, attn_func):
    """
    Process attention maps to create a heatmap of attention-to-instruction scores.
    
    This function extracts how much the last generated token attends to the
    instruction tokens, aggregated across all layers and heads. The result is
    a 2D heatmap that can be used to identify which heads focus on instructions
    vs. which heads might be "distracted" by data content.
    
    The processing supports different aggregation methods:
        - "sum": Sum attention weights to all instruction tokens
        - "max": Take maximum attention weight to any instruction token
        - "normalize": Divide by total attention to instruction + data
    
    Args:
        attention (list): List of attention tensors, one per layer.
                Each tensor has shape [batch, num_heads, seq_len, seq_len].
                The attention[layer][batch, head, i, j] represents how much
                token i attends to token j at that layer/head.
                
        rng (tuple): Two tuples defining token position ranges:
                ((inst_start, inst_end), (data_start, data_end))
                - First tuple: Indices of instruction tokens
                - Second tuple: Indices of data tokens (can be negative for end-relative)
                
        attn_func (str): Aggregation function specifier. Supports combinations:
                - "sum" or "normalize_sum": Sum attention to instruction tokens
                - "max" or "normalize_max": Max attention to instruction tokens
                - "normalize" prefix: Normalize by total attention to inst+data
    
    Returns:
        np.ndarray: Heatmap of shape [num_layers, num_heads].
                Each cell [l, h] contains the aggregated attention score
                from the last generated token to instruction tokens
                for layer l, head h.
    
    Example:
        >>> attention = model_output['attentions']  # List of layer attention tensors
        >>> rng = ((3, 10), (-15, -5))  # instruction tokens 3:10, data tokens -15:-5
        >>> heatmap = process_attn(attention, rng, "normalize_sum")
        >>> print(heatmap.shape)  # (num_layers, num_heads)
        (28, 32)
        >>> print(heatmap[0, 0])  # Attention score for layer 0, head 0
        0.234
    
    How It Works:
        1. For each layer's attention tensor:
           a. Convert to numpy for efficient computation
           b. Extract last token's attention to instruction range: attn[0, :, -1, inst_start:inst_end]
           c. Aggregate using sum or max across instruction tokens
           d. Optionally normalize by total attention to instruction + data
        2. Store results in heatmap array
        3. Replace any NaN values with 0 for stability
    """
    # Initialize heatmap: rows = layers, columns = attention heads
    # Shape is [num_layers, num_heads] to store one score per layer/head combination
    heatmap = np.zeros((len(attention), attention[0].shape[1]))
    
    for i, attn_layer in enumerate(attention):
        # Convert from PyTorch tensor to numpy for efficient array operations
        # Use float32 for precision during calculations
        attn_layer = attn_layer.to(torch.float32).numpy()

        # ---------------------------------------------------------------------
        # STEP 1: Extract attention from last token to instruction tokens
        # ---------------------------------------------------------------------
        # attn_layer shape: [batch=1, num_heads, seq_len, seq_len]
        # We want: attention FROM the last generated token (-1 in dim 2)
        #          TO the instruction tokens (rng[0][0]:rng[0][1] in dim 3)
        # Result shape: [num_heads,] - one value per head
        
        if "sum" in attn_func:
            # Sum all attention weights to instruction tokens
            # This captures total attention directed at the instruction
            # Shape: [num_heads,] after summing over instruction token dimension
            last_token_attn_to_inst = np.sum(
                attn_layer[0, :, -1, rng[0][0]:rng[0][1]],  # [heads, inst_tokens]
                axis=1  # Sum over instruction tokens
            )
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            # Take maximum attention weight to any instruction token
            # This captures peak attention, useful if attention is concentrated
            # Shape: [num_heads,] after taking max over instruction token dimension
            last_token_attn_to_inst = np.max(
                attn_layer[0, :, -1, rng[0][0]:rng[0][1]],  # [heads, inst_tokens]
                axis=1  # Max over instruction tokens
            )
            attn = last_token_attn_to_inst

        else:
            raise NotImplementedError(f"Unknown attention function: {attn_func}")
        
        # ---------------------------------------------------------------------
        # STEP 2: Calculate denominators for normalization (if needed)
        # ---------------------------------------------------------------------
        # Always calculate both sums for potential normalization:
        # - Total attention to instruction tokens
        # - Total attention to data tokens
        
        # Sum of attention to instruction tokens (for normalization denominator)
        last_token_attn_to_inst_sum = np.sum(
            attn_layer[0, :, -1, rng[0][0]:rng[0][1]], 
            axis=1
        )  # Shape: [num_heads,]
        
        # Sum of attention to data tokens (for normalization denominator)
        # Note: rng[1] can have negative indices for end-relative positions
        last_token_attn_to_data_sum = np.sum(
            attn_layer[0, :, -1, rng[1][0]:rng[1][1]], 
            axis=1
        )  # Shape: [num_heads,]

        # ---------------------------------------------------------------------
        # STEP 3: Apply normalization if specified
        # ---------------------------------------------------------------------
        if "normalize" in attn_func:
            # Normalize by total attention to both instruction AND data tokens
            # This gives us the PROPORTION of attention on instruction vs data
            # Result is in [0, 1] where higher = more focus on instruction
            # 
            # epsilon prevents division by zero when total attention is near 0
            epsilon = 1e-8
            heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            # No normalization: use raw attention values
            heatmap[i, :] = attn

    # Replace any NaN values with 0 for numerical stability
    # NaN can occur if attention values are undefined or if division resulted in inf
    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap


def calc_attn_score(heatmap, heads):
    """
    Calculate the final attention focus score from a heatmap.
    
    This function extracts attention values for specific "important" heads
    from the heatmap and computes their mean. The important heads are
    pre-selected per model architecture based on their discriminative power
    for detecting prompt injections.
    
    Not all attention heads are equally useful for detection. Some heads
    consistently show different attention patterns for legitimate vs.
    injection inputs, while others don't. This function focuses only on
    the heads that matter.
    
    Args:
        heatmap (np.ndarray): 2D array of shape [num_layers, num_heads]
                as returned by process_attn(). Each cell contains the
                attention score for that layer/head combination.
                
        heads (list): List of [layer, head] pairs specifying which
                attention heads to include in the score calculation.
                Example: [[5, 18], [7, 12], [9, 29], [17, 2]]
                These are typically determined by running select_head.py
                to identify heads with strong injection detection signal.
    
    Returns:
        float: Mean attention score across the specified important heads.
               Higher scores indicate attention is focused on instruction (safe).
               Lower scores indicate attention diverted to data (potential injection).
    
    Example:
        >>> heatmap = process_attn(attention_maps, ranges, "normalize_sum")
        >>> important_heads = [[5, 18], [7, 12], [9, 29]]
        >>> score = calc_attn_score(heatmap, important_heads)
        >>> print(f"Focus score: {score:.4f}")
        Focus score: 0.5823
    
    Why Important Heads Matter:
        In a 32-layer model with 32 heads each, there are 1024 layer/head
        combinations. Most of these don't show meaningful differences between
        normal inputs and injections. By focusing on 10-50 carefully selected
        heads, we get a much cleaner signal with less noise.
        
        The heads are selected using select_head.py, which tests each head's
        ability to distinguish prompt injections from legitimate inputs.
    """
    # Extract attention values for each important head and compute their mean
    # Using list comprehension to gather values: heatmap[layer, head] for each (layer, head) pair
    # np.mean() then averages these values into a single focus score
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    
    return score
