"""
Model Utility Functions for Attention Processing and Token Sampling.

This module provides utility functions used during model inference:
- Attention map processing for memory efficiency
- Token sampling with temperature and top-k support

These functions are called from AttentionModel and AttentionModelNoSys
during token-by-token generation.

Memory Optimization Context:
----------------------------
During inference, attention maps can be very large:
- Full attention: [batch, heads, seq_len, seq_len]
- For a 4096 token sequence with 32 heads: 32 * 4096 * 4096 * 4 bytes = 2GB per layer!

The get_last_attn() function reduces this dramatically by keeping only the
attention FROM the last generated token, which is all we need for detection.

Token Sampling Context:
-----------------------
Instead of always picking the highest-probability token (greedy decoding),
we use top-k sampling for more natural text generation. This helps prevent
repetitive outputs while still being reasonably focused.
"""

import torch
import torch.nn.functional as F


def get_last_attn(attn_map):
    """
    Extract only the last token's attention row from full attention maps.
    
    This function dramatically reduces memory usage by discarding attention
    information we don't need. For prompt injection detection, we only care
    about where the LAST generated token is looking (its attention pattern),
    not the full bidirectional attention matrix.
    
    Memory Savings Example:
        - Input: [batch, heads, seq_len, seq_len] per layer
        - Output: [batch, heads, 1, seq_len] per layer
        
        For seq_len=4096, this is a 4096x reduction in the sequence dimension!
    
    Why Last Token Only:
        In autoregressive generation, each new token attends to all previous
        tokens. The last token's attention pattern shows where the model is
        "looking" when deciding what to generate next. This is the critical
        signal for detecting prompt injections - if the model attends more to
        injected instructions than the original system prompt, it's likely
        being manipulated.
    
    Args:
        attn_map (list): List of attention tensors, one per transformer layer.
                Each tensor has shape [batch, num_heads, seq_len, seq_len]
                where attn[b, h, i, j] is attention from token i to token j.
    
    Returns:
        list: Modified attention map list where each tensor now has shape
              [batch, num_heads, 1, seq_len]. The "1" dimension preserves
              the structure for consistent indexing.
    
    Example:
        >>> # During inference in AttentionModel
        >>> attention_map = [attn.detach().cpu() for attn in output['attentions']]
        >>> attention_map = get_last_attn(attention_map)
        >>> print(attention_map[0].shape)  # Layer 0
        torch.Size([1, 32, 1, 512])  # Was [1, 32, 512, 512]
    
    Note:
        This function modifies the input list IN PLACE and also returns it.
        Each layer's tensor is replaced with its sliced version.
    
    Implementation Details:
        - layer[:, :, -1, :] extracts the last row (attention FROM last token)
        - .unsqueeze(2) adds back the sequence dimension for consistent shape
    """
    for i, layer in enumerate(attn_map):
        # layer shape: [batch, heads, seq_len, seq_len]
        # layer[:, :, -1, :] extracts: [batch, heads, seq_len] (last token's attention to all tokens)
        # .unsqueeze(2) adds dimension: [batch, heads, 1, seq_len] (maintains 4D structure)
        attn_map[i] = layer[:, :, -1, :].unsqueeze(2)

    return attn_map


def sample_token(logits, top_k=None, top_p=None, temperature=1.0):
    """
    Sample a token from logits using temperature scaling and top-k filtering.
    
    This function implements controlled random sampling for text generation,
    providing a balance between determinism and creativity.
    
    Sampling Methods:
        1. **Temperature**: Scales logits before softmax
           - temp < 1.0: Sharper distribution (more deterministic)
           - temp = 1.0: Standard distribution
           - temp > 1.0: Flatter distribution (more random)
        
        2. **Top-k**: Only consider the k highest-probability tokens
           - Prevents sampling from unlikely tokens
           - k=50 is a common default
        
        3. **Top-p** (nucleus): Not implemented, but parameter is reserved
           - Would keep tokens until cumulative probability reaches p
    
    Sampling Process:
        1. Apply temperature scaling: logits = logits / temperature
        2. Select top-k tokens by logit value
        3. Convert top-k logits to probabilities (softmax)
        4. Sample from these probabilities (multinomial)
        5. Map back to original vocabulary indices
    
    Args:
        logits (torch.Tensor): Raw model output logits for the next token.
                Shape: [vocab_size] (1D tensor for single position)
        top_k (int, optional): Number of top tokens to consider.
                If None or not provided, uses greedy decoding (argmax).
        top_p (float, optional): Nucleus sampling threshold (NOT IMPLEMENTED).
                Reserved for future expansion.
        temperature (float): Temperature for scaling logits. Default 1.0.
                Lower = more deterministic, Higher = more random.
    
    Returns:
        torch.Tensor: Selected token ID.
                Shape: [1] if top_k is used, scalar if greedy decoding.
    
    Example:
        >>> logits = model_output.logits[0, -1, :]  # Last position logits
        >>> next_token = sample_token(logits, top_k=50, temperature=1.0)
        >>> print(f"Selected token ID: {next_token.item()}")
    
    Behavior:
        - With top_k: Returns [1] tensor from multinomial sampling
        - Without top_k: Returns scalar tensor from argmax (greedy)
    
    Why Not Always Greedy:
        Greedy decoding (always picking the highest-probability token) can lead
        to repetitive and boring text. Top-k sampling introduces controlled
        randomness while still focusing on likely tokens.
    """
    # -----------------------------------------------------------------
    # STEP 1: Temperature Scaling
    # -----------------------------------------------------------------
    # Divide logits by temperature to control distribution sharpness
    # Higher temperature = flatter distribution = more randomness
    # Lower temperature = sharper distribution = more deterministic
    logits = logits / temperature

    # -----------------------------------------------------------------
    # STEP 2: Top-k Sampling
    # -----------------------------------------------------------------
    if top_k is not None:
        # Ensure top_k doesn't exceed vocabulary size
        top_k = min(top_k, logits.size(-1))
        
        # Get the top-k logits and their indices in the vocabulary
        # values: the actual logit values (sorted descending)
        # indices: their positions in the original vocabulary
        values, indices = torch.topk(logits, top_k)
        
        # Convert top-k logits to probabilities
        # Only these k tokens will have non-zero probability
        probs = F.softmax(values, dim=-1)
        
        # Sample from the probability distribution
        # multinomial returns an index into the `probs` tensor (0 to k-1)
        # We then use this to look up the actual vocabulary index
        sampled_idx = torch.multinomial(probs, 1)
        next_token_id = indices[sampled_idx]

        return next_token_id

    # -----------------------------------------------------------------
    # STEP 3: Greedy Decoding Fallback
    # -----------------------------------------------------------------
    # If top_k is not specified, simply return the highest-probability token
    # This is deterministic - same input always produces same output
    return logits.argmax(dim=-1).squeeze()
