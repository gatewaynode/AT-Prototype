"""
Attention Model Without System Message Support.

This module provides the AttentionModelNoSys class for language models that
do NOT support system messages in their chat templates. Some models like
Gemma2 only support user/assistant message roles, requiring a different
message formatting approach.

Key Differences from AttentionModel:
------------------------------------
1. **No System Message**: Instead of separate system/user roles, the instruction
   is prepended directly to the user message content.
   
2. **Message Format**: 
   - AttentionModel: [{"role": "system", ...}, {"role": "user", ...}]
   - AttentionModelNoSys: [{"role": "user", "content": instruction + data}]

3. **Token Position Calculation**: Different offsets due to simpler chat template
   structure without system message delimiters.

Supported Models:
-----------------
- Gemma2 (google/gemma-2-9b-it) - Uses provider "attn-hf-no-sys"

To add a new model without system message support:
1. Add model config with "provider": "attn-hf-no-sys"
2. Determine token position offsets (data_range) for the new model
3. Add a conditional branch in the inference() method

Usage:
------
    from models.attn_model_nsys import AttentionModelNoSys
    
    config = {
        "model_info": {"name": "gemma2_9b-attn", "model_id": "google/gemma-2-9b-it"},
        "params": {"max_output_tokens": 32, "important_heads": [[10, 11], ...]}
    }
    model = AttentionModelNoSys(config)
    text, tokens, attns, input_toks, ranges, probs = model.inference(
        "Summarize this:", "Some text to summarize"
    )
"""

import torch
from .model import Model
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import sample_token

# Automatically select GPU if available, otherwise fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionModelNoSys(Model):
    """
    Language model wrapper for attention-based prompt injection detection.
    
    This variant is for models that do NOT support system messages in their
    chat templates. The instruction and data are combined into a single user
    message instead of using separate system/user roles.
    
    The model performs token-by-token generation while extracting attention
    maps at each step. These attention maps are then analyzed by the
    AttentionDetector to identify potential prompt injection attempts.
    
    Attributes:
        name (str): Model identifier from config, used for conditional logic
        max_output_tokens (int): Maximum tokens to generate per inference
        tokenizer: HuggingFace tokenizer for the model
        model: HuggingFace CausalLM model with attention output capability
        important_heads (list): List of [layer, head] pairs for detection
        top_k (int): Top-k sampling parameter (default 50)
        top_p (float): Nucleus sampling parameter (default None = disabled)
    
    Architecture Support:
        - Gemma2: Tested with google/gemma-2-9b-it
    
    Memory Optimization:
        - Uses bfloat16 precision to reduce model memory footprint
        - Attention maps are immediately detached and moved to CPU
        - Converted to half precision (fp16) for storage efficiency
        - torch.no_grad() prevents gradient accumulation
    
    Example:
        >>> model = AttentionModelNoSys(config)
        >>> generated_text, output_tokens, attention_maps, input_tokens, data_range, probs = \\
        ...     model.inference("Say hello", "test input", max_output_tokens=5)
        >>> print(f"Generated: {generated_text}")
        >>> print(f"Attention maps shape: {len(attention_maps)} tokens x {len(attention_maps[0])} layers")
    """
    
    def __init__(self, config):
        """
        Initialize the model with configuration settings.
        
        Loads the tokenizer and model from HuggingFace, configures precision
        settings, and sets up the important attention heads for detection.
        
        Args:
            config (dict): Configuration dictionary with structure:
                {
                    "model_info": {
                        "provider": "attn-hf-no-sys",
                        "name": "model_name",
                        "model_id": "huggingface/model-path"
                    },
                    "params": {
                        "temperature": 0.1,
                        "max_output_tokens": 32,
                        "important_heads": [[layer, head], ...] or "all"
                    }
                }
        
        Configuration Options:
            - important_heads: List of [layer, head] pairs to analyze, or
              "all" to use every head (slower but useful for head selection)
            - max_output_tokens: Limits generation length
            - model_id: HuggingFace model identifier or local path
        
        Model Loading Options:
            - torch_dtype=bfloat16: Reduces memory usage while maintaining precision
            - device_map: Automatically places model on available device
            - attn_implementation="eager": Required for attention map extraction
              (Flash Attention does not support attention output)
        """
        super().__init__(config)
        
        # Store model name for conditional logic (e.g., data_range calculation)
        self.name = config["model_info"]["name"]
        
        # Maximum tokens to generate during inference
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
        model_id = config["model_info"]["model_id"]
        
        # Load tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with optimizations for attention extraction
        # NOTE: attn_implementation="eager" is REQUIRED - Flash Attention cannot output attention maps
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Half precision for memory efficiency
            device_map=device,            # Auto-place on best available device
            attn_implementation="eager"   # Required for attention map output
        ).eval()  # Set to evaluation mode (disables dropout, etc.)
        
        # Configure important attention heads
        # "all" means use every head - useful for initial head selection analysis
        # Specific list means use only pre-selected discriminative heads
        if config["params"]["important_heads"] == "all":
            # Dynamically determine model dimensions by running a test inference
            attn_size = self.get_map_dim()
            # Create list of all [layer, head] combinations
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]
        
        # Sampling parameters for token generation
        self.top_k = 50    # Consider only top 50 tokens when sampling
        self.top_p = None  # Nucleus sampling disabled by default

    def get_map_dim(self):
        """
        Determine attention map dimensions by running a test inference.
        
        This is called when important_heads is set to "all" to discover
        the model's layer count and heads per layer dynamically.
        
        Returns:
            tuple: (num_layers, num_heads_per_layer)
        
        Note:
            This runs a short inference which adds startup time, but ensures
            correct dimensions for any supported model architecture.
        """
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]  # Get first token's attention
        # len(attention_map) = number of layers
        # attention_map[0].shape[1] = number of heads in first layer
        return len(attention_map), attention_map[0].shape[1]

    def inference(self, instruction, data, max_output_tokens=None):
        """
        Run inference and extract attention maps for each generated token.
        
        This method performs token-by-token generation while capturing the
        attention patterns at each step. The attention maps are essential
        for detecting prompt injection attempts.
        
        Message Format (No System Message):
            Since models like Gemma2 don't support system messages, we combine
            the instruction and data into a single user message:
            
            [{"role": "user", "content": "{instruction}\\nData: {data}"}]
            
            This differs from AttentionModel which uses:
            [{"role": "system", "content": instruction}, {"role": "user", "content": data}]
        
        Args:
            instruction (str): The task instruction (what would normally be
                      the system prompt). Placed at the start of user message.
            data (str): User input data to process. This is where potential
                      prompt injections would be located.
            max_output_tokens (int, optional): Override the default token limit.
                      For injection detection, we typically only need 1 token
                      since the first token's attention is most indicative.
        
        Returns:
            tuple: Six-element tuple containing:
                - generated_text (str): The complete generated response
                - output_tokens (list[str]): Individual decoded tokens
                - attention_maps (list): Attention maps for each generated token.
                      Structure: [token_idx][layer][batch, heads, seq, seq]
                - input_tokens (list[str]): Tokenized input for debugging
                - data_range (tuple): Token position ranges for instruction and data:
                      ((inst_start, inst_end), (data_start, data_end))
                - generated_probs (list[float]): Probability of each generated token
        
        Token Position Calculation (data_range):
            Different models have different chat template formats with varying
            numbers of special tokens. The data_range specifies where instruction
            and data tokens are located in the sequence.
            
            For Gemma2 with format: "<bos><start_of_turn>user\\n{instruction}\\nData: {data}<end_of_turn>\\n<start_of_turn>model\\n"
            - Instruction tokens: positions 5 to 5+instruction_len
            - Data tokens: positions -4-data_len to -5 (relative to end)
        
        Memory Management:
            - torch.no_grad(): Prevents gradient computation/storage
            - Attention maps detached immediately after extraction
            - Converted to CPU and half precision to free GPU memory
            - NaN values replaced with 0 for numerical stability
        
        Example:
            >>> model = AttentionModelNoSys(config)
            >>> text, tokens, attns, in_toks, ranges, probs = model.inference(
            ...     "Summarize:", "The quick brown fox", max_output_tokens=10
            ... )
            >>> print(f"Generated: {text}")
            >>> print(f"Instruction range: {ranges[0]}, Data range: {ranges[1]}")
        """
        # ---------------------------------------------------------------------
        # STEP 1: Construct message without system role
        # ---------------------------------------------------------------------
        # Prepend "Data: " to the user data for consistent formatting
        data = "Data: " + data
        
        # Combine instruction and data into a single user message
        # (No system message - this is the key difference from AttentionModel)
        messages = [
            {"role": "user", "content": instruction + "\n" + data},
        ]

        # Apply the model's chat template to format the message
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,          # Return string, not token IDs
            add_generation_prompt=True  # Add assistant turn prefix
        )

        # ---------------------------------------------------------------------
        # STEP 2: Calculate token positions for instruction and data
        # ---------------------------------------------------------------------
        # These lengths are used to compute where instruction/data tokens
        # appear in the final tokenized sequence
        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        # Tokenize the formatted text for model input
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        
        # Keep reference to input tokens for debugging/visualization
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            model_inputs['input_ids'][0])

        # ---------------------------------------------------------------------
        # STEP 3: Set model-specific token position ranges
        # ---------------------------------------------------------------------
        # The data_range tuple defines where instruction and data tokens are:
        #   ((inst_start, inst_end), (data_start, data_end))
        #
        # These offsets are model-specific because different models have:
        # - Different special tokens (<bos>, <start_of_turn>, etc.)
        # - Different chat template structures
        #
        # Negative indices are relative to the end of the sequence
        
        if "gemma2_9b-attn" in self.name:
            # Gemma2 chat template structure:
            # <bos><start_of_turn>user\n{instruction}\nData: {data}<end_of_turn>\n<start_of_turn>model\n
            # 
            # Token breakdown (approximate):
            # [0]: <bos>
            # [1]: <start_of_turn>
            # [2]: user
            # [3]: \n
            # [4]: (start of instruction content)
            # After instruction: \nData: {data}
            # End tokens: <end_of_turn>\n<start_of_turn>model\n
            #
            # Instruction: starts at position 5, extends for instruction_len tokens
            # Data: ends at position -5 (before end tokens), starts at -4-data_len
            data_range = ((5, 5+instruction_len), (-4-data_len, -5))
        else:
            raise NotImplementedError(
                f"Token position mapping not implemented for model: {self.name}. "
                "Add a new branch with appropriate data_range offsets."
            )

        # ---------------------------------------------------------------------
        # STEP 4: Token-by-token generation with attention extraction
        # ---------------------------------------------------------------------
        generated_tokens = []
        generated_probs = []  # Track confidence for each generated token
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        # Store attention maps for each generated token
        attention_maps = []

        # Determine how many tokens to generate
        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        # Disable gradient computation for inference efficiency
        with torch.no_grad():
            for i in range(n_tokens):
                # ---------------------------------------------------------
                # Forward pass with attention output enabled
                # ---------------------------------------------------------
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True  # Critical: enables attention map extraction
                )

                # ---------------------------------------------------------
                # Sample next token using top-k sampling
                # ---------------------------------------------------------
                # Get logits for the last position only (next token prediction)
                logits = output.logits[:, -1, :]
                
                # Convert logits to probabilities for confidence tracking
                probs = F.softmax(logits, dim=-1)
                
                # Sample from top-k tokens (stochastic generation)
                # Using temperature=1.0 for standard sampling
                next_token_id = sample_token(
                    logits[0], top_k=self.top_k, top_p=None, temperature=1.0)[0]

                # Record the probability of the chosen token
                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                # Check for end-of-sequence token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # ---------------------------------------------------------
                # Update inputs for next iteration (autoregressive generation)
                # ---------------------------------------------------------
                # Append new token to input sequence
                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                
                # Extend attention mask to include new token
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

                # ---------------------------------------------------------
                # Extract and store attention maps (memory-efficient)
                # ---------------------------------------------------------
                # Detach from computation graph and move to CPU immediately
                # to free GPU memory for the next forward pass
                attention_map = [attention.detach().cpu().half()
                                 for attention in output['attentions']]
                
                # Replace NaN values with 0 for numerical stability
                # NaN can occur in attention weights under certain conditions
                attention_map = [torch.nan_to_num(
                    attention, nan=0.0) for attention in attention_map]
                
                attention_maps.append(attention_map)

        # ---------------------------------------------------------------------
        # STEP 5: Decode generated tokens to text
        # ---------------------------------------------------------------------
        # Decode each token individually (useful for analysis)
        output_tokens = [self.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        
        # Decode all tokens together (the actual response)
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs
