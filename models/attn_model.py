"""
Attention Model With System Message Support.

This module provides the AttentionModel class for language models that support
system messages in their chat templates. This is the primary model class used
for most LLMs including Qwen, Llama3, Mistral, Phi3, and Granite.

Key Features:
-------------
1. **System Message Support**: Uses separate system and user message roles
   for clear instruction/data separation.
   
2. **Multiple Model Architectures**: Pre-configured token position mappings
   for popular model families.
   
3. **Attention Optimization**: Uses get_last_attn() to extract only the
   last token's attention, reducing memory usage.

Message Format:
---------------
[
    {"role": "system", "content": instruction},
    {"role": "user", "content": "Data: " + data}
]

Supported Models:
-----------------
- Qwen/Qwen2 (provider: "attn-hf")
- Llama3-8B (provider: "attn-hf")  
- Mistral-7B (provider: "attn-hf")
- Phi3 (provider: "attn-hf")
- Granite3-8B (provider: "attn-hf")

For models WITHOUT system message support (e.g., Gemma2), use
AttentionModelNoSys instead.

Security Note:
--------------
This class uses trust_remote_code=True which executes code from the model
repository. Only use with models from trusted sources.

Usage:
------
    from models.attn_model import AttentionModel
    
    config = {
        "model_info": {"name": "qwen-attn", "model_id": "Qwen/Qwen2-1.5B-Instruct"},
        "params": {"max_output_tokens": 32, "important_heads": [[10, 6], ...]}
    }
    model = AttentionModel(config)
    text, tokens, attns, input_toks, ranges, probs = model.inference(
        "Summarize this:", "Some text to summarize"
    )
"""

import torch
from .model import Model
from .utils import sample_token, get_last_attn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Automatically select GPU if available, otherwise fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionModel(Model):
    """
    Language model wrapper for attention-based prompt injection detection.
    
    This class wraps HuggingFace models to enable token-by-token generation
    with attention map extraction. The attention maps are essential for
    detecting prompt injection attempts by analyzing where the model focuses.
    
    This variant supports models with system message capability, allowing
    clean separation between the instruction (system) and user data.
    
    Attributes:
        name (str): Model identifier from config, used for conditional logic
        max_output_tokens (int): Maximum tokens to generate per inference
        tokenizer: HuggingFace tokenizer for the model
        model: HuggingFace CausalLM model with attention output capability
        important_heads (list): List of [layer, head] pairs for detection
        top_k (int): Top-k sampling parameter (default 50)
        top_p (float): Nucleus sampling parameter (default None = disabled)
    
    Architecture Support:
        - Qwen2: Tested with Qwen/Qwen2-1.5B-Instruct
        - Llama3: Tested with meta-llama/Meta-Llama-3-8B-Instruct
        - Mistral: Tested with mistralai/Mistral-7B-Instruct-v0.3
        - Phi3: Tested with microsoft/Phi-3-mini-128k-instruct
        - Granite3: Tested with ibm-granite/granite-3.1-8b-instruct
    
    Memory Optimization:
        - Uses bfloat16 precision to reduce model memory footprint
        - Attention maps are immediately detached and moved to CPU
        - get_last_attn() extracts only the last token's attention row
        - Converted to half precision (fp16) for storage efficiency
        - torch.no_grad() prevents gradient accumulation
    
    Example:
        >>> model = AttentionModel(config)
        >>> generated_text, output_tokens, attention_maps, input_tokens, data_range, probs = \\
        ...     model.inference("Say hello", "test input", max_output_tokens=5)
        >>> print(f"Generated: {generated_text}")
        >>> print(f"Attention maps: {len(attention_maps)} tokens")
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
                        "provider": "attn-hf",
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
            - trust_remote_code=True: Required for some models (e.g., Qwen)
              SECURITY WARNING: Only use with trusted model sources
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
        # CRITICAL: attn_implementation="eager" is REQUIRED - Flash Attention cannot output attention maps
        # SECURITY: trust_remote_code=True allows execution of model-provided code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,   # Half precision for memory efficiency
            device_map=device,             # Auto-place on best available device
            trust_remote_code=True,        # Required for some models (security risk!)
            attn_implementation="eager",   # Required for attention map output
        ).eval()  # Set to evaluation mode (disables dropout, etc.)

        # Sampling parameters for token generation
        self.top_k = 50    # Consider only top 50 tokens when sampling
        self.top_p = None  # Nucleus sampling disabled by default

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
        
        Message Format (With System Message):
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Data: " + data}
            ]
            
            The instruction goes in the system message, and user data goes
            in the user message. This separation is important for the
            attention analysis.
        
        Args:
            instruction (str): The task instruction placed in the system message.
                      This is what the model should follow.
            data (str): User input data placed in the user message. This is
                      where potential prompt injections would be located.
            max_output_tokens (int, optional): Override the default token limit.
                      For injection detection, we typically only need 1 token
                      since the first token's attention is most indicative.
        
        Returns:
            tuple: Six-element tuple containing:
                - generated_text (str): The complete generated response
                - output_tokens (list[str]): Individual decoded tokens
                - attention_maps (list): Attention maps for each generated token.
                      Structure: [token_idx][layer][batch, heads, 1, seq]
                      Note: Only last row of attention matrix is kept (via get_last_attn)
                - input_tokens (list[str]): Tokenized input for debugging
                - data_range (tuple): Token position ranges for instruction and data:
                      ((inst_start, inst_end), (data_start, data_end))
                - generated_probs (list[float]): Probability of each generated token
        
        Token Position Calculation (data_range):
            Different models have different chat template formats with varying
            numbers of special tokens. The data_range specifies where instruction
            and data tokens are located in the sequence.
            
            Model-specific offsets:
            
            - qwen: ((3, 3+inst_len), (-5-data_len, -5))
              Format: <|im_start|>system\\n{inst}<|im_end|>\\n<|im_start|>user\\n{data}<|im_end|>\\n<|im_start|>assistant\\n
              
            - phi3: ((1, 1+inst_len), (-2-data_len, -2))
              Format: <|system|>\\n{inst}<|end|>\\n<|user|>\\n{data}<|end|>\\n<|assistant|>\\n
              
            - llama3-8b: ((5, 5+inst_len), (-5-data_len, -5))
              Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{inst}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{data}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n
              
            - mistral-7b: ((3, 3+inst_len), (-1-data_len, -1))
              Format: [INST]{inst}[/INST]\\n{data}[/INST]
              
            - granite3-8b: ((3, 3+inst_len), (-5-data_len, -5))
              Similar to Qwen format
        
        Memory Management:
            - torch.no_grad(): Prevents gradient computation/storage
            - Attention maps detached immediately after extraction
            - get_last_attn(): Keeps only the last attention row, discarding
              the full seq×seq attention matrix to save significant memory
            - Converted to CPU and half precision to free GPU memory
            - NaN values replaced with 0 for numerical stability
        
        Example:
            >>> model = AttentionModel(config)
            >>> text, tokens, attns, in_toks, ranges, probs = model.inference(
            ...     "Summarize:", "The quick brown fox", max_output_tokens=10
            ... )
            >>> print(f"Generated: {text}")
            >>> print(f"Instruction range: {ranges[0]}, Data range: {ranges[1]}")
        """
        # ---------------------------------------------------------------------
        # STEP 1: Construct messages with system and user roles
        # ---------------------------------------------------------------------
        # Separate system (instruction) and user (data) messages
        # This structure enables clean analysis of where attention flows
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Data: " + data}
        ]

        # Apply the model's chat template to format the messages
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
        # - Different special tokens (<|im_start|>, [INST], etc.)
        # - Different chat template structures
        #
        # Negative indices are relative to the end of the sequence
        
        if "qwen" in self.name:
            # Qwen chat template structure:
            # <|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\nData: {data}<|im_end|>\n<|im_start|>assistant\n
            # Special tokens: <|im_start|>, system, \n = 3 tokens before instruction
            # End tokens: <|im_end|>\n<|im_start|>assistant\n = 5 tokens after data
            data_range = ((3, 3+instruction_len), (-5-data_len, -5))
            
        elif "phi3" in self.name:
            # Phi3 chat template structure (simpler than others):
            # <|system|>\n{instruction}<|end|>\n<|user|>\nData: {data}<|end|>\n<|assistant|>\n
            # 1 token before instruction, 2 tokens after data
            data_range = ((1, 1+instruction_len), (-2-data_len, -2))
            
        elif "llama3-8b" in self.name:
            # Llama3 chat template structure:
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nData: {data}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            # 5 tokens before instruction, 5 tokens after data
            data_range = ((5, 5+instruction_len), (-5-data_len, -5))
            
        elif "mistral-7b" in self.name:
            # Mistral chat template structure:
            # [INST]{instruction}[/INST]\nData: {data}[/INST]
            # 3 tokens before instruction, 1 token after data
            data_range = ((3, 3+instruction_len), (-1-data_len, -1))
            
        elif "granite3-8b" in self.name:
            # Granite3 chat template (similar to Qwen):
            # 3 tokens before instruction, 5 tokens after data
            data_range = ((3, 3+instruction_len), (-5-data_len, -5))
            
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
                    logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0)[0]

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
                attention_map = [torch.nan_to_num(
                    attention, nan=0.0) for attention in attention_map]
                
                # MEMORY OPTIMIZATION: Keep only the last row of attention
                # The full attention matrix is [batch, heads, seq, seq]
                # We only need attention FROM the last token TO all tokens
                # get_last_attn reduces to [batch, heads, 1, seq]
                attention_map = get_last_attn(attention_map)
                
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
