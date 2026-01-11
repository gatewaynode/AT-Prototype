"""
Base Model Class for Attention-Based Detection.

This module provides the abstract base class that all model implementations
must inherit from. It defines the common interface and shared functionality
for language models used in prompt injection detection.

Design Pattern:
---------------
This follows the Template Method pattern where:
- The base Model class defines the interface and common attributes
- Concrete implementations (AttentionModel, AttentionModelNoSys) provide
  specific behavior for different model architectures

Inheritance Hierarchy:
----------------------
    Model (base class)
    ├── AttentionModel       # Models with system message support
    └── AttentionModelNoSys  # Models without system message support

Provider Types:
---------------
The "provider" config field determines which concrete class to instantiate:
- "attn-hf": Uses AttentionModel (supports system messages)
- "attn-hf-no-sys": Uses AttentionModelNoSys (no system message support)

See utils.py create_model() for the factory function that creates the
appropriate model instance based on provider type.

Usage:
------
This class should not be instantiated directly. Use create_model() instead:

    from utils import open_config, create_model
    
    config = open_config("configs/model_configs/qwen2-attn_config.json")
    model = create_model(config)  # Returns AttentionModel or AttentionModelNoSys
    model.print_model_info()
"""


class Model:
    """
    Abstract base class for language models in the attention detection system.
    
    This class defines the common interface that all model implementations must
    follow. It stores shared configuration values and provides utility methods
    that work across all model types.
    
    Subclasses must implement:
        - inference(): Token generation with attention map extraction
        - get_map_dim(): Discover attention map dimensions
    
    Subclasses may optionally implement:
        - set_API_key(): For API-based models (not currently used)
        - query(): Generic query interface (not currently used)
    
    Attributes:
        provider (str): Model provider identifier from config:
                       - "attn-hf": HuggingFace model with system message
                       - "attn-hf-no-sys": HuggingFace model without system message
        name (str): Model name identifier used for conditional logic
                   (e.g., "qwen-attn", "llama3-8b-attn", "gemma2_9b-attn")
        temperature (float): Sampling temperature for generation
                            Lower = more deterministic, Higher = more creative
    
    Configuration File Structure:
        {
            "model_info": {
                "provider": "attn-hf",
                "name": "model-name",
                "model_id": "huggingface/model-path"
            },
            "params": {
                "temperature": 0.1,
                "max_output_tokens": 32,
                "important_heads": [[layer, head], ...]
            }
        }
    
    Example:
        >>> # Typically created via factory function, not directly
        >>> from utils import create_model, open_config
        >>> config = open_config("configs/model_configs/qwen2-attn_config.json")
        >>> model = create_model(config)
        >>> model.print_model_info()
        -------------------
        | Provider: attn-hf
        | Model name: qwen-attn
        -------------------
    """
    
    def __init__(self, config):
        """
        Initialize base model attributes from configuration.
        
        This constructor extracts common configuration values that apply
        to all model implementations. Subclasses should call super().__init__(config)
        in their constructors before setting up model-specific attributes.
        
        Args:
            config (dict): Configuration dictionary loaded from JSON file.
                Must contain:
                    - config["model_info"]["provider"]: Provider identifier string
                    - config["model_info"]["name"]: Model name string
                    - config["params"]["temperature"]: Temperature float (0.0 to 2.0)
        
        Example config:
            {
                "model_info": {
                    "provider": "attn-hf",
                    "name": "qwen-attn",
                    "model_id": "Qwen/Qwen2-1.5B-Instruct"
                },
                "params": {
                    "temperature": 0.1,
                    "max_output_tokens": 32,
                    "important_heads": [[10, 6], [11, 0], ...]
                }
            }
        """
        # Provider determines which subclass implementation to use
        # This is used by the factory function in utils.py
        self.provider = config["model_info"]["provider"]
        
        # Model name is used for conditional logic in subclasses
        # (e.g., determining token position offsets in data_range)
        self.name = config["model_info"]["name"]
        
        # Temperature controls randomness in token sampling
        # Lower values (0.1) = more deterministic/focused
        # Higher values (1.0+) = more random/creative
        self.temperature = float(config["params"]["temperature"])

    def print_model_info(self):
        """
        Print formatted model information to console.
        
        Displays the provider and model name in a bordered box format.
        Useful for logging at startup to confirm which model is loaded.
        
        Output format:
            -------------------
            | Provider: attn-hf
            | Model name: qwen-attn
            -------------------
        
        Note:
            The box width automatically adjusts to fit the model name.
        """
        # Calculate border width based on the longest line (model name line)
        border = '-' * len(f'| Model name: {self.name}')
        print(f"{border}\n| Provider: {self.provider}\n| Model name: {self.name}\n{border}")

    def set_API_key(self):
        """
        Set the API key for API-based model providers.
        
        This method is a placeholder for potential future API-based models
        (e.g., OpenAI, Anthropic). Currently not used since all models are
        local HuggingFace models.
        
        Raises:
            NotImplementedError: Always, unless overridden by subclass.
        
        Note:
            If implementing an API-based model, override this method to
            configure the API client with authentication credentials.
        """
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        """
        Generic query method for simple text generation.
        
        This method is a placeholder for a simplified query interface that
        doesn't require attention map extraction. Currently not used as all
        queries go through the inference() method for attention analysis.
        
        Raises:
            NotImplementedError: Always, unless overridden by subclass.
        
        Note:
            The inference() method in subclasses provides the full
            functionality needed for prompt injection detection.
        """
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
