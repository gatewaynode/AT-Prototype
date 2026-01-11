# Model Configuration Files

This directory contains JSON configuration files for each supported LLM model. These configurations define how the Attention-Tracker system loads and uses each model for prompt injection detection.

## Table of Contents

- [JSON Schema](#json-schema)
- [Field Reference](#field-reference)
- [Provider Types](#provider-types)
- [Creating a New Model Configuration](#creating-a-new-model-configuration)
- [Extracting Configuration from LLM Metadata](#extracting-configuration-from-llm-metadata)
- [Finding Important Heads](#finding-important-heads)
- [Complete Examples](#complete-examples)

---

## JSON Schema

```json
{
    "model_info": {
        "provider": "<string: provider type>",
        "name": "<string: display name>",
        "model_id": "<string: Hugging Face model ID or local path>"
    },
    "params": {
        "temperature": <float: sampling temperature>,
        "max_output_tokens": <int: maximum tokens to generate>,
        "important_heads": [[<layer>, <head>], ...]
    }
}
```

### JSON Schema (Formal Definition)

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["model_info", "params"],
    "properties": {
        "model_info": {
            "type": "object",
            "required": ["provider", "name", "model_id"],
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["attn-hf", "attn-hf-no-sys"],
                    "description": "Provider type determining model class and prompt formatting"
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable display name for logging"
                },
                "model_id": {
                    "type": "string",
                    "description": "Hugging Face model ID or local filesystem path"
                }
            }
        },
        "params": {
            "type": "object",
            "required": ["temperature", "max_output_tokens", "important_heads"],
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Sampling temperature (lower = more deterministic)"
                },
                "max_output_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum tokens to generate during inference"
                },
                "important_heads": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "description": "Array of [layer, head] pairs for attention analysis"
                }
            }
        }
    }
}
```

---

## Field Reference

### `model_info.provider`

Determines which model class to instantiate and how prompts are formatted.

| Value | Model Class | System Message Support | Use Case |
|-------|------------|------------------------|----------|
| `attn-hf` | `AttentionModel` | ✅ Yes | Most models (Qwen2, Llama3, Mistral, Phi3, Granite3) |
| `attn-hf-no-sys` | `AttentionModelNoSys` | ❌ No | Models without system message (Gemma2) |

### `model_info.name`

Display name used in logging and result files. Can be any descriptive string.

### `model_info.model_id`

Either a Hugging Face model repository ID or a local filesystem path:

- **Hugging Face**: `"Qwen/Qwen2-1.5B-Instruct"`, `"meta-llama/Meta-Llama-3-8B-Instruct"`
- **Local path**: `"./GLM-4.7"`, `"/path/to/model"`

### `params.temperature`

Controls sampling randomness. Recommended value: `0.1` for deterministic detection.

- `0.0`: Greedy decoding (always pick highest probability token)
- `0.1`: Slightly randomized (recommended for this project)
- `1.0`: Standard sampling
- `>1.0`: High creativity / randomness

### `params.max_output_tokens`

Maximum tokens to generate per inference. Recommended: `32` for detection tasks.

### `params.important_heads`

Array of `[layer_index, head_index]` pairs identifying attention heads with high discriminative power for detecting prompt injections.

Example: `[[10, 6], [11, 0], [11, 2]]` means:
- Layer 10, Head 6
- Layer 11, Head 0
- Layer 11, Head 2

**Important**: Layer and head indices are 0-based.

---

## Provider Types

### `attn-hf` (Standard with System Message)

For models that support system messages in their chat template:

```
<|system|>
{system_prompt}
<|user|>
Instruction: {instruction}
Data: {data}
<|assistant|>
```

**Supported models**: Qwen2, Llama3, Mistral, Phi3, Granite3

### `attn-hf-no-sys` (Without System Message)

For models that don't have a separate system message role:

```
<|user|>
{instruction}
{data}
<|assistant|>
```

**Supported models**: Gemma2

---

## Creating a New Model Configuration

### Step 1: Gather Model Information

1. **Find the Hugging Face model ID** from https://huggingface.co
2. **Download or examine the model's `config.json`** to understand its architecture
3. **Check if the model supports system messages** by examining its chat template

### Step 2: Create Initial Configuration

Create a new file: `{model_name}_config.json`

```json
{
    "model_info": {
        "provider": "attn-hf",
        "name": "your-model-name",
        "model_id": "organization/model-name"
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": []
    }
}
```

### Step 3: Find Important Heads

See [Finding Important Heads](#finding-important-heads) section below.

### Step 4: Update Model Code (if needed)

If the model has a unique architecture, you may need to update the `data_range` logic in [`models/attn_model.py`](../../models/attn_model.py) to correctly identify instruction and data token positions.

---

## Extracting Configuration from LLM Metadata

When adding support for a new model, you'll need to extract key information from the model's metadata files.

### Source Files in LLM Metadata

| File | Information Extracted |
|------|----------------------|
| `config.json` | Architecture, layer count, head count, model type |
| `tokenizer_config.json` | Chat template, special tokens |
| `chat_template.jinja` | Message formatting (if separate file) |

### Key Metadata Fields

#### From `config.json`:

```json
{
    "architectures": ["Qwen2ForCausalLM"],    // Model class name
    "num_hidden_layers": 28,                   // Total transformer layers
    "num_attention_heads": 20,                 // Heads per layer
    "num_key_value_heads": 4,                  // For GQA models
    "model_type": "qwen2"                      // Model type identifier
}
```

**How to use this information**:

- **`num_hidden_layers`**: Defines the range for layer indices in `important_heads` (0 to num_hidden_layers-1)
- **`num_attention_heads`**: Defines the range for head indices (0 to num_attention_heads-1)
- **`model_type`**: May require special handling in `data_range` calculation

#### From `tokenizer_config.json`:

```json
{
    "chat_template": "{% for message in messages %}...",
    "bos_token": "<s>",
    "eos_token": "</s>"
}
```

**How to use this information**:

- **`chat_template`**: Determines if system messages are supported
  - If template includes `system` role handling → use `attn-hf`
  - If template only handles `user`/`assistant` → use `attn-hf-no-sys`

### Step-by-Step Extraction Process

#### 1. Download Model Metadata

```bash
# Using Hugging Face CLI
huggingface-cli download organization/model-name config.json tokenizer_config.json --local-dir ./model-metadata

# Or using Python
from huggingface_hub import hf_hub_download
config = hf_hub_download("organization/model-name", "config.json")
tokenizer_config = hf_hub_download("organization/model-name", "tokenizer_config.json")
```

#### 2. Examine Architecture

```bash
# Check layer and head counts
cat config.json | jq '{layers: .num_hidden_layers, heads: .num_attention_heads, model_type: .model_type}'
```

Example output:
```json
{
  "layers": 28,
  "heads": 20,
  "model_type": "qwen2"
}
```

This tells you:
- `important_heads` layer indices: 0-27
- `important_heads` head indices: 0-19

#### 3. Check System Message Support

```bash
# Look for 'system' role in chat template
cat tokenizer_config.json | jq '.chat_template' | grep -i system
```

- If found → `"provider": "attn-hf"`
- If not found → `"provider": "attn-hf-no-sys"`

#### 4. Identify Special Token Positions

For the `data_range` calculation in `models/attn_model.py`, you need to understand how the tokenizer handles:
- System message tokens
- User/Assistant role markers
- Beginning/End of sequence tokens

Use this test script:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("organization/model-name")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Instruction: Say hello\nData: World"}
]

tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
decoded = [tokenizer.decode([t]) for t in tokens[0]]

for i, token in enumerate(decoded):
    print(f"{i:3d}: {repr(token)}")
```

This helps identify where instruction and data tokens begin/end for the `data_range` tuple.

---

## Finding Important Heads

The `important_heads` parameter identifies which attention heads are most discriminative for detecting prompt injections.

### Using the Head Selection Script

```bash
python select_head.py --model_name your-model-name --num_data 30 --dataset llm
```

**Parameters**:
- `--model_name`: Config file name (without `_config.json`)
- `--num_data`: Number of samples per class (30 recommended)
- `--dataset`: Either `llm` (synthetic) or `deepset` (real dataset)

### Understanding the Output

The script compares attention patterns between:
- **Normal data**: Regular user inputs
- **Attack data**: Inputs with appended prompt injections

It outputs heads where the difference is statistically significant:

```
======== index pos (n=2) =========
[[10, 6], [11, 0], [11, 2], [11, 8], ...]
proportion: 14 (0.025)
```

**Interpretation**:
- `n=2`: Heads where mean difference > 2× standard deviation
- Copy the inner list directly to `important_heads` in your config

### Selection Criteria

| n value | Strictness | Typical Count | Use Case |
|---------|------------|---------------|----------|
| n=1 | Loose | 50+ heads | Initial exploration |
| n=2 | Moderate | 10-30 heads | Recommended default |
| n=3 | Strict | 3-10 heads | Minimal set |
| n=4+ | Very strict | 0-5 heads | May miss injections |

**Recommendation**: Start with `n=2` results. If detection performance is poor, try `n=1` results.

---

## Complete Examples

### Example 1: Qwen2 (Standard Model)

**Model metadata** (`config.json`):
```json
{
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 28,
    "num_attention_heads": 20,
    "model_type": "qwen2"
}
```

**Resulting configuration** (`qwen2-attn_config.json`):
```json
{
    "model_info": {
        "provider": "attn-hf",
        "name": "qwen-attn",
        "model_id": "Qwen/Qwen2-1.5B-Instruct"
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": [[10, 6], [11, 0], [11, 2], [11, 8], [11, 9], [11, 11], [12, 8], [13, 10], [14, 8], [15, 7], [15, 11], [17, 0], [18, 9], [19, 7]]
    }
}
```

### Example 2: Gemma2 (No System Message)

**Model metadata** (`config.json`):
```json
{
    "architectures": ["Gemma2ForCausalLM"],
    "num_hidden_layers": 42,
    "num_attention_heads": 16,
    "model_type": "gemma2"
}
```

**Chat template** (no system role → use `attn-hf-no-sys`):
```jinja
{% for message in messages %}
{% if message['role'] == 'user' %}...{% endif %}
{% if message['role'] == 'assistant' %}...{% endif %}
{% endfor %}
```

**Resulting configuration** (`gemma2_9b-attn_config.json`):
```json
{
    "model_info": {
        "provider": "attn-hf-no-sys",
        "name": "gemma2_9b-attn",
        "model_id": "google/gemma-2-9b-it"
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": [[10, 11], [11, 6], [11, 9], ...]
    }
}
```

### Example 3: Local Model (GLM-4)

**Model metadata** (`GLM-4.7/config.json`):
```json
{
    "architectures": ["Glm4MoeForCausalLM"],
    "num_hidden_layers": 92,
    "num_attention_heads": 96,
    "model_type": "glm4_moe"
}
```

**Resulting configuration** (`glm4-attn_config.json`):
```json
{
    "model_info": {
        "provider": "attn-hf",
        "name": "glm4-moe-attn",
        "model_id": "./GLM-4.7"
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": []
    }
}
```

**Note**: For local models, use a relative or absolute filesystem path as `model_id`.

---

## Troubleshooting

### Config Not Loading

- Verify JSON syntax is valid (no trailing commas)
- Ensure file name matches pattern: `{model_name}_config.json`
- Check that `model_id` is accessible (valid HF ID or path exists)

### Important Heads Empty or Too Few

- Try running `select_head.py` with `--num_data 50` for more samples
- Use lower `n` value (n=1) to include more heads
- Verify the model is loading correctly and generating output

### Wrong Provider Type

- If model errors on system message → switch to `attn-hf-no-sys`
- If attention patterns seem wrong → check `data_range` calculation in model code

### Memory Issues

- Reduce `max_output_tokens` to 1 for head selection
- Use smaller models for initial testing
- Enable bfloat16 precision (already default)
