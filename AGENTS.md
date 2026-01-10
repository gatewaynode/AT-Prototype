# Repository Guidelines

## Project Structure & Module Organization
- `detector/`: Core detection logic and utilities (`attn.py`, `utils.py`)
- `models/`: Model implementations and base classes (`attn_model.py`, `attn_model_nsys.py`, `model.py`, `utils.py`)
- `configs/model_configs/`: JSON configuration files for different LLM models
- `scripts/`: Shell scripts for batch processing and analysis
- `result/`: Output directory for detection results (created at runtime)
- Root level: Main entry points (`run.py`, `run_dataset.py`, `select_head.py`, `utils.py`)

## Build, Test, and Development Commands
- Run single query test: `python run.py --model_name qwen2-attn`
- Run dataset evaluation: `python run_dataset.py --model_name qwen2-attn --dataset_name deepset/prompt-injections`
- Find important attention heads: `python select_head.py --model_name qwen2-attn --num_data 30 --dataset llm`
- Batch head selection: `bash scripts/find_heads.sh`
- Batch dataset evaluation: `bash scripts/run_dataset.sh`

## Coding Style & Naming Conventions
- Python: PEP 8 compliant; use snake_case for functions/variables, PascalCase for classes
- Imports: Group imports (standard library, third-party, local) with blank lines between groups
- Type hints: Use type hints where appropriate for function signatures
- Docstrings: Include docstrings for all public functions and classes

## Logging & Output
- Results are saved to `./result/{dataset_name}/` directory
- Logs include JSON-formatted detection results and summary metrics (AUC, AUPRC, FNR, FPR)
- Use tqdm progress bars for long-running operations
- Print model information before inference starts

## Testing Guidelines
- No formal unit tests currently implemented
- Manual testing with known prompt injection examples
- Validate detection thresholds using positive/negative example sets
- Test across multiple model architectures (Qwen2, Llama3, Mistral, Phi3, Gemma2, Granite3)

## Security & Configuration Tips
- **Critical**: `trust_remote_code=True` is used in [`models/attn_model.py:20`](models/attn_model.py:20) - only use with trusted model sources
- Validate user inputs before file path construction to prevent directory traversal
- Model configurations are loaded from JSON files in `configs/model_configs/`
- Important attention heads are pre-configured per model based on analysis results

## Documentation
- Update README.md when adding new models or changing detection logic
- Document any changes to attention head selection methodology
- Keep configuration files synchronized with supported models

## Model Configuration Guidelines

### Adding a New Model
1. Create JSON config in `configs/model_configs/{model_name}_config.json`
2. Specify provider: `"attn-hf"` (with system prompt) or `"attn-hf-no-sys"` (without)
3. Set model_id to Hugging Face repository path
4. Configure temperature and max_output_tokens parameters
5. Determine important_heads by running `select_head.py` with sample data

### Important Heads Selection
- Run head selection script: `python select_head.py --model_name {name} --num_data 30 --dataset llm`
- Analyze output to identify heads that show significant attention pattern differences
- Update config file with selected [layer, head] pairs
- Validate detection performance on test dataset

## Detection System Architecture

### AttentionDetector Class
- Located in [`detector/attn.py`](detector/attn.py:1)
- Uses attention pattern analysis to detect prompt injection attempts
- Supports dynamic threshold calculation from positive/negative examples
- Returns boolean detection result and focus_score metric

### Model Classes
- **AttentionModel**: Standard model with system message support ([`models/attn_model.py`](models/attn_model.py:1))
- **AttentionModelNoSys**: Model without system message formatting ([`models/attn_model_nsys.py`](models/attn_model_nsys.py:1))
- Both inherit from base [`Model`](models/model.py:1) class

### Attention Processing
- Located in [`detector/utils.py`](detector/utils.py:1)
- `process_attn()`: Normalizes and processes attention maps
- `calc_attn_score()`: Computes focus score from important heads
- Supports multiple normalization strategies (sum, max)

## Performance Optimization Tips
- Use `torch.no_grad()` context during inference to reduce memory usage
- Detach and move attention maps to CPU early ([`models/attn_model.py:112`](models/attn_model.py:112))
- Set appropriate `max_output_tokens` to limit generation length
- Use half precision (bfloat16) for model weights

## Common Issues & Troubleshooting

### Model Loading Errors
- Ensure Hugging Face model ID is correct and accessible
- Check CUDA availability if using GPU acceleration
- Verify sufficient memory for model loading

### Attention Map Extraction
- Some models require specific token position calculations
- Update data_range logic in [`models/attn_model.py:62-73`](models/attn_model.py:62) for new architectures
- Use `get_map_dim()` to verify attention map dimensions

### Detection Threshold Tuning
- Default threshold may not work for all use cases
- Provide pos_examples and neg_examples to calculate optimal threshold
- Adjust manually based on false positive/negative rates

## Dataset Integration
- Uses Hugging Face datasets library ([`run_dataset.py:34`](run_dataset.py:34))
- Expects dataset with 'text' and 'label' fields
- Label 0 = normal text, Label 1 = prompt injection
- Supports custom datasets via --dataset_name argument

## Shell Script Guidelines
- Scripts in `scripts/` directory automate batch processing
- Use bash arrays for model lists
- Append output to analysis files with `>>`
- Include error handling and logging where appropriate
