# GLM-4-7B Model Integration Tasks

## Task 1: Create Initial GLM-4-7B Configuration File

**Title**: Create initial GLM-4-7B model configuration

**Description**:
Create a new configuration file `configs/model_configs/glm4-7b-attn_config.json` based on the local model in `./GLM-4.7/`. The configuration should use the `attn-hf` provider (supports system messages) and initially set `important_heads` to `"all"` for testing purposes.

**Acceptance Criteria**:
- [ ] File `configs/model_configs/glm4-7b-attn_config.json` exists
- [ ] Configuration includes:
  - `provider`: `"attn-hf"`
  - `name`: `"glm4-7b-attn"`
  - `model_id`: `"./GLM-4.7"`
  - `temperature`: `0.1`
  - `max_output_tokens`: `32`
  - `important_heads`: `"all"` (temporary, will be updated after head selection)
- [ ] Configuration follows same JSON structure as existing model configs

**Reference Files**:
- Example config: `configs/model_configs/qwen2-attn_config.json`
- Model metadata: `GLM-4.7/config.json`, `GLM-4.7/chat_template.jinja`

---

## Task 2: Add GLM-4 Token Position Logic to AttentionModel

**Title**: Implement GLM-4 data_range token position calculation

**Description**:
Add a conditional branch in `models/attn_model.py` to calculate the token position offsets (`data_range`) for GLM-4's chat template format. GLM-4 uses the format: `[gMASK]<sop><|system|>{instruction}<|user|>Data: {data}<|assistant|><think>`. The offsets need to be determined by examining tokenized output.

**Acceptance Criteria**:
- [ ] New conditional branch added in `models/attn_model.py` around line 72
- [ ] Branch checks for `"glm4" in self.name` or similar identifier
- [ ] `data_range` tuple calculated as `((start_offset, start_offset+instruction_len), (end_offset-data_len, end_offset))`
- [ ] Offsets determined by manually testing tokenization with a sample input
- [ ] Code follows existing pattern (similar to qwen, phi3, mistral branches)

**Testing Steps**:
1. Create a test script to tokenize a sample message and identify positions:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./GLM-4.7")
messages = [
    {"role": "system", "content": "Say hello"},
    {"role": "user", "content": "Data: test"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
```
2. Identify where instruction and data tokens appear in the sequence
3. Calculate appropriate offsets based on token positions

**Reference Files**:
- Implementation file: `models/attn_model.py` lines 62-73
- Chat template: `GLM-4.7/chat_template.jinja`

---

## Task 3: Test Basic GLM-4 Model Loading and Inference

**Title**: Verify GLM-4 model loads and generates attention maps

**Description**:
Test that the GLM-4 model loads correctly and can perform basic inference with attention map extraction. This verifies the configuration and code changes work before attempting head selection.

**Acceptance Criteria**:
- [ ] Model loads without errors when running: `python run.py --model_name glm4-7b-attn --test_query "Hello"`
- [ ] No exceptions related to:
  - Model architecture compatibility
  - Token position calculations
  - Attention map extraction
- [ ] Output shows:
  - Model information printed
  - Detector initialized
  - Focus score calculated (any value)
  - No Python errors or warnings

**Troubleshooting**:
- If model fails to load: Check if `trust_remote_code=True` is set in `models/attn_model.py:20`
- If token position errors: Review data_range calculations from Task 2
- If attention errors: GLM-4's MoE architecture may need attention output format adjustments

**Reference Files**:
- Test script: `run.py`
- Model loader: `models/attn_model.py`

---

## Task 4: Run Attention Head Selection for GLM-4

**Title**: Identify important attention heads for GLM-4 prompt injection detection

**Description**:
Use the `select_head.py` script to analyze GLM-4's attention patterns and identify which heads are most relevant for detecting prompt injections. This requires 30 sample prompts (mix of normal and injection attempts) to calculate head importance scores.

**Acceptance Criteria**:
- [ ] Head selection script runs successfully: `python select_head.py --model_name glm4-7b-attn --num_data 30 --dataset llm`
- [ ] Script completes without errors
- [ ] Output provides list of important head pairs `[layer, head]`
- [ ] Important heads list has 10-40 entries (based on other model configs)
- [ ] Results are interpretable and show clear performance differences between heads

**Expected Output Format**:
```
Important heads: [[layer1, head1], [layer2, head2], ...]
```

**Reference Files**:
- Selection script: `select_head.py`
- Example outputs: See other model configs' `important_heads` parameter

---

## Task 5: Update GLM-4 Config with Selected Important Heads

**Title**: Replace "all" heads with empirically selected important heads

**Description**:
Update the GLM-4 configuration file to replace `"important_heads": "all"` with the specific head list identified in Task 4. Using specific heads improves detection accuracy and performance.

**Acceptance Criteria**:
- [ ] `configs/model_configs/glm4-7b-attn_config.json` updated
- [ ] `important_heads` parameter contains array of `[layer, head]` pairs
- [ ] Array has 10-40 entries based on selection results
- [ ] Configuration file is valid JSON
- [ ] Basic test still runs: `python run.py --model_name glm4-7b-attn --test_query "test"`

**Reference Files**:
- Config file: `configs/model_configs/glm4-7b-attn_config.json`
- Selection output: From Task 4

---

## Task 6: Validate GLM-4 Detection on Test Dataset

**Title**: Evaluate GLM-4 prompt injection detection performance

**Description**:
Run the full dataset evaluation to measure GLM-4's prompt injection detection accuracy. This validates that the complete integration works correctly and provides performance metrics.

**Acceptance Criteria**:
- [ ] Dataset evaluation runs successfully: `python run_dataset.py --model_name glm4-7b-attn --dataset_name deepset/prompt-injections`
- [ ] Script completes without errors
- [ ] Results saved to `./result/deepset/prompt-injections/glm4-7b-attn-0.json`
- [ ] Summary metrics calculated: AUC, AUPRC, FNR, FPR
- [ ] Performance metrics are reasonable (compare to other models in project)

**Expected Output**:
```
AUC Score: [value]; AUPRC Score: [value]; FNR: [value]; FPR: [value]
```

**Reference Files**:
- Evaluation script: `run_dataset.py`
- Results directory: `./result/`
- Comparison: Results from other models (qwen2, llama3, etc.)

---

## Notes

- GLM-4-7B is a Mixture of Experts (MoE) model with 92 layers and 96 attention heads per layer
- Local model files are in `./GLM-4.7/` directory
- Model uses custom `Glm4MoeForCausalLM` architecture requiring `trust_remote_code=True`
- Chat template is defined in `GLM-4.7/chat_template.jinja`

## Dependencies

Tasks must be completed in order (1 → 2 → 3 → 4 → 5 → 6) as each depends on the previous task's completion.
