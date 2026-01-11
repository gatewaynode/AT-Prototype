"""
Dataset Evaluation Script for Prompt Injection Detection.

This script evaluates the attention-based prompt injection detector on a
standard dataset and computes comprehensive performance metrics. Unlike run.py
which tests single queries, this script processes entire datasets and generates
statistical evaluation results.

Usage Examples:
--------------
# Evaluate on deepset/prompt-injections dataset (default):
python run_dataset.py --model_name qwen2-attn

# Evaluate with a specific seed:
python run_dataset.py --model_name qwen2-attn --seed 42

# Use a different Hugging Face dataset:
python run_dataset.py --model_name llama3_8b-attn --dataset_name your-org/your-dataset

# Batch evaluation (see scripts/run_dataset.sh):
bash scripts/run_dataset.sh

Output Files:
------------
Results are saved to ./result/{dataset_name}/:

1. {model_name}-{seed}.json
   - Detailed per-sample results
   - Contains text, label, detection result, and focus score for each sample

2. result.jsonl  
   - Summary metrics appended as JSON lines
   - Format: {"model": "...", "seed": 0, "auc": 0.95, "auprc": 0.92, "fnr": 0.03, "fpr": 0.05}

Evaluation Metrics:
------------------
- AUC (Area Under ROC Curve): Overall ranking quality (0.0 to 1.0, higher is better)
- AUPRC (Area Under Precision-Recall Curve): Performance on positive class
- FNR (False Negative Rate): Missed injections (lower is better for security)
- FPR (False Positive Rate): False alarms (lower is better for usability)

Dataset Format Expected:
-----------------------
The dataset should have:
- 'test' split
- 'text' field: The input text to classify
- 'label' field: 0 = normal, 1 = prompt injection

Default dataset: deepset/prompt-injections
  - ~650 test samples
  - Balanced between normal text and injection attempts
  - Source: https://huggingface.co/datasets/deepset/prompt-injections

Architecture:
------------
run_dataset.py
  └── datasets.load_dataset()  → Loads HuggingFace dataset
  └── AttentionDetector.detect() → Classifies each sample
  └── sklearn.metrics           → Computes AUC, AUPRC, confusion matrix
"""

import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import open_config, create_model
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Ensures identical results when re-running with the same seed, which is
    important for:
    - Reproducible benchmark comparisons
    - Debugging detection issues
    - Scientific validity of experiments
    
    Args:
        seed (int): Random seed value.
    
    Note:
        Identical to set_seed() in run.py. Both scripts need deterministic
        behavior for fair comparisons.
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy random state
    np.random.seed(seed)
    
    # PyTorch random states (CPU + GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Force deterministic CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main execution function for dataset-wide prompt injection evaluation.
    
    Workflow:
    ---------
    1. Set random seed for reproducibility
    2. Configure output paths for logs and results
    3. Load model and initialize detector
    4. Load and iterate through test dataset
    5. Collect predictions and calculate metrics
    6. Save detailed logs and summary results
    
    Args:
        args: Parsed command-line arguments containing:
            - model_name (str): Name of model config
            - dataset_name (str): Hugging Face dataset identifier
            - seed (int): Random seed
    
    Metrics Explained:
    -----------------
    - **AUC (Area Under ROC Curve)**:
      Measures the model's ability to rank positive samples higher than
      negative samples. AUC = 1.0 means perfect ranking, 0.5 means random.
      
    - **AUPRC (Area Under Precision-Recall Curve)**:
      Like AUC but more relevant for imbalanced datasets. Higher values
      indicate better performance on the positive (injection) class.
      
    - **FNR (False Negative Rate)** = FN / (FN + TP):
      Proportion of actual injections that were missed. Critical for security -
      we want this as low as possible to catch attacks.
      
    - **FPR (False Positive Rate)** = FP / (FP + TN):
      Proportion of normal inputs incorrectly flagged as injections.
      Affects usability - too many false alarms annoy users.
    
    Important Note on Score Transformation:
        The detector returns focus_score where:
        - High score = safe (model focuses on instruction)
        - Low score = unsafe (model focuses on injection)
        
        But sklearn metrics expect:
        - High score = positive class (injection)
        - Low score = negative class (normal)
        
        So we use (1 - focus_score) for metric calculations.
    """
    # =========================================================================
    # STEP 1: Set Random Seed
    # =========================================================================
    set_seed(args.seed)

    # =========================================================================
    # STEP 2: Configure Output Paths
    # =========================================================================
    # Detailed logs: individual results for each sample
    # Format: {"result": [{"text": "...", "label": 0, "result": [...]}, ...]}
    output_logs = f"./result/{args.dataset_name}/{args.model_name}-{args.seed}.json"
    
    # Summary results: one JSON line per evaluation run (appended)
    # Format: {"model": "...", "seed": 0, "auc": 0.95, ...}
    output_result = f"./result/{args.dataset_name}/result.jsonl"
    
    # =========================================================================
    # STEP 3: Load Model and Initialize Detector
    # =========================================================================
    # Load configuration from JSON file
    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)

    # Create model instance (AttentionModel or AttentionModelNoSys based on provider)
    model = create_model(config=model_config)
    model.print_model_info()

    # =========================================================================
    # STEP 4: Load Dataset
    # =========================================================================
    # Load from Hugging Face Hub (or local cache if previously downloaded)
    # Expected format:
    #   - 'test' split for evaluation
    #   - 'text' field: input text to classify
    #   - 'label' field: 0 = normal, 1 = injection
    dataset = load_dataset(args.dataset_name)
    test_data = dataset['test']
    
    # Initialize the attention-based detector
    # Uses important_heads from model config to calculate focus scores
    detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")

    # =========================================================================
    # STEP 5: Run Detection on All Samples
    # =========================================================================
    # Tracking arrays for metrics calculation
    labels = []       # Ground truth labels (0 = normal, 1 = injection)
    predictions = []  # Binary detection decisions (True/False -> 1/0)
    scores = []       # Continuous scores for AUC/AUPRC (1 - focus_score)
    logs = []         # Detailed results for each sample

    # Process each sample in the test set with progress bar
    for data in tqdm(test_data, desc="Evaluating samples"):
        # Run detection on this sample
        # result[0]: bool - whether injection was detected
        # result[1]: dict - contains 'focus_score' and other metadata
        result = detector.detect(data['text'])
        detect = result[0]
        score = result[1]['focus_score']

        # Collect ground truth and predictions
        labels.append(data['label'])
        predictions.append(detect)
        
        # IMPORTANT: Invert score for sklearn metrics
        # focus_score: high = safe, low = unsafe
        # sklearn expects: high = positive class (injection)
        # So we use (1 - focus_score) to align with label=1 meaning injection
        scores.append(1 - score)

        # Store detailed result for this sample
        result_data = {
            "text": data['text'],
            "label": data['label'],
            "result": result
        }
        logs.append(result_data)

    # =========================================================================
    # STEP 6: Calculate Performance Metrics
    # =========================================================================
    # AUC: How well does the continuous score rank injections above normal text?
    # Perfect AUC = 1.0 means all injections have higher scores than all normal text
    auc_score = roc_auc_score(labels, scores)
    
    # AUPRC: Area under Precision-Recall curve
    # Better metric when classes are imbalanced (more normal than injection samples)
    auprc_score = average_precision_score(labels, scores)

    # Confusion Matrix breakdown:
    # TN (True Negative): Correctly identified as normal
    # FP (False Positive): Normal text incorrectly flagged as injection
    # FN (False Negative): Injection missed (classified as normal) - DANGEROUS
    # TP (True Positive): Correctly detected injection
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # FNR (False Negative Rate): What fraction of injections did we miss?
    # fnr = fn / (fn + tp) = fn / total_actual_positives
    # Lower is better - we want to catch all injections
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # FPR (False Positive Rate): What fraction of normal inputs were falsely flagged?
    # fpr = fp / (fp + tn) = fp / total_actual_negatives  
    # Lower is better - we don't want to annoy users with false alarms
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Round to 3 decimal places for readability
    auc_score = round(auc_score, 3)
    auprc_score = round(auprc_score, 3)
    fnr = round(fnr, 3)
    fpr = round(fpr, 3)

    # Print summary to console
    print(f"AUC Score: {auc_score}; AUPRC Score: {auprc_score}; FNR: {fnr}; FPR: {fpr}")
    
    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    # Save detailed per-sample logs
    # Creates directory structure if it doesn't exist
    os.makedirs(os.path.dirname(output_logs), exist_ok=True)
    with open(output_logs, "w") as f_out:
        f_out.write(json.dumps({"result": logs}, indent=4))

    # Append summary metrics to results file (JSON Lines format)
    # Each line is a complete JSON object for easy parsing
    os.makedirs(os.path.dirname(output_result), exist_ok=True)
    with open(output_result, "a") as f_out:
        f_out.write(json.dumps({
            "model": args.model_name,
            "seed": args.seed,
            "auc": auc_score,
            "auprc": auprc_score,
            "fnr": fnr,
            "fpr": fpr
        }) + "\n")


if __name__ == "__main__":
    # =========================================================================
    # COMMAND-LINE ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Prompt Injection Detection - Dataset Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dataset.py
  python run_dataset.py --model_name llama3_8b-attn
  python run_dataset.py --model_name qwen2-attn --dataset_name deepset/prompt-injections --seed 42

Evaluation Metrics:
  AUC    : Area Under ROC Curve (ranking quality, higher is better)
  AUPRC  : Area Under Precision-Recall Curve (positive class performance)
  FNR    : False Negative Rate (missed injections, lower is better)
  FPR    : False Positive Rate (false alarms, lower is better)

Output Files (in ./result/{dataset_name}/):
  {model_name}-{seed}.json  : Detailed per-sample results
  result.jsonl              : Summary metrics (appended per run)

Available Models (in configs/model_configs/):
  - qwen2-attn        : Qwen2-1.5B-Instruct
  - llama3_8b-attn    : Meta-Llama-3-8B-Instruct
  - mistral_7b-attn   : Mistral-7B-Instruct
  - phi3-attn         : Phi-3-mini-128k-instruct
  - gemma2_9b-attn    : Gemma-2-9B-it
  - granite3_8b-attn  : Granite-3.0-8B-Instruct
        """
    )
    
    # --model_name: Which model configuration to use
    # Maps to: configs/model_configs/{model_name}_config.json
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="qwen2-attn",
        help="Name of the model configuration (without _config.json suffix). "
             "See configs/model_configs/ for available options."
    )
    
    # --dataset_name: Hugging Face dataset to evaluate on
    # Must have 'test' split with 'text' and 'label' fields
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="deepset/prompt-injections", 
        help="Hugging Face dataset identifier. Must have 'test' split with "
             "'text' (input) and 'label' (0=normal, 1=injection) fields."
    )
    
    # --seed: Random seed for reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    
    # Parse arguments and run main function
    args = parser.parse_args()

    main(args)
