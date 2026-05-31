"""
Run ensemble evaluation combining multiple models (T5 + BERT) with voting strategies.
Produces comparison metrics between individual models and ensemble methods.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report

from ensemble import HateSpeechEnsemble, optimize_weights, evaluate_ensemble
from data_loader import load_vihsd, load_victsd


def detect_model_type(model_path: str):
    """Detect if model is T5 or BERT from config.json."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    architectures = config.get("architectures", [])
    for arch in architectures:
        if "T5" in arch:
            return "t5", config
        if "ForSequenceClassification" in arch or "ForTokenClassification" in arch:
            return "bert", config

    # Fallback: check model_type field
    model_type = config.get("model_type", "")
    if "t5" in model_type.lower():
        return "t5", config
    return "bert", config


def detect_num_labels(config: dict) -> int:
    """Get num_labels from model config."""
    return config.get("num_labels", 2)


def get_default_models():
    """Return the curated 3-model default preset."""
    return [
        "models/vit5_finetune_balanced",
        "models/vit5_focal_loss_exp",
        "models/visobert_labeling",
    ]


def get_all_models():
    """Get all fine-tuned models (exclude pretrain-only)."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_dirs = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if "pretrain" in entry:
            continue
        config_path = os.path.join(full_path, "config.json")
        if os.path.exists(config_path):
            model_dirs.append(os.path.join("models", entry))
    return model_dirs


def parse_args(args=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run ensemble evaluation combining T5 + BERT models"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Custom model paths (default: curated 3-model preset)"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Use all fine-tuned models in models/ directory"
    )
    parser.add_argument(
        "--task", choices=["vihsd", "victsd"], default="vihsd",
        help="Task to evaluate (default: vihsd)"
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip weight optimization (use equal weights)"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Manual weights for models (must match number of models)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--output", type=str, default="results/ensemble_results.csv",
        help="Output CSV path (default: results/ensemble_results.csv)"
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Local CSV data file (columns: text_col, label_col) as fallback when HF dataset is unavailable"
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    # Determine model list
    if args.models:
        model_paths = args.models
    elif args.all_models:
        model_paths = get_all_models()
    else:
        model_paths = get_default_models()

    # Device detection (MPS-aware)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Models: {len(model_paths)}")

    # [1/6] Load models
    print(f"\n[1/6] Loading {len(model_paths)} models...")
    ensemble = HateSpeechEnsemble(device=device)

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"  WARNING: {model_path} not found, skipping")
            continue

        model_type, config = detect_model_type(model_path)
        name = os.path.basename(model_path)

        if model_type == "t5":
            ensemble.add_t5_model(name, model_path)
            print(f"  + {name} (T5)")
        else:
            num_labels = detect_num_labels(config)
            # For BERT models on ViHSD: remap 2-class (NONE/HATE) to 3-class (CLEAN/OFFENSIVE/HATE)
            label_remap = None
            if args.task == "vihsd" and num_labels == 2:
                label_remap = {0: 0, 1: 2}  # NONE→CLEAN, HATE→HATE
            ensemble.add_bert_model(name, model_path, num_labels, label_remap=label_remap)
            remap_str = f" [remap: {label_remap}]" if label_remap else ""
            print(f"  + {name} (BERT, {num_labels} labels){remap_str}")

    if not ensemble.models:
        print("ERROR: No models loaded. Check model paths.")
        sys.exit(1)

    # [2/6] Load test data
    print(f"\n[2/6] Loading test data...")
    if args.task == "vihsd":
        label_map = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
        num_classes = 3
        task_prefix = "hate-speech-detection"
        text_col = "free_text"
        label_col = "label_id"
    else:
        label_map = {"NONE": 0, "TOXIC": 1}
        num_classes = 2
        task_prefix = "toxic-speech-detection"
        text_col = "Comment"
        label_col = "Toxicity"

    if args.data_file:
        # Load from local CSV file
        df = pd.read_csv(args.data_file)
        # Auto-detect columns
        if text_col not in df.columns:
            text_col = df.columns[0]
        if label_col not in df.columns:
            label_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
        # Split into val/test (80/20)
        from sklearn.model_selection import train_test_split
        val_df, test_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df[label_col])
        print(f"  Loaded local data: {len(df)} samples")
    else:
        try:
            if args.task == "vihsd":
                _, val_df, test_df, metadata = load_vihsd()
                text_col = metadata["text_col"]
                label_col = metadata["label_col"]
            else:
                _, val_df, test_df, metadata = load_victsd()
                text_col = metadata["text_col"]
                label_col = metadata["label_col"]
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            print("  TIP: Use --data-file path/to/local.csv as fallback")
            sys.exit(1)

    test_texts = test_df[text_col].tolist()
    test_labels = test_df[label_col].values
    val_texts = val_df[text_col].tolist()
    val_labels = val_df[label_col].values

    print(f"  Test samples: {len(test_texts)}")
    print(f"  Val samples: {len(val_texts)}")

    # [3/6] Get individual predictions
    print(f"\n[3/6] Running individual model predictions...")
    individual_preds = {}
    individual_val_preds = {}

    for name, info in ensemble.models.items():
        print(f"  Predicting with {name}...")
        if info["type"] == "t5":
            preds = ensemble.predict_t5(info, test_texts, task_prefix, label_map, batch_size=args.batch_size)
            val_preds = ensemble.predict_t5(info, val_texts, task_prefix, label_map, batch_size=args.batch_size)
        else:
            preds, _ = ensemble.predict_bert(info, test_texts, batch_size=args.batch_size)
            val_preds, _ = ensemble.predict_bert(info, val_texts, batch_size=args.batch_size)
            if info.get("label_remap"):
                preds = np.array([info["label_remap"].get(int(p), int(p)) for p in preds])
                val_preds = np.array([info["label_remap"].get(int(p), int(p)) for p in val_preds])
        individual_preds[name] = preds
        individual_val_preds[name] = val_preds
        f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        print(f"    Macro F1: {f1:.4f}")

    # [4/6] Optimize weights
    print(f"\n[4/6] Optimizing weights...")
    if args.weights:
        if len(args.weights) != len(ensemble.models):
            print(f"ERROR: --weights requires {len(ensemble.models)} values, got {len(args.weights)}")
            sys.exit(1)
        for (name, _), w in zip(ensemble.models.items(), args.weights):
            ensemble.weights[name] = w
        print("  Using manual weights:", {n: w for n, w in zip(ensemble.models.keys(), args.weights)})
    elif args.no_optimize:
        print("  Skipping optimization (equal weights)")
    else:
        optimal_weights = optimize_weights(
            individual_val_preds, val_labels, num_classes, n_trials=1000
        )
        for name, w in optimal_weights.items():
            ensemble.weights[name] = w

    # [5/6] Compute ensemble predictions
    print(f"\n[5/6] Computing ensemble predictions...")
    weighted_preds = ensemble.weighted_vote(individual_preds, num_classes)
    majority_preds = ensemble.majority_vote(individual_preds, num_classes)

    weighted_f1 = f1_score(test_labels, weighted_preds, average="macro", zero_division=0)
    majority_f1 = f1_score(test_labels, majority_preds, average="macro", zero_division=0)
    print(f"  Weighted vote Macro F1: {weighted_f1:.4f}")
    print(f"  Majority vote Macro F1: {majority_f1:.4f}")

    # [6/6] Evaluation & save results
    print(f"\n[6/6] Evaluation...")
    evaluate_ensemble(weighted_preds, test_labels, individual_preds, args.task.upper())

    # Build results DataFrame
    rows = []
    label_names = list(label_map.keys())

    for name, preds in individual_preds.items():
        acc = accuracy_score(test_labels, preds)
        macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        per_class = f1_score(test_labels, preds, average=None, zero_division=0, labels=range(num_classes))
        row = {"Model": name, "Task": args.task, "Method": "individual", "Accuracy": acc, "Macro_F1": macro_f1}
        for i, lbl in enumerate(label_names):
            row[f"F1_{lbl}"] = per_class[i] if i < len(per_class) else 0.0
        rows.append(row)

    # Ensemble weighted
    acc_w = accuracy_score(test_labels, weighted_preds)
    per_class_w = f1_score(test_labels, weighted_preds, average=None, zero_division=0, labels=range(num_classes))
    row_w = {"Model": "Ensemble", "Task": args.task, "Method": "weighted", "Accuracy": acc_w, "Macro_F1": weighted_f1}
    for i, lbl in enumerate(label_names):
        row_w[f"F1_{lbl}"] = per_class_w[i] if i < len(per_class_w) else 0.0
    rows.append(row_w)

    # Ensemble majority
    acc_m = accuracy_score(test_labels, majority_preds)
    per_class_m = f1_score(test_labels, majority_preds, average=None, zero_division=0, labels=range(num_classes))
    row_m = {"Model": "Ensemble", "Task": args.task, "Method": "majority", "Accuracy": acc_m, "Macro_F1": majority_f1}
    for i, lbl in enumerate(label_names):
        row_m[f"F1_{lbl}"] = per_class_m[i] if i < len(per_class_m) else 0.0
    rows.append(row_m)

    results_df = pd.DataFrame(rows)

    # Save CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n  Results saved to: {args.output}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
