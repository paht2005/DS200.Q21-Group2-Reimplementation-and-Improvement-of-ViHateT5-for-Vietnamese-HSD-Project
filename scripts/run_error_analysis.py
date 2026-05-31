"""
Multi-model error analysis CLI.
Runs error analysis for all fine-tuned models (T5 + BERT), caches predictions,
performs McNemar's statistical significance tests, and produces combined comparison chart.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from error_analysis import run_full_error_analysis, mcnemar_report, plot_combined_comparison
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


def get_all_finetuned_models():
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


def predict_t5(model_path, texts, task, device, batch_size):
    """Generate T5 predictions in batches."""
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()

    prefix = "hate-speech-detection" if task == "vihsd" else "toxic-speech-detection"
    label_map = (
        {"clean": 0, "offensive": 1, "hate": 2}
        if task == "vihsd"
        else {"none": 0, "toxic": 1}
    )
    default_label = 0

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {t}" for t in texts[i:i + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for d in decoded:
            preds.append(label_map.get(d.strip().lower(), default_label))
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return preds


def predict_bert(model_path, texts, device, batch_size):
    """Generate BERT predictions in batches."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
        preds.extend(batch_preds)
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return preds


def get_predictions(model_path, texts, task, device, batch_size):
    """Unified prediction dispatcher with BERT label remapping."""
    model_type, config = detect_model_type(model_path)

    if model_type == "t5":
        return predict_t5(model_path, texts, task, device, batch_size)
    else:
        preds = predict_bert(model_path, texts, device, batch_size)
        # BERT label remap for vihsd: model outputs {0: clean, 1: hate} -> {0: clean, 2: hate}
        if task == "vihsd":
            num_labels = config.get("num_labels", len(config.get("id2label", {})))
            if num_labels == 2:
                label_remap = {0: 0, 1: 2}
                preds = [label_remap.get(p, p) for p in preds]
        return preds


def load_cached_predictions(model_name, task):
    """Load cached predictions from CSV if available."""
    cache_path = os.path.join("results", "analysis", f"{model_name}_{task}_predictions.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df["pred_label"].tolist()
    return None


def save_predictions(model_name, task, texts, true_labels, pred_labels):
    """Save predictions to CSV for reuse."""
    os.makedirs(os.path.join("results", "analysis"), exist_ok=True)
    df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "pred_label": pred_labels,
    })
    cache_path = os.path.join("results", "analysis", f"{model_name}_{task}_predictions.csv")
    df.to_csv(cache_path, index=False, encoding="utf-8")
    print(f"    Cached: {cache_path}")


def parse_args(args=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-model error analysis with McNemar tests and combined visualization"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model paths to analyze"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Use all fine-tuned models in models/ directory (default when --models not specified)"
    )
    parser.add_argument(
        "--task", choices=["vihsd", "victsd", "both"], default="both",
        help="Task to analyze (default: both)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-run inference even if cached predictions exist"
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Local CSV file to use as test data (columns: free_text, label_id). Bypasses HuggingFace dataset loading."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Base output directory (default: results)"
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    # [1/7] Device detection
    print("\n[1/7] Device detection")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # [2/7] Resolve model list
    print("\n[2/7] Resolving model list")
    if args.models:
        model_paths = args.models
    else:
        model_paths = get_all_finetuned_models()
    print(f"  Models ({len(model_paths)}):")
    for mp in model_paths:
        print(f"    - {mp}")

    # [3/7] Load test data
    print("\n[3/7] Loading test datasets")
    tasks_to_run = []
    test_data = {}

    if args.task in ("vihsd", "both"):
        if args.data_file:
            # Local CSV fallback
            local_df = pd.read_csv(args.data_file)
            test_data["vihsd"] = {
                "texts": local_df["free_text"].tolist(),
                "true_labels": local_df["label_id"].tolist(),
            }
            print(f"  ViHSD test (local): {len(local_df)} samples from {args.data_file}")
        else:
            _, _, vihsd_test, _ = load_vihsd()
            test_data["vihsd"] = {
                "texts": vihsd_test["free_text"].tolist(),
                "true_labels": vihsd_test["label_id"].tolist(),
            }
            print(f"  ViHSD test: {len(vihsd_test)} samples")
        tasks_to_run.append("vihsd")

    if args.task in ("victsd", "both"):
        if args.data_file:
            print("  ViCTSD: skipped (--data-file only supports vihsd format)")
        else:
            _, _, victsd_test, _ = load_victsd()
            test_data["victsd"] = {
                "texts": victsd_test["Comment"].tolist(),
                "true_labels": victsd_test["Toxicity"].tolist(),
            }
            tasks_to_run.append("victsd")
            print(f"  ViCTSD test: {len(victsd_test)} samples")

    # [4/7] Generate/load predictions
    print("\n[4/7] Generating predictions")
    predictions = {}  # predictions[model_name][task] = list[int]

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        predictions[model_name] = {}

        for task in tasks_to_run:
            # Check cache
            if not args.no_cache:
                cached = load_cached_predictions(model_name, task)
                if cached is not None:
                    predictions[model_name][task] = cached
                    print(f"  {model_name} on {task} — cached ({len(cached)} preds)")
                    continue

            # Run inference
            print(f"  {model_name} on {task} — computing...")
            texts = test_data[task]["texts"]
            true_labels = test_data[task]["true_labels"]
            preds = get_predictions(model_path, texts, task, device, args.batch_size)
            predictions[model_name][task] = preds
            save_predictions(model_name, task, texts, true_labels, preds)

    # [5/7] Per-model error analysis
    print("\n[5/7] Running per-model error analysis")
    for model_name in predictions:
        for task in tasks_to_run:
            if task not in predictions[model_name]:
                continue
            preds = predictions[model_name][task]
            texts = test_data[task]["texts"]
            true_labels = test_data[task]["true_labels"]

            run_full_error_analysis(
                vihsd_true=true_labels,
                vihsd_pred=preds,
                vihsd_texts=texts,
                model_name=model_name,
            )

    # [6/7] McNemar's tests
    print("\n[6/7] McNemar's statistical significance tests")
    baseline = "vit5_finetune_balanced"
    challengers = [m for m in predictions if m != baseline]
    test_pairs = [(baseline, c) for c in challengers]

    for task in tasks_to_run:
        task_preds = {m: predictions[m][task] for m in predictions if task in predictions[m]}
        true_labels = test_data[task]["true_labels"]
        mcnemar_report(
            test_pairs=test_pairs,
            y_true=true_labels,
            predictions_dict=task_preds,
            task_name=task,
            filename=f"mcnemar_results_{task}.csv",
        )

    # [7/7] Combined comparison chart
    print("\n[7/7] Combined comparison chart")
    chart_rows = []

    for model_name in predictions:
        for task in tasks_to_run:
            if task not in predictions[model_name]:
                continue
            true_labels = test_data[task]["true_labels"]
            preds = predictions[model_name][task]
            macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
            chart_rows.append({
                "model_name": model_name,
                "task": task,
                "macro_f1": round(macro_f1, 4),
            })

    # Add ensemble results if available
    ensemble_path = os.path.join(args.output_dir, "ensemble_results.csv")
    if os.path.exists(ensemble_path):
        ens_df = pd.read_csv(ensemble_path)
        for task in tasks_to_run:
            task_key = "vihsd" if task == "vihsd" else "victsd"
            task_rows = ens_df[ens_df["Task"].str.lower() == task_key]
            if not task_rows.empty:
                best = task_rows.loc[task_rows["Macro_F1"].idxmax()]
                chart_rows.append({
                    "model_name": f"ensemble ({best['Method']})",
                    "task": task,
                    "macro_f1": round(best["Macro_F1"], 4),
                })

    results_df = pd.DataFrame(chart_rows)
    print("\n  Summary:")
    print(results_df.to_string(index=False))

    plot_combined_comparison(
        results_df,
        output_path=os.path.join(args.output_dir, "images", "combined_comparison.png"),
    )

    # Final summary
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 80)
    images_dir = os.path.join(args.output_dir, "images")
    analysis_dir = os.path.join(args.output_dir, "analysis")
    if os.path.isdir(images_dir):
        imgs = os.listdir(images_dir)
        print(f"  results/images/: {len(imgs)} files")
    if os.path.isdir(analysis_dir):
        csvs = os.listdir(analysis_dir)
        print(f"  results/analysis/: {len(csvs)} files")


if __name__ == "__main__":
    main()
