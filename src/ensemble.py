"""
Ensemble methods for Vietnamese hate speech detection.

Combines predictions from multiple models (BERT-based + T5-based)
using weighted voting or stacking for improved performance.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, AutoTokenizer as T5Tokenizer,
)
from tqdm import tqdm


class HateSpeechEnsemble:
    """
    Ensemble combining encoder (BERT-based) and decoder (T5-based) models
    for Vietnamese hate speech detection.

    Supports:
    - Majority voting
    - Weighted voting (based on per-class F1)
    - Soft voting (probability averaging)
    """

    def __init__(self, device: str = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = {}
        self.weights = {}

    def add_t5_model(self, name: str, model_id: str, weight: float = 1.0):
        """Add a T5-based model to the ensemble."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id)
        model.to(self.device)
        model.eval()
        self.models[name] = {"type": "t5", "model": model, "tokenizer": tokenizer}
        self.weights[name] = weight

    def add_bert_model(self, name: str, model_path: str, num_labels: int, weight: float = 1.0,
                       label_remap: Optional[Dict[int, int]] = None):
        """Add a BERT-based model to the ensemble."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        model.to(self.device)
        model.eval()
        self.models[name] = {"type": "bert", "model": model, "tokenizer": tokenizer, "label_remap": label_remap}
        self.weights[name] = weight

    def predict_t5(self, model_info: dict, texts: List[str], task_prefix: str,
                   label_map: Dict[str, int], batch_size: int = 32) -> np.ndarray:
        """Get predictions from a T5 model."""
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch = [f"{task_prefix}: {t}" for t in texts[i:i + batch_size]]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text in decoded:
                text_clean = text.strip().upper()
                predictions.append(label_map.get(text_clean, -1))

        return np.array(predictions)

    def predict_bert(self, model_info: dict, texts: List[str],
                     batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities from a BERT model."""
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        all_preds = []
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def weighted_vote(self, all_predictions: Dict[str, np.ndarray],
                      num_classes: int) -> np.ndarray:
        """Weighted majority voting across model predictions."""
        n_samples = len(next(iter(all_predictions.values())))
        weighted_votes = np.zeros((n_samples, num_classes))

        for name, preds in all_predictions.items():
            weight = self.weights.get(name, 1.0)
            for i, pred in enumerate(preds):
                if 0 <= pred < num_classes:
                    weighted_votes[i, pred] += weight

        return np.argmax(weighted_votes, axis=1)

    def majority_vote(self, all_predictions: Dict[str, np.ndarray],
                      num_classes: int) -> np.ndarray:
        """Simple majority voting."""
        equal_weights = {name: 1.0 for name in all_predictions}
        old_weights = self.weights.copy()
        self.weights = equal_weights
        result = self.weighted_vote(all_predictions, num_classes)
        self.weights = old_weights
        return result

    def predict_vihsd(self, texts: List[str], method: str = "weighted") -> np.ndarray:
        """Ensemble prediction for ViHSD task."""
        label_map = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
        all_preds = {}

        for name, info in self.models.items():
            if info["type"] == "t5":
                preds = self.predict_t5(info, texts, "hate-speech-detection", label_map)
            else:
                preds, _ = self.predict_bert(info, texts)
                if info.get("label_remap"):
                    preds = np.array([info["label_remap"].get(int(p), int(p)) for p in preds])
            all_preds[name] = preds

        if method == "weighted":
            return self.weighted_vote(all_preds, num_classes=3)
        else:
            return self.majority_vote(all_preds, num_classes=3)

    def predict_victsd(self, texts: List[str], method: str = "weighted") -> np.ndarray:
        """Ensemble prediction for ViCTSD task."""
        label_map = {"NONE": 0, "TOXIC": 1}
        all_preds = {}

        for name, info in self.models.items():
            if info["type"] == "t5":
                preds = self.predict_t5(info, texts, "toxic-speech-detection", label_map)
            else:
                preds, _ = self.predict_bert(info, texts)
                if info.get("label_remap"):
                    preds = np.array([info["label_remap"].get(int(p), int(p)) for p in preds])
            all_preds[name] = preds

        if method == "weighted":
            return self.weighted_vote(all_preds, num_classes=2)
        else:
            return self.majority_vote(all_preds, num_classes=2)


def optimize_weights(
    models_predictions: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    num_classes: int,
    n_trials: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Find optimal ensemble weights via random search on validation set.

    Args:
        models_predictions: Dict of model_name -> predictions array.
        true_labels: Ground truth labels.
        num_classes: Number of classes.
        n_trials: Number of random weight combinations to try.
        seed: Random seed.

    Returns:
        Dict of model_name -> optimal weight.
    """
    rng = np.random.RandomState(seed)
    model_names = list(models_predictions.keys())
    n_models = len(model_names)

    best_f1 = -1
    best_weights = {name: 1.0 for name in model_names}

    for _ in range(n_trials):
        # Random weights from Dirichlet distribution
        raw_weights = rng.dirichlet(np.ones(n_models))
        weights = {name: w for name, w in zip(model_names, raw_weights)}

        # Weighted voting
        n_samples = len(true_labels)
        weighted_votes = np.zeros((n_samples, num_classes))
        for name, preds in models_predictions.items():
            for i, pred in enumerate(preds):
                if 0 <= pred < num_classes:
                    weighted_votes[i, pred] += weights[name]
        ensemble_preds = np.argmax(weighted_votes, axis=1)

        f1 = f1_score(true_labels, ensemble_preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights.copy()

    print(f"\n  Optimal weights (Macro F1 = {best_f1:.4f}):")
    for name, w in best_weights.items():
        print(f"    {name}: {w:.4f}")

    return best_weights


def evaluate_ensemble(
    ensemble_preds: np.ndarray,
    true_labels: np.ndarray,
    individual_preds: Dict[str, np.ndarray],
    task_name: str,
):
    """Compare ensemble performance against individual models."""
    print(f"\n  {'=' * 60}")
    print(f"  {task_name} — Ensemble vs Individual Models")
    print(f"  {'=' * 60}")

    # Ensemble metrics
    ens_f1 = f1_score(true_labels, ensemble_preds, average="macro", zero_division=0)
    ens_acc = accuracy_score(true_labels, ensemble_preds)
    print(f"  {'Ensemble':<25} Macro F1: {ens_f1:.4f}  Accuracy: {ens_acc:.4f}")

    # Individual metrics
    for name, preds in individual_preds.items():
        ind_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        ind_acc = accuracy_score(true_labels, preds)
        delta = ens_f1 - ind_f1
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {name:<25} Macro F1: {ind_f1:.4f}  Accuracy: {ind_acc:.4f}  ({arrow}{abs(delta):.4f})")

    print(f"  {'=' * 60}")
