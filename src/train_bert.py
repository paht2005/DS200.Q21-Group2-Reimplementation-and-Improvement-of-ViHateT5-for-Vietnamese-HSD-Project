"""
Training script for Vietnamese hate speech detection.

Usage:
    python src/train.py --dataset ViHSD --epochs 10 --batch_size 16
"""

import argparse
import os
import time
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dotenv import load_dotenv

from config import TrainConfig
from data_loader import load_dataset_by_name, build_torch_dataset
from model import build_model
from utils import set_seed, evaluate, train_epoch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train hate speech detection model")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset to train on (ViHSD, ViCTSD, ViHOS, ViHSD_processed, NCPhat2005/VOZ-HSD_2M)")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base",
                       help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for model")
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = TrainConfig(
        dataset_name=args.dataset,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    
    # Set seed
    set_seed(config.seed)
    
    print("=" * 80)
    print(f"Training Configuration:")
    print("=" * 80)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📚 Loading {config.dataset_name} dataset...")
    train_df, val_df, test_df, metadata = load_dataset_by_name(config.dataset_name)
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Text column: {metadata['text_col']}")
    print(f"  Label column: {metadata['label_col']}")
    print(f"  Number of labels: {metadata['num_labels']}")
    
    # Build model and tokenizer
    print(f"\n🤖 Building model: {config.model_name}")
    model, tokenizer = build_model(
        config.model_name,
        metadata["num_labels"],
        config.device
    )
    
    # Build datasets
    print("\n🔨 Building PyTorch datasets...")
    train_dataset = build_torch_dataset(
        train_df, metadata["text_col"], metadata["label_col"],
        tokenizer, config.max_length
    )
    val_dataset = build_torch_dataset(
        val_df, metadata["text_col"], metadata["label_col"],
        tokenizer, config.max_length
    )
    test_dataset = build_torch_dataset(
        test_df, metadata["text_col"], metadata["label_col"],
        tokenizer, config.max_length
    )
    
    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    print(f"  Total training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    
    # Training loop
    print(f"\n🚀 Starting training on {config.device}...")
    print("=" * 80)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epoch_seconds": [],
        "gpu_reserved_gb": [],
        "gpu_allocated_gb": [],
        "lr": [],
    }
    
    gpu_device_index = None
    if torch.cuda.is_available() and "cuda" in str(config.device):
        gpu_device_index = torch.cuda.current_device()
    
    training_start = time.time()
    
    for epoch in range(1, config.epochs + 1):
        if gpu_device_index is not None:
            torch.cuda.reset_peak_memory_stats(gpu_device_index)
        
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.device)
        
        # Validate
        val_preds, val_labels, val_loss = evaluate(model, val_loader, config.device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        epoch_time = time.time() - epoch_start
        
        # GPU memory tracking
        reserved_gb = 0.0
        allocated_gb = 0.0
        if gpu_device_index is not None:
            torch.cuda.synchronize()
            reserved_gb = torch.cuda.max_memory_reserved(gpu_device_index) / (1024 ** 3)
            allocated_gb = torch.cuda.max_memory_allocated(gpu_device_index) / (1024 ** 3)
        
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epoch_seconds"].append(epoch_time)
        history["gpu_reserved_gb"].append(reserved_gb)
        history["gpu_allocated_gb"].append(allocated_gb)
        history["lr"].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.epochs} | ⏱️  {epoch_time:.2f}s")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.8f}")
        if gpu_device_index is not None:
            print(f"  GPU peak: reserved {reserved_gb:.3f} GB | allocated {allocated_gb:.3f} GB")
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  🎯 New best macro F1: {best_val_f1:.4f}. Saving checkpoint...")
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                print("  🛑 Early stopping triggered.")
                break
    
    training_time = time.time() - training_start
    print(f"\n✅ Training finished in {training_time/60:.2f} minutes.")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    
    # Test evaluation
    print("\n🔍 Evaluating on test set...")
    from model import load_trained_model
    best_model, _ = load_trained_model(str(config.output_dir), config.device)
    
    test_preds, test_labels, test_loss = evaluate(best_model, test_loader, config.device)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    
    print(f"  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4))
    
    # Save metrics
    print("\n💾 Saving metrics...")
    
    # Save epoch metrics
    epoch_rows = []
    for idx in range(len(history["train_loss"])):
        epoch_rows.append({
            "epoch": idx + 1,
            "train_loss": history["train_loss"][idx],
            "val_loss": history["val_loss"][idx],
            "val_acc": history["val_acc"][idx],
            "val_f1": history["val_f1"][idx],
            "epoch_seconds": history["epoch_seconds"][idx],
            "learning_rate": history["lr"][idx],
        })
    
    epoch_metrics_df = pd.DataFrame(epoch_rows)
    epoch_csv = config.output_dir / "epoch_metrics.csv"
    epoch_metrics_df.to_csv(epoch_csv, index=False)
    print(f"  Saved epoch metrics to {epoch_csv}")
    
    # Save run summary
    summary = {
        "dataset": config.dataset_name,
        "model": config.model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "training_minutes": training_time / 60,
        "epochs_trained": len(history["train_loss"]),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_csv = config.output_dir / "run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved run summary to {summary_csv}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
