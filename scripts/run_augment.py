"""Standalone data augmentation script for inspection and offline augmentation.

Usage:
    python scripts/run_augment.py --dataset ViHSD --target_ratio 0.8
    python scripts/run_augment.py --dataset ViHSD --target_ratio 0.8 --save data/augmented_vihsd.csv
"""
import argparse
import sys
sys.path.insert(0, '.')

from src.augment import augment_minority_classes
from src.data_loader import load_dataset_by_name


def main():
    parser = argparse.ArgumentParser(description='Augment minority classes in hate speech datasets')
    parser.add_argument('--dataset', type=str, default='ViHSD',
                       help='Dataset name (ViHSD, ViCTSD, ViHOS, ViHSD_processed)')
    parser.add_argument('--target_ratio', type=float, default=0.8,
                       help='Target ratio of minority to majority (0.0-1.0)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save augmented data to CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.target_ratio < 0.0 or args.target_ratio > 1.0:
        parser.error("--target_ratio must be between 0.0 and 1.0")

    print(f"\n{'='*50}")
    print(f"Data Augmentation - {args.dataset}")
    print(f"{'='*50}")
    print(f"Target ratio: {args.target_ratio}")
    print(f"Seed: {args.seed}")

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    train_df, _, _, metadata = load_dataset_by_name(args.dataset)
    print(f"Original samples: {len(train_df)}")

    # Show class distribution before
    label_col = metadata['label_col']
    print(f"\nClass distribution (before):")
    for label, count in train_df[label_col].value_counts().sort_index().items():
        print(f"  Class {label}: {count}")

    # Augment
    augmented_df = augment_minority_classes(
        train_df,
        text_col=metadata['text_col'],
        label_col=metadata['label_col'],
        target_ratio=args.target_ratio,
        seed=args.seed
    )

    # Show class distribution after
    print(f"\nAugmented samples: {len(augmented_df)}")
    print(f"\nClass distribution (after):")
    for label, count in augmented_df[label_col].value_counts().sort_index().items():
        print(f"  Class {label}: {count}")

    # Save if requested
    if args.save:
        augmented_df.to_csv(args.save, index=False)
        print(f"\nSaved augmented data to: {args.save}")

    print(f"\n{'='*50}")
    print("Done!")


if __name__ == '__main__':
    main()
