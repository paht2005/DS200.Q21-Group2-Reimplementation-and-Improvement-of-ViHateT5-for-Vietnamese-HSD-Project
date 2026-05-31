#!/bin/bash
# Script to run BERT-based model training
# Usage: bash scripts/run_train_bert.sh

# Default values
DATASET="ViHSD"
MODEL_NAME="vinai/phobert-base"
MAX_LENGTH=256
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
PATIENCE=3
SEED=42
OUTPUT_DIR="outputs/bert_${DATASET}_$(date +%Y%m%d_%H%M%S)"
AUGMENT_MINORITY=false
AUGMENT_FACTOR=0.8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --augment_minority)
            AUGMENT_MINORITY=true
            shift 1
            ;;
        --augment_factor)
            AUGMENT_FACTOR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_train_bert.sh [--dataset DATASET] [--model_name MODEL] [--max_length N] [--batch_size N] [--epochs N] [--learning_rate LR] [--weight_decay WD] [--warmup_ratio RATIO] [--patience N] [--seed N] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "BERT Training Configuration:"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "Max length: $MAX_LENGTH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Warmup ratio: $WARMUP_RATIO"
echo "Patience: $PATIENCE"
echo "Seed: $SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Augment minority: $AUGMENT_MINORITY"
echo "Augment factor: $AUGMENT_FACTOR"
echo "=========================================="
echo ""

# Build augmentation arguments conditionally
AUGMENT_ARGS=""
if [ "$AUGMENT_MINORITY" = true ]; then
    AUGMENT_ARGS="--augment_minority --augment_factor $AUGMENT_FACTOR"
fi

# Run training
python src/train_bert.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    $AUGMENT_ARGS

echo ""
echo "✅ Training completed! Model saved to: $OUTPUT_DIR"

