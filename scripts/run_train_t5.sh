#!/bin/bash
# Script to run T5 fine-tuning
# Usage: bash scripts/run_train_t5.sh

# Default values
SAVE_MODEL_NAME="ViHateT5-finetuned"
PRE_TRAINED_CKPT="VietAI/vit5-base"
OUTPUT_DIR="outputs/t5_finetuned"
BATCH_SIZE=32
NUM_EPOCHS=4
LEARNING_RATE=2e-4
MAX_LENGTH=256
GRAD_ACC_STEPS=1
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.0
LR_SCHEDULER="constant"
SEED=42
GPU="0"
USE_FOCAL_LOSS=false
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.0
AUGMENT_MINORITY=false
AUGMENT_FACTOR=0.8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --save_model_name)
            SAVE_MODEL_NAME="$2"
            shift 2
            ;;
        --pre_trained_ckpt)
            PRE_TRAINED_CKPT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRAD_ACC_STEPS="$2"
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
        --lr_scheduler_type)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --use_focal_loss)
            USE_FOCAL_LOSS=true
            shift 1
            ;;
        --focal_gamma)
            FOCAL_GAMMA="$2"
            shift 2
            ;;
        --label_smoothing)
            LABEL_SMOOTHING="$2"
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
            echo "Usage: bash scripts/run_train_t5.sh [options]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "T5 Fine-tuning Configuration:"
echo "=========================================="
echo "Save model name: $SAVE_MODEL_NAME"
echo "Pre-trained checkpoint: $PRE_TRAINED_CKPT"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max length: $MAX_LENGTH"
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"
echo "Weight decay: $WEIGHT_DECAY"
echo "Warmup ratio: $WARMUP_RATIO"
echo "LR scheduler: $LR_SCHEDULER"
echo "Seed: $SEED"
echo "GPU: $GPU"
echo "Use focal loss: $USE_FOCAL_LOSS"
echo "Focal gamma: $FOCAL_GAMMA"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Augment minority: $AUGMENT_MINORITY"
echo "Augment factor: $AUGMENT_FACTOR"
echo "=========================================="
echo ""

# Build focal loss arguments conditionally
FOCAL_ARGS=""
if [ "$USE_FOCAL_LOSS" = true ]; then
    FOCAL_ARGS="--use_focal_loss --focal_gamma $FOCAL_GAMMA --label_smoothing $LABEL_SMOOTHING"
fi

# Build augmentation arguments conditionally
AUGMENT_ARGS=""
if [ "$AUGMENT_MINORITY" = true ]; then
    AUGMENT_ARGS="--augment_minority --augment_factor $AUGMENT_FACTOR"
fi

# Run training
python src/train_t5.py \
    --save_model_name "$SAVE_MODEL_NAME" \
    --pre_trained_ckpt "$PRE_TRAINED_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --seed $SEED \
    --gpu "$GPU" \
    $FOCAL_ARGS \
    $AUGMENT_ARGS

echo ""
echo "✅ Fine-tuning completed! Model saved to: $OUTPUT_DIR"
