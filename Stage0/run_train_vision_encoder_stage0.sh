#! /bin/bash

# --- Debug mode for better error reporting ---
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=1,2,3
# --- Configuration for Stage 0: Vision Encoder SigLIP Fine-tuning ---

# --- Model ---
# Paper uses SigLIP-Large (ViT-L/16) pre-trained on WebLi, extended to 512 input.
MODEL_NAME="google/siglip2-so400m-patch16-512"
TRUST_REMOTE_CODE_FLAG="--trust_remote_code" # Add this flag if the model requires it

# --- Dataset (Image-Text Pairs) ---
TRAIN_JSON="/mnt/data/CXR/filtered/formatted/ultimate_combined_dataset_balanced_caption.json"
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"

# --- Augmentation ---
USE_ONLINE_AUGMENTATION=true # Set to true to enable online augmentation, false to disable
ONLINE_AUG_FLAG=""
if [ "$USE_ONLINE_AUGMENTATION" = true ] ; then
    ONLINE_AUG_FLAG="--use_online_augmentation"
fi


# --- Training Hyperparameters ---
BATCH_SIZE=16
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
NUM_EPOCHS=100
GRAD_ACCUM_STEPS=4
WARMUP_RATIO=0.05
MAX_TEXT_LEN=128

# --- Output & Logging ---
RUN_NAME="SigLIP_FineTune_$(basename $MODEL_NAME)_lr${LEARNING_RATE}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./trained_vision_encoder_stage0/$RUN_NAME"
WANDB_PROJECT="vision_encoder_siglip_stage0"
SAVE_EVERY=3 # Save model every N epochs (after min_save_epoch)
MIN_SAVE_EPOCH=5 # Don't save any models before this epoch
LOGGING_STEPS=10 # Log every N global steps
LOG_WITH="wandb"  # or "tensorboard"
DISABLE_WANDB_FLAG="" # Use --disable_wandb to turn off
MIXED_PRECISION="bf16"  # "bf16", "fp16", or "no"
SEED=42

# --- Accelerator Configuration ---
NUM_GPUS=3 # Set to 1 for single-GPU debugging

# --- Display Configuration ---
echo "Starting Stage 0 SigLIP Fine-tuning Training with torchrun"
echo "Model: $MODEL_NAME"
echo "Dataset: $TRAIN_JSON"
echo "Output Directory: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Will validate after each epoch"

# --- Run with torchrun instead of accelerate ---
torchrun --nproc_per_node=$NUM_GPUS \
    train_vision_encoder_stage0.py \
    --model_name "$MODEL_NAME" \
    --train_json "$TRAIN_JSON" \
    --image_root "$IMAGE_ROOT" \
    --image_root_2 "$IMAGE_ROOT_2" \
    --output_dir "$OUTPUT_DIR" \
    --max_text_len $MAX_TEXT_LEN \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --save_every_n_epochs $SAVE_EVERY \
    --min_save_epoch $MIN_SAVE_EPOCH \
    --logging_steps $LOGGING_STEPS \
    --log_with "$LOG_WITH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    $DISABLE_WANDB_FLAG \
    $TRUST_REMOTE_CODE_FLAG \
    $ONLINE_AUG_FLAG \
    --seed $SEED

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Stage 0 SigLIP fine-tuning completed successfully."
else
    echo "Stage 0 SigLIP fine-tuning failed with exit code $EXIT_CODE."
fi
exit $EXIT_CODE