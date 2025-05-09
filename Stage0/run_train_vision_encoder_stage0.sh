#! /bin/bash

# --- Configuration for Stage 0: Vision Encoder SigLIP Fine-tuning ---

# --- Model ---
MODEL_NAME="StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli"
TRUST_REMOTE_CODE_FLAG="--trust_remote_code" # Add this flag if the model requires it

# --- Dataset (Image-Text Pairs) ---
TRAIN_JSON="/home/compu/samuel/ProjectionTrainer/Stage1/data/single_label_image_caption.json" # "/home/compu/samuel/ProjectionTrainer/Stage0/combined_transformed_data.json"" # "/home/compu/samuel/ProjectionTrainer/Stage0/combined_transformed_data.json"
IMAGE_ROOT="/home/compu/DATA/NIH Chest X-rays_jpg"

# --- Training Hyperparameters ---
BATCH_SIZE=8
LEARNING_RATE=3e-5
WEIGHT_DECAY=0.01
NUM_EPOCHS=30
GRAD_ACCUM_STEPS=4
WARMUP_RATIO=0.05
MAX_TEXT_LEN=128

# --- Output & Logging ---
RUN_NAME="SigLIP_FineTune_$(basename $MODEL_NAME)_lr${LEARNING_RATE}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./trained_vision_encoder_stage0/$RUN_NAME"
WANDB_PROJECT="vision_encoder_siglip_stage0"
SAVE_EVERY=1
LOGGING_STEPS=10 # Log every N global steps
LOG_WITH="wandb"  # or "tensorboard"
DISABLE_WANDB_FLAG="" # Use --disable_wandb to turn off
MIXED_PRECISION="bf16"  # "bf16", "fp16", or "no"
SEED=42

# --- Accelerator Configuration ---
NUM_GPUS=4 # Set to 1 for single-GPU debugging

# --- Display Configuration ---
echo "Starting Stage 0 SigLIP Fine-tuning Training with Accelerate"
echo "Model: $MODEL_NAME"
echo "Dataset: $TRAIN_JSON"
echo "Output Directory: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"

# --- Run Training with Accelerate Launch ---
# Note: Ensure no trailing whitespace after backslashes
accelerate launch --num_processes $NUM_GPUS --mixed_precision "$MIXED_PRECISION" train_vision_encoder_stage0.py \
    --model_name "$MODEL_NAME" \
    --train_json "$TRAIN_JSON" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_text_len $MAX_TEXT_LEN \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --save_every_n_epochs $SAVE_EVERY \
    --logging_steps $LOGGING_STEPS \
    --log_with "$LOG_WITH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    $DISABLE_WANDB_FLAG \
    $TRUST_REMOTE_CODE_FLAG \
    --seed $SEED

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Stage 0 SigLIP fine-tuning completed successfully."
else
    echo "Stage 0 SigLIP fine-tuning failed with exit code $EXIT_CODE."
fi
exit $EXIT_CODE