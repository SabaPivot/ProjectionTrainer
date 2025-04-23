#! /bin/bash

# --- Configuration for Stage 0: Vision Encoder SigLIP Fine-tuning ---

# --- Model ---
# Paper uses SigLIP-Large (ViT-L/16) pre-trained on WebLi, extended to 512 input.
MODEL_NAME="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
TRUST_REMOTE_CODE_FLAG="--trust_remote_code" # Add this flag if the model requires it

# --- Dataset (Image-Text Pairs) ---
# Paper uses 1,052,257 image-text pairs from CheXinstruct.
TRAIN_JSON="/home/compu/DATA/CXR_VDQA/Train/formatted_VD_class.json"
IMAGE_ROOT="/home/compu/DATA/NIH Chest X-rays_jpg"

# --- Training Hyperparameters ---
BATCH_SIZE=8
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
NUM_EPOCHS=10
GRAD_ACCUM_STEPS=4
WARMUP_RATIO=0.05
MAX_TEXT_LEN=128

# --- Freezing Strategy ---
# By default, text encoder and logit scale are frozen
# We can optionally unfreeze with these flags
# FREEZE_FLAGS="--no_freeze_text_encoder"  # Uncomment to train text encoder
# FREEZE_FLAGS="--no_freeze_logit_scale"   # Uncomment to train logit scale
FREEZE_FLAGS=""  # No flags means keep defaults (freeze both)
FREEZE_LAYERS_RATIO=0.0 # 0.0 = train all vision layers

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
    --freeze_layers_ratio $FREEZE_LAYERS_RATIO \
    $FREEZE_FLAGS \
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