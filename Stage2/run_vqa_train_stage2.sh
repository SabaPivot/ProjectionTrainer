#! /bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3 # Adjust GPU IDs as needed
# --- Configuration --- #
RUN_NAME="VD_Class_20_lr5e-5_cogito-v1-3b_vit-l-384-QA_BALANCED-MIMIC" # Choose a descriptive name for your run
STAGE1_RUN_NAME="VD_Class_20_lr5e-5_cogito-v1-3b_vit-l-384-QA_BALANCED-MIMIC" # Name of the Stage 1 run directory

# --- Paths --- #
# Adjust these paths according to your setup
TRAIN_JSON="/mnt/samuel/Siglip/combined_train.json" # Path to the combined JSON file
VAL_JSON="/mnt/WHY/VLM/Deepseek/VLM-R1/QA_DATASET/Train/Atelectasis+Cardiomegaly+Effusion/QA_Balancing/transformed_val.json" # Path to your VQA JSON data
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg" # Path to your images
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"  # Path to your secondary image root for MIMIC-CXR images
STAGE1_PROJECTOR_PATH="/mnt/samuel/Siglip/ProjectionTrainer/0417Stage1" # Path to the saved Stage 1 projector
OUTPUT_DIR="./trained_vqa_stage2/${RUN_NAME}" # Output directory for this Stage 2 run

# --- Model Names --- #
VISION_MODEL="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
LLM_MODEL="deepcogito/cogito-v1-preview-llama-3B"

# --- Hyperparameters --- #
# Adjust these based on your GPU memory and dataset size
BATCH_SIZE=1
NUM_EPOCHS=10
LEARNING_RATE=1e-5
GRAD_ACCUM_STEPS=8    
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
IMG_SIZE=384
MAX_Q_LEN=256
MAX_A_LEN=1024

# --- Freezing Options --- #
# Set to true to fine-tune these components alongside the LLM
UNFREEZE_PROJECTOR=true # Common to fine-tune the projector
UNFREEZE_LLM=true       # Must be true to fine-tune the LLM  

freeze_proj_arg=""
if [ "$UNFREEZE_PROJECTOR" = true ]; then
  freeze_proj_arg="--unfreeze_projection_layer"
fi

freeze_llm_arg=""
if [ "$UNFREEZE_LLM" = true ]; then
  freeze_llm_arg="--unfreeze_llm"
fi

# --- Accelerator Launch --- #
# Restore bf16 mixed precision
accelerate launch --mixed_precision bf16 train_vqa_stage2.py \
    --train_json "$TRAIN_JSON" \
    --val_json "$VAL_JSON" \
    --image_root "$IMAGE_ROOT" \
    --image_root_2 "$IMAGE_ROOT_2" \
    --vision_model_name "$VISION_MODEL" \
    --llm_name "$LLM_MODEL" \
    --stage1_projector_path "$STAGE1_PROJECTOR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --img_size $IMG_SIZE \
    --max_q_len $MAX_Q_LEN \
    --max_a_len $MAX_A_LEN \
    $freeze_proj_arg \
    $freeze_llm_arg \
    --wandb_project "xray_vqa_training_stage2" \
    --wandb_run_name "$RUN_NAME" \
    --train_ve_first_epoch

echo "Stage 2 Training Script finished." 