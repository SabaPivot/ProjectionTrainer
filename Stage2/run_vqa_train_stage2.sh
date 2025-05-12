#! /bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3 # Adjust GPU IDs as needed
# --- Configuration --- #
RUN_NAME="VD_Class_20_lr5e-5_QWEN3-8B-QLoRA_vit-l-384-QA_BALANCED-MIMIC" # Choose a descriptive name for your run
STAGE1_RUN_NAME="VD_Class_20_lr5e-5_QWEN3-8B-QLoRA_vit-l-384-QA_BALANCED-MIMIC" # Name of the Stage 1 run directory

# --- Paths --- #
# Adjust these paths according to your setup
TRAIN_JSON="/mnt/samuel/Siglip/filtered_formatted_Class_QA.json" # Path to the combined JSON file
VAL_JSON="/mnt/WHY/VLM/Deepseek/VLM-R1/QA_DATASET/Train/Atelectasis+Cardiomegaly+Effusion/QA_Balancing/transformed_val.json" # Path to your VQA JSON data
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg" # Path to your images
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"  # Path to your secondary image root for MIMIC-CXR images
STAGE1_PROJECTOR_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage1/trained_projection_stage1/VD_lr3e-5_qwen3-8b-QLoRA-Load_l-384-10" # Path to the saved Stage 1 projector
OUTPUT_DIR="./trained_vqa_stage2/${RUN_NAME}" # Output directory for this Stage 2 run

# --- Model Names --- #
VISION_MODEL="/mnt/samuel/Siglip/soombit/checkpoint/epoch_16"
LLM_MODEL="Qwen/Qwen3-8B"

# --- Path to optionally resume QLoRA training ---
# Set this path to the directory containing the adapter_model.safetensors (or .bin)
# e.g., "/path/to/your/trained_vqa_stage2/RUN_NAME/checkpoint-epoch_X/language_model"
RESUME_QLORA_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr5e-5_QWEN3-8B-QLoRA_vit-l-384-QA_BALANCED-MIMIC/checkpoint-epoch_2/language_model" # Leave empty to train QLoRA from scratch

# --- Hyperparameters --- #
# Adjust these based on your GPU memory and dataset size
BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=1e-5
GRAD_ACCUM_STEPS=8    
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
IMG_SIZE=384
MAX_Q_LEN=256
MAX_A_LEN=1024

# --- Freezing Options --- #
# Set to true to fine-tune these components alongside the LLM
UNFREEZE_PROJECTOR=false # Common to fine-tune the projector
# UNFREEZE_LLM=true       # This is ignored if ENABLE_QLORA is true
ENABLE_QLORA=true       # Set to true to enable QLoRA

freeze_proj_arg=""
if [ "$UNFREEZE_PROJECTOR" = true ]; then
  freeze_proj_arg="--unfreeze_projection_layer"
fi

# Logic for QLoRA flag
qlora_arg=""
if [ "$ENABLE_QLORA" = true ]; then
  qlora_arg="--enable_qlora"
fi

# Logic for resuming QLoRA adapter
resume_qlora_arg=""
if [ -n "$RESUME_QLORA_PATH" ] && [ "$ENABLE_QLORA" = true ]; then
  # Check if the specified path exists
  if [ -d "$RESUME_QLORA_PATH" ]; then
    resume_qlora_arg="--resume_qlora_adapter_path $RESUME_QLORA_PATH"
    echo "INFO: Resuming QLoRA training from adapter at $RESUME_QLORA_PATH"
  else
    echo "WARNING: RESUME_QLORA_PATH specified but directory not found: $RESUME_QLORA_PATH. Training QLoRA from scratch."
    # Optionally, exit here if resuming is mandatory:
    # echo "ERROR: Adapter path not found. Exiting." >&2; exit 1
  fi # This fi closes the inner if
elif [ -n "$RESUME_QLORA_PATH" ] && [ "$ENABLE_QLORA" = false ]; then # This elif correctly pairs with the outer if
    echo "WARNING: RESUME_QLORA_PATH is set, but ENABLE_QLORA is false. Path will be ignored."
fi # This fi closes the outer if

# --- Accelerator Launch --- #
# Using bf16 as QLoRA compute dtype will be bf16 or fp16
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
    $qlora_arg \
    $resume_qlora_arg \
    --wandb_project "xray_vqa_training_stage2" \
    --wandb_run_name "$RUN_NAME"

echo "Stage 2 Training Script finished." 