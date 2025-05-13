#!/bin/bash

# Basic configuration - Adjust paths if necessary
PYTHON_EXE="python" # Added Python executable variable
MODULE_NAME="soombit.train" # Module path to run
EVAL_MODULE_NAME="soombit.evaluate_experiment" # Evaluation module name
OUTPUT_BASE_DIR="./soombit/checkpoints_distributed"
COMBINED_RESULTS_FILE="./soombit/all_experiments_summary_distributed.tsv" # Central results file (TSV)
DATA_JSON="/mnt/samuel/Siglip/soombit/data/single_label_dataset.json"
PRIMARY_TEST_JSON="/mnt/samuel/Siglip/soombit/data/transformed_test_888.json"
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
EPOCHS=10
BASE_VISION_MODEL_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_siglip2-so400m-patch16-512_lr5e-5_bs16_ep300_20250512_134054/epoch_6"


NUM_GPUS=3  # Hard-code to use 3 GPUs
CUDA_DEVICES=(1 2 3)  # Specify which CUDA devices to use
echo "Using 3 GPUs: CUDA devices ${CUDA_DEVICES[*]}"

# Initialize Combined Results File
echo "Initializing combined results file: $COMBINED_RESULTS_FILE"
echo -e "ExpID\tBatchSize\tBestEpoch\tBestAcc\tBestAUC\tBestCheckpoint" > "$COMBINED_RESULTS_FILE"

# Function to run a single experiment
run_exp() {
  EXP_ID=$1
  CLASSES=$2       # This will likely be received with quotes, e.g., "Class A,Class B"
  FREEZE_MODE=$3
  HANDLE_ABNORMAL=$4
  FILTER_NF=$5
  BATCH_SIZE=$6    # Batch size parameter
  DEVICE_ID=$7     # CUDA device ID parameter

  # Update experiment ID to include batch size and device ID
  FULL_EXP_ID="${EXP_ID}_BS${BATCH_SIZE}_GPU${DEVICE_ID}"

  echo ""
  echo "========================================="
  echo "===== Starting Experiment: $FULL_EXP_ID ====="
  echo "========================================="
  # Clean outer quotes from CLASSES for logging and argument passing
  CLEAN_CLASSES=$(echo "$CLASSES" | sed 's/^\"//;s/\"$//')
  echo "Classes        : $CLEAN_CLASSES"
  echo "Freeze Mode    : $FREEZE_MODE"
  echo "Handle Abnormal: $HANDLE_ABNORMAL"
  echo "Filter NF      : $FILTER_NF"
  echo "Batch Size     : $BATCH_SIZE"
  echo "CUDA Device    : $DEVICE_ID"
  echo "Output Dir     : ${OUTPUT_BASE_DIR}/${FULL_EXP_ID}"
  # echo "Vision Checkpt : $VISION_ENCODER_CHECKPOINT" # Removed
  echo "Base Vision Model: $BASE_VISION_MODEL_PATH"
  echo "Test JSON (Base): $PRIMARY_TEST_JSON (Evaluation script will filter)"

  # --- Removed Test JSON Selection Logic --- #
  # The evaluation script will handle filtering based on CLEAN_CLASSES

  echo "-----------------------------------------"

  # Construct python command arguments in an array for safety
  CMD_ARGS=(
    "$PYTHON_EXE" -m "$MODULE_NAME"
    --exp_id "$FULL_EXP_ID"
    --class_names "$CLEAN_CLASSES"
    --freeze_mode "$FREEZE_MODE"
    --output_base_dir "$OUTPUT_BASE_DIR"
    --data_json "$DATA_JSON"
    --image_root "$IMAGE_ROOT"
    --image_root_2 "$IMAGE_ROOT_2"
    --epochs "$EPOCHS"
    --lr "5e-5"
    --bb_lr "1e-5"
    --batch_size "$BATCH_SIZE"
    --device_id "$DEVICE_ID"
    # Pass the base model path via the standard vision_model_name argument
    --vision_model_name "$BASE_VISION_MODEL_PATH"
  )

  # Add boolean flags if true
  if [ "$HANDLE_ABNORMAL" = "true" ]; then
    CMD_ARGS+=(--handle_abnormal)
  fi
  if [ "$FILTER_NF" = "true" ]; then
    CMD_ARGS+=(--filter_no_finding)
  fi

  # Execute the command directly using the array
  echo "Executing Training Command:"
  # Use printf to safely quote arguments for display
  printf "%q " "${CMD_ARGS[@]}"; echo
  echo "-----------------------------------------"
  "${CMD_ARGS[@]}" | cat # Execute directly and pipe through cat

  # Check exit status
  if [ $? -ne 0 ]; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!!!! Experiment $FULL_EXP_ID FAILED !!!!!!!!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # Optionally exit script on failure: exit 1
  else
    echo ""
    echo "========================================="
    echo "===== Finished Experiment: $FULL_EXP_ID ====="
    echo "========================================="

    # --- Run Evaluation Script and Capture Output --- #
    EXP_DIR="${OUTPUT_BASE_DIR}/${FULL_EXP_ID}"
    echo ""
    echo "--- Running Evaluation for $FULL_EXP_ID ---"
    echo "Evaluation Dir : $EXP_DIR"
    echo "Test JSON Base : $PRIMARY_TEST_JSON"
    # Determine the correct class names for evaluation based on HANDLE_ABNORMAL
    if [ "$HANDLE_ABNORMAL" = "true" ]; then
        # Check if No Finding was an original target class
        if [[ "$CLEAN_CLASSES" == *"No Finding"* ]]; then
            EVAL_TARGET_CLASSES="No Finding,Abnormal"
        else
            # If No Finding wasn't targeted, only Abnormal remains effectively
            EVAL_TARGET_CLASSES="Abnormal"
            echo "WARNING: handle_abnormal=true but 'No Finding' not in original classes. Evaluating only for 'Abnormal'."
        fi
    else
        EVAL_TARGET_CLASSES="$CLEAN_CLASSES" # Use original classes if not handling abnormal
    fi
    echo "Eval Classes   : $EVAL_TARGET_CLASSES" # Changed from CLEAN_CLASSES
    echo "-----------------------------------------"
    # Construct evaluation command arguments array
    EVAL_CMD_ARGS=(
        "$PYTHON_EXE" -m "$EVAL_MODULE_NAME" # Use module name variable
        --exp_dir "$EXP_DIR"
        --test_json "$PRIMARY_TEST_JSON" # Pass the base 4-class JSON
        # Pass the EFFECTIVE classes for filtering and evaluation consistency
        --eval_class_names "$EVAL_TARGET_CLASSES" # MODIFIED HERE
        --device_id "$DEVICE_ID"
    )
    # Execute evaluation safely
    echo "Executing Evaluation Command:"
    printf "%q " "${EVAL_CMD_ARGS[@]}"; echo
    echo "-----------------------------------------"
    EVAL_OUTPUT=$("${EVAL_CMD_ARGS[@]}" 2>&1) # Capture stdout and stderr

    # Check evaluation exit status
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!!!! Evaluation for $FULL_EXP_ID FAILED !!!!!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Evaluation Output:"
        echo "$EVAL_OUTPUT"
    else
        echo "--- Finished Evaluation for $FULL_EXP_ID ---"
        # Extract best result line and append to combined file
        # Assuming evaluation script still outputs "BEST_RESULT\tEpoch\tAcc\tAUC\tCheckpointPath"
        BEST_LINE=$(echo "$EVAL_OUTPUT" | grep "^BEST_RESULT")
        if [ -n "$BEST_LINE" ]; then
            # Remove the "BEST_RESULT\t" prefix (cut fields 2 onwards using tab delimiter)
            RESULT_DATA=$(echo "$BEST_LINE" | cut -f 2-)
            # Prepend the full ExpID (including BS and GPU) to the result data
            echo "Appending result to $COMBINED_RESULTS_FILE: $FULL_EXP_ID $RESULT_DATA"
            echo -e "$FULL_EXP_ID\t$BATCH_SIZE\t$RESULT_DATA" >> "$COMBINED_RESULTS_FILE"
        else
            echo "WARNING: Could not find BEST_RESULT line in evaluation output for $FULL_EXP_ID. Appending placeholder."
            echo -e "${FULL_EXP_ID}\t$BATCH_SIZE\tERROR\tERROR\tERROR\tERROR" >> "$COMBINED_RESULTS_FILE"
            echo "Evaluation Output was:"
            echo "$EVAL_OUTPUT"
        fi

        # --- Added: Clean up checkpoint files --- #
        # echo "--- Skipping checkpoint cleanup for $FULL_EXP_ID (cleanup disabled) ---"
        # Keep the best checkpoint info but don't delete others
        BEST_CKPT_PATH_FROM_EVAL=$(echo "$BEST_LINE" | cut -f 5)
        # if [ -n "$BEST_CKPT_PATH_FROM_EVAL" ] && [ -f "$BEST_CKPT_PATH_FROM_EVAL" ]; then
        #     echo "Best checkpoint identified: $BEST_CKPT_PATH_FROM_EVAL (all checkpoints preserved)"
        # else
        #     echo "Warning: Could not determine best checkpoint path from evaluation output or file not found."
        # fi
        
        # Original cleanup code commented out below
        if [ -n "$BEST_CKPT_PATH_FROM_EVAL" ] && [ -f "$BEST_CKPT_PATH_FROM_EVAL" ]; then
            echo "Keeping best checkpoint: $BEST_CKPT_PATH_FROM_EVAL"
            # Find all .pth files EXCEPT the best one and remove them
            find "$EXP_DIR" -name "*.pth" -print0 | grep -zvF "$(basename "$BEST_CKPT_PATH_FROM_EVAL")" | xargs -0 rm -f
            if [ $? -eq 0 ]; then
                echo "Successfully removed intermediate checkpoints."
            else
                echo "Warning: Failed to remove some intermediate checkpoints in $EXP_DIR."
            fi
        else
             echo "Warning: Could not determine best checkpoint path from evaluation output or file not found. Attempting to keep files matching 'best_model*.pth'."
             find "$EXP_DIR" -name "*.pth" -print0 | grep -zv 'best_model.*\.pth$' | xargs -0 rm -f
             if [ $? -eq 0 ]; then
                echo "Successfully removed intermediate checkpoints based on pattern 'best_model*.pth'."
             else
                echo "Warning: Failed to remove some or all intermediate checkpoints in $EXP_DIR using fallback pattern."
             fi
        fi

    fi
    # -------------------------------------------- #

  fi
  echo "" # Add spacing between experiments
}

# === Define and Run Experiments in Parallel ===

# Run experiments with different batch sizes and distribute them across available GPUs
# BATCH_SIZES=(16 32 64) # Removed batch size variation
FIXED_BATCH_SIZE=32
echo "Using fixed batch size: $FIXED_BATCH_SIZE"

# Define array of experiment configs
declare -a EXP_CONFIGS=(
  "EXP1 \"No Finding,Atelectasis,Cardiomegaly,Effusion\" Freeze false false"
  "EXP2 \"No Finding,Atelectasis\" Freeze false false"
  "EXP3 \"No Finding,Cardiomegaly\" Freeze false false"
  "EXP4 \"No Finding,Effusion\" Freeze false false"
  "EXP5 \"Atelectasis,Cardiomegaly,Effusion\" Freeze false true" # Note: This is 3-class, handle_abnormal=true was likely intended for EXP6? Check logic.
  "EXP6 \"No Finding,Atelectasis,Cardiomegaly,Effusion\" Freeze true false" # Note: filter_nf=false, handle_abnormal=false. Consider if this requires special handling in python train/eval.
)

MAX_CONCURRENT_JOBS=10 # Limit the number of parallel experiment runs

echo "Setting up distributed execution across $NUM_GPUS GPUs: ${CUDA_DEVICES[*]}"
echo "Total experiment configurations: ${#EXP_CONFIGS[@]}"
# echo "Batch sizes: ${BATCH_SIZES[*]}"
TOTAL_RUNS=${#EXP_CONFIGS[@]} # Only depends on number of configs now
echo "Total runs: $TOTAL_RUNS (at fixed batch size $FIXED_BATCH_SIZE)"
echo "Maximum concurrent jobs: $MAX_CONCURRENT_JOBS"

# Launch experiments directly in nested loops
job_idx=0
# Removed outer loop for BS in "${BATCH_SIZES[@]}"; do
for EXP_CONFIG in "${EXP_CONFIGS[@]}"; do
    # Read experiment config into an array, preserving quotes
    eval "CONFIG=($EXP_CONFIG)" # CONFIG = [EXP_ID, "Quoted Classes", FREEZE_MODE, HANDLE_ABNORMAL, FILTER_NF]

    # Assign GPU based on round-robin from our CUDA_DEVICES array
    gpu_idx_in_array=$((job_idx % NUM_GPUS))
    GPU_ID=${CUDA_DEVICES[gpu_idx_in_array]}

    # --- Job Control: Wait if max concurrent jobs reached --- #
    # Get current number of background jobs
    current_jobs=$(jobs -p | wc -l)
    # Check if we need to wait
    while [[ $current_jobs -ge $MAX_CONCURRENT_JOBS ]]; do
      echo "Reached max concurrent jobs ($MAX_CONCURRENT_JOBS). Waiting for a slot..."
      # Wait for any background job to finish (-n option)
      wait -n
      # Update job count after waiting
      current_jobs=$(jobs -p | wc -l)
    done
    # -------------------------------------------------------- #

    echo "-----------------------------------------"
    echo "Queueing Job $((job_idx + 1))/$TOTAL_RUNS: Exp=${CONFIG[0]}, BS=$FIXED_BATCH_SIZE, GPU=$GPU_ID (Current Jobs: $current_jobs)"
    echo "Config: ${CONFIG[*]}"
    echo "-----------------------------------------"

    # Launch run_exp in the background, passing arguments individually, using FIXED_BATCH_SIZE
    (run_exp "${CONFIG[0]}" "${CONFIG[1]}" "${CONFIG[2]}" "${CONFIG[3]}" "${CONFIG[4]}" "$FIXED_BATCH_SIZE" "$GPU_ID") &

    # Optional: Add a small delay to prevent resource contention during startup
    sleep 2

    job_idx=$((job_idx + 1))
done
# Removed closing done for the BS loop

# Wait for all background processes (jobs) to complete
wait

echo ""
echo "#########################################"
echo "##### All Specified Experiments Run #####"
echo "#########################################"
echo ""

exit 0 