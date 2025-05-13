#!/bin/bash

# Basic configuration - Adjust paths if necessary
PYTHON_EXE="python" # Added Python executable variable
MODULE_NAME="soombit.train" # Module path to run
EVAL_SCRIPT="soombit/evaluate_experiment.py" # Renamed script
OUTPUT_BASE_DIR="./soombit/checkpoints"
COMBINED_RESULTS_FILE="./soombit/all_experiments_summary.tsv" # Central results file (TSV)
DATA_JSON="/mnt/samuel/Siglip/soombit/data/single_label_dataset.json"
TEST_JSON_DIR="/mnt/samuel/Siglip/balanced_samples" # Base directory for test sets
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
EPOCHS=10 # Updated epoch count

# Initialize Combined Results File
echo "Initializing combined results file: $COMBINED_RESULTS_FILE"
echo -e "ExpID\tBestEpoch\tBestAcc\tBestAUC\tBestCheckpoint" > "$COMBINED_RESULTS_FILE"

# Function to run a single experiment
run_exp() {
  EXP_ID=$1
  CLASSES=$2
  FREEZE_MODE=$3
  HANDLE_ABNORMAL=$4
  FILTER_NF=$5

  echo ""
  echo "========================================="
  echo "===== Starting Experiment: $EXP_ID ====="
  echo "========================================="
  echo "Classes        : $CLASSES"
  echo "Freeze Mode    : $FREEZE_MODE"
  echo "Handle Abnormal: $HANDLE_ABNORMAL"
  echo "Filter NF      : $FILTER_NF"
  echo "Output Dir     : ${OUTPUT_BASE_DIR}/${EXP_ID}"

  # --- Determine correct Test JSON --- #
  local test_json_filename
  # Default to 4-class, will be overridden by specific cases
  test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json"

  if [ "$FILTER_NF" = "true" ]; then
      # EXP06, EXP18, EXP30 (A, C, E) - Use 4-class test set as no specific 3-class exists
      test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json"
      echo "INFO: Filter NF active, using 4-class test set for evaluation."
  elif [ "$HANDLE_ABNORMAL" = "true" ]; then
      # EXP05, EXP17, EXP29 (NF, Abnormal) - Use 4-class test set for evaluation
      test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json"
      echo "INFO: Handle Abnormal active, using 4-class test set for evaluation."
  else
      # Specific class combinations based on EXP_ID pattern
      case "$EXP_ID" in
          EXP01|EXP13|EXP25) # 4-class (NF, A, C, E)
              test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json" ;;
          EXP02|EXP14|EXP26) # NF vs A
              test_json_filename="filtered_Atelectasis_No_Finding.json" ;;
          EXP03|EXP15|EXP27) # NF vs C
              test_json_filename="filtered_Cardiomegaly_No_Finding.json" ;;
          EXP04|EXP16|EXP28) # NF vs E
              test_json_filename="filtered_Effusion_No_Finding.json" ;;
          EXP05|EXP17|EXP29) # Abnormal vs NF (Assuming handled_abnormal is true)
              test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json" ;;
          EXP06|EXP18|EXP30) # A vs C vs E (Assuming filter_nf is true)
              test_json_filename="filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json" ;;
          # Add other specific cases if needed
      esac
  fi
  local test_json_path="${TEST_JSON_DIR}/${test_json_filename}"
  echo "Test JSON      : $test_json_path" # Log the selected test file
  # ----------------------------------- #

  echo "-----------------------------------------"

  # Construct python command using -m and ensuring paths are quoted
  CMD="$PYTHON_EXE -m $MODULE_NAME \
    --exp_id $EXP_ID \
    --class_names \"$CLASSES\" \
    --freeze_mode $FREEZE_MODE \
    --output_base_dir \"$OUTPUT_BASE_DIR\" \
    --data_json \"$DATA_JSON\" \
    --image_root \"$IMAGE_ROOT\" \
    --image_root_2 \"$IMAGE_ROOT_2\" \
    --epochs $EPOCHS \
    --lr 1e-5 \
    --bb_lr 1e-5 \
    "

  # Add boolean flags if true
  if [ "$HANDLE_ABNORMAL" = "true" ]; then
    CMD="$CMD --handle_abnormal"
  fi

  if [ "$FILTER_NF" = "true" ]; then
    CMD="$CMD --filter_no_finding"
  fi

  # Execute the command
  echo "Executing Command:"
  echo "$CMD"
  echo "-----------------------------------------"
  eval $CMD | cat # Pipe through cat to potentially clean up progress bar output

  # Check exit status
  if [ $? -ne 0 ]; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!!!! Experiment $EXP_ID FAILED !!!!!!!!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # Optionally exit script on failure: exit 1
  else
    echo ""
    echo "========================================="
    echo "===== Finished Experiment: $EXP_ID ====="
    echo "========================================="

    # --- Run Evaluation Script and Capture Output --- #
    EXP_DIR="${OUTPUT_BASE_DIR}/${EXP_ID}"
    echo ""
    echo "--- Running Evaluation for $EXP_ID ---"
    echo "Evaluation Dir : $EXP_DIR"
    echo "Test JSON used : $test_json_path" # Log again for clarity
    echo "-----------------------------------------"
    # Execute evaluation as a module to handle relative imports correctly
    EVAL_OUTPUT=$($PYTHON_EXE -m soombit.evaluate_experiment --exp_dir "$EXP_DIR" --test_json "$test_json_path" 2>&1) # Capture stdout and stderr

    # Check evaluation exit status
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!!!! Evaluation for $EXP_ID FAILED !!!!!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Evaluation Output:"
        echo "$EVAL_OUTPUT"
    else
        echo "--- Finished Evaluation for $EXP_ID ---"
        # Extract best result line and append to combined file
        BEST_LINE=$(echo "$EVAL_OUTPUT" | grep "^BEST_RESULT")
        if [ -n "$BEST_LINE" ]; then
            # Remove the "BEST_RESULT\t" prefix (cut fields 2 onwards using tab delimiter)
            RESULT_DATA=$(echo "$BEST_LINE" | cut -f 2-)
            echo "Appending result to $COMBINED_RESULTS_FILE: $RESULT_DATA"
            echo -e "$RESULT_DATA" >> "$COMBINED_RESULTS_FILE"
        else
            echo "WARNING: Could not find BEST_RESULT line in evaluation output for $EXP_ID. Appending placeholder."
            echo -e "${EXP_ID}\tERROR\tERROR\tERROR\tERROR" >> "$COMBINED_RESULTS_FILE"
            echo "Evaluation Output was:"
            echo "$EVAL_OUTPUT"
        fi

        # --- Added: Clean up checkpoint files ---
        echo "--- Cleaning up checkpoints for $EXP_ID ---"
        find "$EXP_DIR" -name "*.pth" -print0 | grep -zv 'best_model.pth' | xargs -0 rm -f
        if [ $? -eq 0 ]; then
            echo "Successfully removed intermediate checkpoints."
        else
            echo "Warning: Failed to remove some or all intermediate checkpoints in $EXP_DIR."
        fi
        # --- End Cleanup ---

    fi
    # -------------------------------------------- #

  fi
  echo "" # Add spacing between experiments
}

# === Define and Run Experiments ===

# --- All Experiments use Freeze Mode --- #

# EXP1: 4-Class (NF, A, C, E), Freeze
run_exp "EXP1" "No Finding,Atelectasis,Cardiomegaly,Effusion" "Freeze" false false

# EXP2: NF vs A, Freeze
run_exp "EXP2" "No Finding,Atelectasis" "Freeze" false false

# EXP3: NF vs C, Freeze
run_exp "EXP3" "No Finding,Cardiomegaly" "Freeze" false false

# EXP4: NF vs E, Freeze
run_exp "EXP4" "No Finding,Effusion" "Freeze" false false

# EXP5: 3-Class (A, C, E - Filter NF), Freeze
# The class list should contain the target classes *after* filtering
run_exp "EXP5" "Atelectasis,Cardiomegaly,Effusion" "Freeze" false true

# EXP6: NF vs Abnormal (Map A,C,E to Abnormal), Freeze
# The class list should contain the source classes *before* mapping
run_exp "EXP6" "No Finding,Atelectasis,Cardiomegaly,Effusion" "Freeze" true false

echo ""
echo "#########################################"
echo "##### All Specified Experiments Run #####"
echo "#########################################"
echo ""

exit 0 