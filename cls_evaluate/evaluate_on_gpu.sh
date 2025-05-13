#!/bin/bash

# This script runs evaluation for an experiment on a specified GPU
# Usage: ./evaluate_on_gpu.sh <exp_dir> <test_json> <device_id>
# Example: ./evaluate_on_gpu.sh "./soombit/checkpoints/EXP1_BS32_GPU0" "/mnt/samuel/Siglip/balanced_samples/filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json" 1

# Validate arguments
if [ $# -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <exp_dir> <test_json> <device_id>"
    echo "Example: $0 \"./soombit/checkpoints/EXP1_BS32_GPU0\" \"/mnt/samuel/Siglip/balanced_samples/filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json\" 1"
    exit 1
fi

# Extract arguments
EXP_DIR=$1
TEST_JSON=$2
DEVICE_ID=$3

# Basic configuration
PYTHON_EXE="python"

echo "========================================="
echo "===== Evaluating: $(basename $EXP_DIR) ====="
echo "========================================="
echo "Experiment Dir : $EXP_DIR"
echo "Test JSON      : $TEST_JSON"
echo "CUDA Device    : $DEVICE_ID"
echo "-----------------------------------------"

# Construct python command
CMD="$PYTHON_EXE -m soombit.evaluate_experiment \
  --exp_dir \"$EXP_DIR\" \
  --test_json \"$TEST_JSON\" \
  --device_id $DEVICE_ID \
  "

# Execute the command
echo "Executing Command:"
echo "$CMD"
echo "-----------------------------------------"
eval $CMD

# Check exit status
if [ $? -ne 0 ]; then
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!!!!! Evaluation for $(basename $EXP_DIR) FAILED !!!!!!!!!"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
else
  echo ""
  echo "========================================="
  echo "===== Evaluation Finished: $(basename $EXP_DIR) ====="
  echo "========================================="
  exit 0
fi 