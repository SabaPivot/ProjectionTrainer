#!/bin/bash

# This script runs a single experiment on a specified GPU
# Usage: ./run_single_experiment.sh <exp_id> <class_names> <freeze_mode> <handle_abnormal> <filter_nf> <batch_size> <device_id>
# Example: ./run_single_experiment.sh EXP1 "No Finding,Atelectasis,Cardiomegaly,Effusion" Freeze false false 32 0

# Validate arguments
if [ $# -ne 7 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <exp_id> <class_names> <freeze_mode> <handle_abnormal> <filter_nf> <batch_size> <device_id>"
    echo "Example: $0 EXP1 \"No Finding,Atelectasis,Cardiomegaly,Effusion\" Freeze false false 32 0"
    exit 1
fi

# Extract arguments
EXP_ID=$1
CLASS_NAMES=$2
FREEZE_MODE=$3
HANDLE_ABNORMAL=$4
FILTER_NF=$5
BATCH_SIZE=$6
DEVICE_ID=$7

# Basic configuration - Adjust paths if necessary
PYTHON_EXE="python"
MODULE_NAME="soombit.train"
OUTPUT_BASE_DIR="./soombit/checkpoints"
DATA_JSON="/mnt/samuel/Siglip/soombit/data/single_label_dataset.json"
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
EPOCHS=10

# Update experiment ID to include batch size and device ID
FULL_EXP_ID="${EXP_ID}_BS${BATCH_SIZE}_GPU${DEVICE_ID}"

echo "========================================="
echo "===== Starting Experiment: $FULL_EXP_ID ====="
echo "========================================="
echo "Classes        : $CLASS_NAMES"
echo "Freeze Mode    : $FREEZE_MODE"
echo "Handle Abnormal: $HANDLE_ABNORMAL"
echo "Filter NF      : $FILTER_NF"
echo "Batch Size     : $BATCH_SIZE"
echo "CUDA Device    : $DEVICE_ID"
echo "Output Dir     : ${OUTPUT_BASE_DIR}/${FULL_EXP_ID}"

# Construct python command
CMD="$PYTHON_EXE -m $MODULE_NAME \
  --exp_id $FULL_EXP_ID \
  --class_names \"$CLASS_NAMES\" \
  --freeze_mode $FREEZE_MODE \
  --output_base_dir \"$OUTPUT_BASE_DIR\" \
  --data_json \"$DATA_JSON\" \
  --image_root \"$IMAGE_ROOT\" \
  --image_root_2 \"$IMAGE_ROOT_2\" \
  --epochs $EPOCHS \
  --lr 1e-5 \
  --bb_lr 1e-5 \
  --batch_size $BATCH_SIZE \
  --device_id $DEVICE_ID \
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
eval $CMD

# Check exit status
if [ $? -ne 0 ]; then
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!!!!! Experiment $FULL_EXP_ID FAILED !!!!!!!!!"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
else
  echo ""
  echo "========================================="
  echo "===== Finished Experiment: $FULL_EXP_ID ====="
  echo "========================================="
  exit 0
fi 