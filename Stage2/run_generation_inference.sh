#! /bin/bash

# Script to run Stage2 generation inference with different candidate label combinations
# Now with integrated balanced random sampling

export CUDA_VISIBLE_DEVICES=0

# Define the candidate label combinations to test
declare -a CANDIDATE_LABELS=(
    "Atelectasis, No Finding"
    "Atelectasis, Cardiomegaly, No Finding"
    "Atelectasis, Cardiomegaly, Effusion, No Finding"
    "Cardiomegaly, No Finding"
    "Effusion, No Finding"
    "Cardiomegaly, Effusion, No Finding"
    "Atelectasis, Effusion, No Finding"
)

# Path to the filtered dataset
OUTPUT_PATH="/mnt/samuel/Siglip/filtered_formatted_Class_QA.json"

# --- Paths to the trained models --- #
# !! IMPORTANT: Set BASE_LLM_NAME to the model used for training the adapters below !!
BASE_LLM_NAME="Qwen/Qwen3-8B" # Placeholder - CHANGE IF NEEDED
ADAPTER_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr5e-5_QWEN3-8B-QLoRA_vit-l-384-QA_BALANCED-MIMIC/checkpoint-epoch_3/language_model"
PROJECTOR_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage1/trained_projection_stage1/VD_lr3e-5_qwen3-8b-QLoRA-Load_l-384-10"

# Create output directories
RESULTS_DIR="generation_results_stage2"
BALANCED_SAMPLES_DIR="/mnt/samuel/Siglip/balanced_samples"
mkdir -p $RESULTS_DIR
mkdir -p $BALANCED_SAMPLES_DIR

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the root directory (Siglip)
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

for candidate_labels in "${CANDIDATE_LABELS[@]}"; do
    echo "Processing with candidate labels: $candidate_labels"
    
    # Create a filename-friendly version of the candidate labels for balanced samples
    filename_labels=$(echo "$candidate_labels" | sed 's/, /_/g' | sed 's/ /_/g')
    balanced_output_path="$BALANCED_SAMPLES_DIR/filtered_${filename_labels}.json"
    
    # Step 1: Run balanced random sampling
    echo "Creating balanced random sample for: $candidate_labels"
    python "$SCRIPT_DIR/balanced_random_sample.py" \
        --candidate_labels "$candidate_labels" \
        --output_path "$balanced_output_path" \
        --sample_size 100 \
        --seed 42
    
    echo "Saved balanced sample to: $balanced_output_path"
    
    # Step 2: Filter the dataset for the current candidate labels
    # (This is kept for backward compatibility but we'll use the balanced sample for inference)
    echo "Filtering dataset for labels: $candidate_labels"
    python "$ROOT_DIR/filter_json.py" \
    --candidate_labels "$candidate_labels" \
    --output_path "$OUTPUT_PATH"
    
    # Step 3: Run the generation inference using the balanced sample
    echo "Running generation inference for: $candidate_labels using balanced sample"
    python "$SCRIPT_DIR/inference_generation.py" \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --base_llm_name "$BASE_LLM_NAME" \
    --adapter_path "$ADAPTER_PATH" \
    --projector_path "$PROJECTOR_PATH" \
    --input_json "$balanced_output_path" \
    --image_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --candidate_labels "$candidate_labels" \
    --max_length 128 | tee "$RESULTS_DIR/generation_results_${candidate_labels}.txt"
    
    echo "Completed processing for: $candidate_labels"
    echo "----------------------------------------------"
done

echo "All balanced sampling and generation inference runs completed."
echo "Balanced samples saved in $BALANCED_SAMPLES_DIR directory."
echo "Results saved in $RESULTS_DIR directory." 