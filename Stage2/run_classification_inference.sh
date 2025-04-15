#! /bin/bash

# Script to run Stage2 classification inference with different candidate label combinations

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

# Path to the trained models
LLM_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr5e-5_gemma3_vit-l-384-QA_BALANCED/checkpoint-epoch_4/language_model"
PROJECTOR_PATH="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr5e-5_gemma3_vit-l-384-QA_BALANCED/checkpoint-epoch_4/projection_layer"

# Create output directory for results
RESULTS_DIR="classification_results_stage2"
mkdir -p $RESULTS_DIR

for candidate_labels in "${CANDIDATE_LABELS[@]}"; do
    echo "Processing with candidate labels: $candidate_labels"

    # Filter the dataset for the current candidate labels
    echo "Filtering dataset for labels: $candidate_labels"
    python filter_json.py \
    --candidate_labels "$candidate_labels" \
    --output_path "$OUTPUT_PATH"
    
    # Run the classification inference
    python ProjectionTrainer/Stage2/inference_classification.py \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --llm_path "$LLM_PATH" \
    --projector_path "$PROJECTOR_PATH" \
    --input_json "$OUTPUT_PATH" \
    --image_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --candidate_labels "$candidate_labels" | tee "$RESULTS_DIR/classification_results_${candidate_labels}.txt"
    
    echo "Completed processing for: $candidate_labels"
    echo "----------------------------------------------"
done

echo "All classification inference runs completed."
echo "Results saved in $RESULTS_DIR directory." 