#! /bin/bash

# Script to run classification on pre-generated balanced samples

declare -a CANDIDATE_LABELS=(
    "Atelectasis, No Finding"
    "Atelectasis, Cardiomegaly, No Finding"
    "Atelectasis, Cardiomegaly, Effusion, No Finding"
    "Cardiomegaly, No Finding"
    "Effusion, No Finding"
    "Cardiomegaly, Effusion, No Finding"
    "Atelectasis, Effusion, No Finding"
)

BALANCED_SAMPLES_DIR="/mnt/samuel/Siglip/balanced_samples"
RESULTS_DIR="."

mkdir -p "$RESULTS_DIR"

for candidate_labels in "${CANDIDATE_LABELS[@]}"; do
    echo "Running classification for: $candidate_labels on balanced sample"

    # Construct the balanced sample filename
    filename_labels=$(echo "$candidate_labels" | sed 's/, /_/g' | sed 's/ /_/g')
    balanced_json_path="$BALANCED_SAMPLES_DIR/filtered_${filename_labels}.json"
    
    # Check if the balanced file exists
    if [ ! -f "$balanced_json_path" ]; then
        echo "Error: Balanced sample file not found: $balanced_json_path" 
        echo "Skipping classification for: $candidate_labels"
        echo "----------------------------------------------"
        continue # Skip to the next iteration
    fi

    # Run the classifier using the balanced sample
    python classifier.py \
    --image_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --json_file "$balanced_json_path" \
    --candidate_labels "$candidate_labels" \
    --prompt_template "none" | tee "$RESULTS_DIR/balanced_classification_results_${filename_labels}.txt"
    
    echo "Completed processing for: $candidate_labels"
    echo "----------------------------------------------"
done

echo "All balanced classification runs completed." 