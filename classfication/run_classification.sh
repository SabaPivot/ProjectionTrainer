#! /bin/bash

declare -a CANDIDATE_LABELS=(
    "Atelectasis, No Finding"
    "Atelectasis, Cardiomegaly, No Finding"
    "Atelectasis, Cardiomegaly, Effusion, No Finding"
    "Cardiomegaly, No Finding"
    "Effusion, No Finding"
    "Cardiomegaly, Effusion, No Finding"
    "Atelectasis, Effusion, No Finding"
)

# Define the input JSON file (adjust if needed)
INPUT_PATH="./filtered_formatted_Class_QA.json"
# Define the output path for the temporarily filtered file within the current directory
OUTPUT_PATH="./temp_filtered_data.json"
# Define the correct root path for the images
IMAGE_ROOT="/home/compu/DATA/NIH Chest X-rays_jpg"

for candidate_labels in "${CANDIDATE_LABELS[@]}"; do
    echo "Processing labels: $candidate_labels"
    # Filter the dataset using the correct input path
    python ./filter_json.py \
    --input_path "$INPUT_PATH" \
    --candidate_labels "$candidate_labels" \
    --output_path "$OUTPUT_PATH"

    # Check if filtering produced output before running classifier
    if [ -s "$OUTPUT_PATH" ]; then
        # Run the classifier using the correct image root and the temporary filtered file
        python classifier.py \
        --image_root "$IMAGE_ROOT" \
        --json_file "$OUTPUT_PATH" \
        --candidate_labels "$candidate_labels" \
        --prompt_template "photo" | tee "classification_results_${candidate_labels}.txt"

        echo "Completed processing for: $candidate_labels"
    else
        echo "Skipping classifier for [$candidate_labels] as filtering produced an empty file (or failed)."
    fi
    echo "----------------------------------------------"
done

# Clean up temporary file
rm -f "$OUTPUT_PATH"
echo "Cleaned up temporary file: $OUTPUT_PATH"

echo "Classification script finished."