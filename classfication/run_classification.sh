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

for candidate_labels in "${CANDIDATE_LABELS[@]}"; do
    python ../../filter_json.py \
    --candidate_labels "$candidate_labels"

    python classifier.py \
    --image_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --json_file "/mnt/samuel/Siglip/filtered_formatted_Class_QA.json" \
    --candidate_labels "$candidate_labels" \
    --prompt_template "photo" | tee "classification_results_${candidate_labels}.txt"
done