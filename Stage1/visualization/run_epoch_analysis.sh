#!/bin/bash

# Base directory containing epoch folders
BASE_MODEL_DIR="/mnt/samuel/Siglip/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_siglip2-so400m-patch16-512_lr5e-5_bs16_ep300_20250512_134054"
# Check if the base directory exists
if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "Error: Base directory $BASE_MODEL_DIR not found."
    exit 1
fi

# Loop through each epoch directory
for epoch_dir in "$BASE_MODEL_DIR"/epoch_*; do
    if [ -d "$epoch_dir" ]; then
        # Extract epoch number from directory name (e.g., epoch_1 -> 1)
        epoch_num=$(basename "$epoch_dir" | sed 's/epoch_//')

        # Define the output filename
        output_filename="t-sne_epoch_${epoch_num}.png"
        echo "Processing $epoch_dir, saving to $output_filename"

        # Run the python script
        python tsne_embedding_analysis.py \
            --json_path /mnt/samuel/Siglip/soombit/data/transformed_test_888.json \
            --images_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
            --projector_ckpt "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_20_lr5e-5_gemma3_vit-l-384/final_model" \
            --output_dir . \
            --batch_size 256 \
            --model_name "$epoch_dir" \
            --n_jobs 4 \
            --output_filename "$output_filename" # Assuming the script accepts this argument

        echo "Finished processing epoch $epoch_num"
    fi
done

echo "All epochs processed."

# Make the script executable
chmod +x run_epoch_analysis.sh

echo "run_epoch_analysis.sh created and made executable." 