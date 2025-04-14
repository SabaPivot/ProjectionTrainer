#! /bin/bash

# Define paths
LLM_DIR="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr2e-5_gemma3_vit-l-384-DR-with disease classes-epoch4/final_model/language_model"
PROJECTOR_DIR="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr2e-5_gemma3_vit-l-384-DR-with disease classes-epoch4/final_model/projection_layer"
INPUT_JSON="/mnt/samuel/Siglip/ProjectionTrainer/VDvalidation_with_Atelectasis.json"
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg" # Base directory for images

python inference_vqa_stage2.py \
    --llm_path "${LLM_DIR}" \
    --projector_path "${PROJECTOR_DIR}" \
    --input_json "${INPUT_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --max_new_tokens 1024 \
    --temperature 0.3 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.8 \
    --length_penalty 1.2 \
    --num_beams 3 \
    --device "cuda:0"