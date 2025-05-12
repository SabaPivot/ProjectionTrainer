#! /bin/bash

# Define paths
BASE_LLM_NAME="Qwen/Qwen3-8B" # Specify the base model used for QLoRA
ADAPTER_DIR="/mnt/samuel/Siglip/ProjectionTrainer/Stage2/trained_vqa_stage2/VD_Class_20_lr5e-5_QWEN3-8B-QLoRA_vit-l-384-QA_BALANCED-MIMIC/checkpoint-epoch_3/language_model"
PROJECTOR_DIR="/mnt/samuel/Siglip/ProjectionTrainer/Stage1/trained_projection_stage1/VD_lr3e-5_qwen3-8b-QLoRA-Load_l-384-10"
INPUT_JSON="/mnt/samuel/Siglip/ProjectionTrainer/VDvalidation_with_Atelectasis.json"
IMAGE_ROOT="/mnt/data/CXR/NIH Chest X-rays_jpg" # Base directory for NIH CXR images
IMAGE_ROOT_2="/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files" # Base directory for MIMIC-CXR images

python inference_vqa_stage2.py \
    --base_llm_name "${BASE_LLM_NAME}" \
    --adapter_path "${ADAPTER_DIR}" \
    --projector_path "${PROJECTOR_DIR}" \
    --input_json "${INPUT_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --image_root_2 "${IMAGE_ROOT_2}" \
    --max_new_tokens 1024 \
    --temperature 0.3 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.8 \
    --length_penalty 1.2 \
    --num_beams 3 \
    --device "cuda:0"