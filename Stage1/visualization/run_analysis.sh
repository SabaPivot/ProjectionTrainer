#! /bin/bash

python tsne_embedding_analysis.py \
    --json_path /home/compu/DATA/CXR_VDQA/Train/formatted_Class_QA.json \
    --images_root "/home/compu/DATA/NIH Chest X-rays_jpg" \
    --projector_ckpt "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_20_lr5e-5_gemma3_vit-l-384/final_model" \
    --output_dir . \
    --batch_size 128 \
    --model_name "/home/compu/samuel/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_XraySigLIP__vit-l-16-siglip-384__webli_lr1e-4_bs8_ep10_20250422_190634/epoch_10" \
    --n_jobs 4