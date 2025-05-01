#! /bin/bash

python tsne_embedding_analysis.py \
    --json_path /home/compu/samuel/ProjectionTrainer/Stage1/data/single_label_image_caption.json \
    --images_root "/home/compu/DATA/NIH Chest X-rays_jpg" \
    --projector_ckpt "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_20_lr5e-5_gemma3_vit-l-384/final_model" \
    --output_dir . \
    --batch_size 256 \
    --model_name "/home/compu/samuel/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_XraySigLIP__vit-b-16-siglip-512__webli_lr3e-5_bs8_ep30_20250501_095423/epoch_19" \
    --n_jobs 4