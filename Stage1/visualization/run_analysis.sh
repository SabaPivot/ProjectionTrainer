#! /bin/bash

python tsne_embedding_analysis.py \
    --json_path /mnt/samuel/Siglip/soombit/data/transformed_test_888.json \
    --images_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --projector_ckpt "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_20_lr5e-5_gemma3_vit-l-384/final_model" \
    --output_dir . \
    --batch_size 256 \
    --model_name "/mnt/samuel/Siglip/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_siglip2-so400m-patch16-512_lr5e-5_bs16_ep300_20250512_134054/epoch_3" \
    --n_jobs 4