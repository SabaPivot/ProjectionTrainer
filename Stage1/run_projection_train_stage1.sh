#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_NAME="image-caption-truncated_lr7e-5_phi-4-mini-it-l-384-20"

accelerate launch --mixed_precision bf16 train_projection_stage1.py \
    --train_json /home/compu/samuel/ProjectionTrainer/Stage1/data/single_label_image_caption_truncated.json \
    --image_root "/home/compu/DATA/NIH Chest X-rays_jpg" \
    --vision_model_name "/home/compu/samuel/ProjectionTrainer/Stage0/trained_vision_encoder_stage0/SigLIP_FineTune_XraySigLIP__vit-l-16-siglip-384__webli_lr3e-5_bs8_ep30_20250425_113227/epoch_16" \
    --llm_name "microsoft/Phi-4-mini-instruct" \
    --output_dir ./trained_projection_stage1/$RUN_NAME \
    --batch_size 1 \
    --num_epochs 20 \
    --learning_rate 7e-5 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --train_val_split 0.01 \
    --wandb_project "xray_patch_projection_training" \
    --wandb_run_name $RUN_NAME \
    --save_every_n_epochs 5