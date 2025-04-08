#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_NAME="VD_Class_10_lr5e-5_gemma3_vit-l-384"

accelerate launch --mixed_precision bf16 train_projection_stage1.py \
    --train_json /home/compu/DATA/CXR_VDQA/Train/formatted_VD_class.json \
    --image_root "/home/compu/DATA/NIH Chest X-rays_jpg" \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --llm_name "google/gemma-3-1b-it" \
    --output_dir ./trained_projection_stage1/$RUN_NAME \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.05 \
    --wandb_project "xray_patch_projection_training" \
    --wandb_run_name $RUN_NAME