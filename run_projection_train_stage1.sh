#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_NAME="VD_Class_10_lr1e-4"

accelerate launch --mixed_precision bf16 train_projection_stage1.py \
    --train_json /home/compu/DATA/CXR_VDQA/Train/formatted_VD_class.json \
    --image_root "/home/compu/DATA/NIH Chest X-rays_jpg" \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli" \
    --llm_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --output_dir ./trained_projection_stage1/$RUN_NAME \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.05 \
    --wandb_project "xray_patch_projection_training" \
    --wandb_run_name $RUN_NAME