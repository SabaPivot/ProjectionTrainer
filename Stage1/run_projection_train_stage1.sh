#! /bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3
RUN_NAME="VD_lr3e-5_qwen3-8b-QLoRA-Load_l-384-10"

accelerate launch --mixed_precision bf16 train_projection_stage1.py \
    --train_json /mnt/samuel/Siglip/combined_train.json \
    --val_json /mnt/samuel/Siglip/projector_validation_QA_100.json \
    --image_root "/mnt/data/CXR/NIH Chest X-rays_jpg" \
    --image_root_2 "/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files" \
    --vision_model_name "/mnt/samuel/Siglip/soombit/checkpoint/epoch_16" \
    --llm_name "Qwen/Qwen3-8B" \
    --output_dir ./trained_projection_stage1/$RUN_NAME \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --enable_qlora \
    --wandb_project "xray_patch_projection_training" \
    --wandb_run_name $RUN_NAME \
    --save_every_n_epochs 2 