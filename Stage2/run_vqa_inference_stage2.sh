#! /bin/bash

python inference_vqa_stage2.py \
    --llm_path /mnt/samuel/Siglip/Stage2/trained_vqa_stage2/VQA_Stage2_VD_Class_20_lr2e-5_gemma3_vit-l-384-SFT:lr2e-5-epoch5-DR/checkpoint-epoch_4/language_model \
    --projector_path /mnt/samuel/Siglip/Stage2/trained_vqa_stage2/VQA_Stage2_VD_Class_20_lr2e-5_gemma3_vit-l-384-SFT:lr2e-5-epoch5-DR/checkpoint-epoch_4/projection_layer \
    --image_path "/mnt/data/CXR/NIH Chest X-rays_jpg/images_001/images/00000004_000.jpg" \
    --question "Examine the provided chest X-ray and write a report addressing the overall health of the lungs and heart." \
    --max_new_tokens 512 \
    --temperature 0.3 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.8 \
    --length_penalty 1.2 \
    --num_beams 3 \
    --device "cuda:1"