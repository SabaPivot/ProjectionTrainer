#! /bin/bash

python inference_stage1.py \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli" \
    --llm_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --projection_path "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_10_lr1e-4/final_model" \
    --image_path "/home/compu/samuel/Siglip/images/001.png" \
    --max_new_tokens 512 \
    --temperature 0.6 \
    --top_p 0.9 \
    --repetition_penalty 1.4 \
