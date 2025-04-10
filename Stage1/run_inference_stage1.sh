#! /bin/bash

python inference_stage1.py \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --llm_name "google/gemma-3-1b-it" \
    --projection_path "/home/compu/samuel/Siglip/trained_projection_stage1/VD_Class_10_lr5e-5_gemma3_vit-l-384/final_model" \
    --image_path "/home/compu/samuel/Siglip/images/001.png" \
    --max_new_tokens 128 \
    --temperature 0.6 \
    --top_p 0.9 \
    --repetition_penalty 1.4 \
