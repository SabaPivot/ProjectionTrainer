import os
import torch
import torch.nn as nn
from PIL import Image
import logging
import json
import argparse
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from projectors import MLPProjector 

# --- Configuration --- (Use argparse for flexibility)
def parse_args():
    parser = argparse.ArgumentParser(description="Inference using Stage 1 aligned projector.")
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli", help="Vision encoder model name.")
    parser.add_argument("--llm_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="LLM name.")
    parser.add_argument("--projection_path", type=str, required=True, help="Path to the trained Stage 1 projector directory (containing model.safetensors and projector_config.json).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0'). Defaults to cuda if available, else cpu.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for LLM generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Sampling top-p.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling.")
    return parser.parse_args()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    args = parse_args()

    # --- Device Setup ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Define Model Dtype ---
    # Use bfloat16 if available, else float16 if cuda, else float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    logger.info(f"Using model dtype: {model_dtype}")

    # --- Load Models and Processors --- #
    logger.info("Loading base models and processors...")
    try:
        processor = AutoProcessor.from_pretrained(args.vision_model_name)
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        vision_encoder = AutoModel.from_pretrained(
            args.vision_model_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        ).to(device).eval()
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        ).to(device).eval()
        logger.info(f"Loaded Vision Encoder: {args.vision_model_name}")
        logger.info(f"Loaded LLM: {args.llm_name}")
    except Exception as e:
        logger.error(f"Failed to load base models: {e}", exc_info=True)
        exit(1)

    # Handle padding token
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model.config.pad_token_id = llm_model.config.eos_token_id
        logger.info("Set tokenizer pad_token to eos_token")

    # --- Load Trained Projector --- #
    logger.info(f"Loading trained projection layer from {args.projection_path}...")
    try:
        # Load config
        proj_config_path = os.path.join(args.projection_path, "projector_config.json")
        if not os.path.exists(proj_config_path):
            raise FileNotFoundError(f"Projector config (projector_config.json) not found in {args.projection_path}")
        with open(proj_config_path, 'r') as f:
            proj_config = json.load(f)
        logger.info(f"Loaded projector config from {proj_config_path}")

        # Initialize projector from config
        projection_layer = MLPProjector(
            vision_dim=proj_config["vision_dim"],
            llm_dim=proj_config["llm_dim"]
        )

        # Load weights
        # Assumes weights are saved as model.safetensors by accelerator.save_model
        projection_weights_path = os.path.join(args.projection_path, "model.safetensors")
        if not os.path.exists(projection_weights_path):
             # Fallback for older save method? (less likely with current train script)
             projection_weights_path = os.path.join(args.projection_path, "pytorch_model.bin")
             if not os.path.exists(projection_weights_path):
                  raise FileNotFoundError(f"Projector weights (model.safetensors or pytorch_model.bin) not found in {args.projection_path}")
             state_dict = torch.load(projection_weights_path, map_location="cpu")
             logger.info(f"Loaded projector weights from {projection_weights_path} (pytorch_model.bin)")
        else:
             state_dict = load_file(projection_weights_path, device="cpu") # Load to CPU first
             logger.info(f"Loaded projector weights from {projection_weights_path} (model.safetensors)")

        # Adjust keys if needed (e.g., if saved with DDP wrapper 'module.' prefix)
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from projector state_dict keys.")

        # --- FIX: Add 'model.' prefix if it's missing --- 
        # This handles checkpoints saved directly from the nn.Sequential using accelerator.save_model
        if not all(key.startswith('model.') for key in state_dict.keys()):
            # Check if *any* key lacks the prefix, implying the whole dict needs adjustment
            if any(not key.startswith('model.') for key in state_dict.keys()): 
                logger.info("Adding missing 'model.' prefix to projector state_dict keys to match MLPProjector structure.")
                state_dict = {"model." + k: v for k, v in state_dict.items()}
        
        projection_layer.load_state_dict(state_dict)
        logger.info("Loaded projector weights into model.")

        # Move projector to device, cast dtype, set to eval mode
        projection_layer = projection_layer.to(device).to(dtype=model_dtype).eval()
        logger.info(f"Projector moved to {device}, cast to {model_dtype}, set to eval mode.")
        logger.info(f"Projector parameter dtype: {next(projection_layer.parameters()).dtype}")

    except FileNotFoundError as e:
        logger.error(f"Error loading projection model: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred loading the projection model: {e}", exc_info=True)
        exit(1)

    # --- Prepare Inputs --- #
    logger.info(f"Loading and processing image: {args.image_path}")
    try:
        image = Image.open(args.image_path).convert("RGB")
        # Process image for vision encoder (handles resizing based on processor config)
        image_inputs = processor(images=image, return_tensors="pt").to(device)
    except FileNotFoundError:
        logger.error(f"Image file not found: {args.image_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Error preparing inputs: {e}", exc_info=True)
        exit(1)

    # --- Inference --- #
    logger.info("Running inference using Stage 1 projector...")
    generated_text = "[Inference Error]"
    with torch.no_grad():
        try:
            # 1. Get patch embeddings from vision encoder
            vision_outputs = vision_encoder.vision_model(
                pixel_values=image_inputs.pixel_values.to(model_dtype),
                return_dict=True
            )
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :] # Discard CLS
            logger.info(f"Got patch embeddings shape: {patch_embeddings.shape}, dtype: {patch_embeddings.dtype}")

            # 2. Project visual features using the trained MLP projector
            projected_vision_embeds = projection_layer(patch_embeddings)
            logger.info(f"Projected visual embeddings shape: {projected_vision_embeds.shape}, dtype: {projected_vision_embeds.dtype}")

            # 3. Prepare for LLM generation - ONLY visual embeddings
            # This tests the direct image-to-text capability learned in Stage 1
            inputs_embeds = projected_vision_embeds
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            logger.info(f"Input embeddings for generation shape: {inputs_embeds.shape}")
            logger.info(f"Attention mask for generation shape: {attention_mask.shape}")

            # 4. Generate text using LLM
            logger.info("Generating text based on visual embeddings...")
            outputs = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            logger.info("Text generation finished.")

            # 5. Decode generated tokens
            # When using only inputs_embeds, generate returns only the new tokens.
            generated_ids = outputs[0] # Use the full output directly

            generated_text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error during model inference/generation: {e}", exc_info=True)

    # --- Print Output --- #
    print("\n--- Generated Text (Stage 1 Alignment Test) ---")
    print(f"Image: {args.image_path}")
    print(f"Generated Description: {generated_text}")
    print("--- End ---")

if __name__ == "__main__":
    main() 