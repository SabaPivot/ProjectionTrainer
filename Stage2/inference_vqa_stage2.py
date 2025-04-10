# Stage2/inference_vqa_stage2.py

import os
import torch
import logging
import argparse
import json
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from transformers import Gemma3ForCausalLM
from safetensors.torch import load_file
import sys

# --- Add parent directory for imports ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Assuming projectors.py is in the parent directory (Siglip/)
try:
    from projectors import MLPProjector
except ImportError:
    print("Error: Could not import MLPProjector. Ensure projectors.py is in the parent directory (Siglip/).")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(args):
    """Loads models and runs inference on a single image and question."""

    # --- Determine Device and Dtype ---
    device = torch.device(args.device) # Use device from argument
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine dtype based on selected device capabilities
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif device.type == 'cuda':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    logger.info(f"Using device: {device}")
    logger.info(f"Using model dtype: {model_dtype}")

    # --- Load Vision Components ---
    logger.info(f"Loading vision processor from: {args.vision_model_name}")
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    logger.info(f"Loading vision encoder from: {args.vision_model_name}")
    vision_encoder = AutoModel.from_pretrained(
        args.vision_model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    ).to(device).eval()
    # Infer vision dimension
    vision_dim = vision_encoder.config.vision_config.hidden_size
    logger.info(f"Inferred vision dimension: {vision_dim}")

    # --- Load Language Model Components ---
    logger.info(f"Loading LLM tokenizer from: {args.llm_path}")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    logger.info(f"Loading LLM using Gemma3ForCausalLM from path: {args.llm_path}")

    # Check if config.json exists in the LLM path - STILL REQUIRED!
    llm_config_path = os.path.join(args.llm_path, "config.json")
    if not os.path.exists(llm_config_path):
        logger.error(f"Critical Error: 'config.json' not found in {args.llm_path}")
        logger.error("This is required by Gemma3ForCausalLM.from_pretrained to load the correct model architecture.")
        logger.error("The absence of config.json is the primary reason for loading failures.")
        logger.error("Please ensure the config.json from the original base LLM ('google/gemma-3-1b-it') is saved alongside the fine-tuned weights in this directory.")
        sys.exit(1)

    # Attempt to load directly using the path and the specific Gemma3 class
    try:
        llm_model = Gemma3ForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager" # Force eager attention
        ).to(device).eval()
    except RuntimeError as e:
        logger.error(f"RuntimeError loading LLM using Gemma3ForCausalLM from {args.llm_path}: {e}")
        logger.error("This likely means the weights in model.safetensors (or .bin) do not match the architecture defined in config.json.")
        logger.error("Verify that the config.json belongs to the model whose weights are saved here AND that the weights are compatible with Gemma3ForCausalLM.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load LLM using Gemma3ForCausalLM from {args.llm_path}: {e}")
        sys.exit(1)

    # Infer LLM dimension (should be correct if loading succeeded)
    llm_dim = llm_model.config.hidden_size
    logger.info(f"Inferred LLM dimension: {llm_dim}")

    # Handle padding token for tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")

    # --- Load Projector ---
    logger.info(f"Loading projector from: {args.projector_path}")
    # Instantiate using inferred dimensions
    # Assuming default expansion_factor=10 from projectors.py definition
    projection_layer = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    projector_weights_path = os.path.join(args.projector_path, "model.safetensors") # Based on list_dir result
    if not os.path.exists(projector_weights_path):
        raise FileNotFoundError(f"Projector weights not found at {projector_weights_path}")

    try:
        state_dict = load_file(projector_weights_path, device="cpu") # Load weights to CPU first
        projection_layer.load_state_dict(state_dict)
        logger.info("Successfully loaded projector weights.")
    except Exception as e:
        logger.error(f"Error loading projector state_dict: {e}", exc_info=True)
        raise

    projection_layer = projection_layer.to(device, dtype=model_dtype).eval()

    # --- Load and Preprocess Image ---
    logger.info(f"Loading image from: {args.image_path}")
    try:
        image = Image.open(args.image_path).convert('RGB')
        # Determine image size (use processor's config or default from training)
        img_size = processor.image_processor.size['height'] if 'height' in processor.image_processor.size else 384
        image = image.resize((img_size, img_size))
    except FileNotFoundError:
        logger.error(f"Image file not found: {args.image_path}")
        return
    except Exception as e:
        logger.error(f"Error opening or processing image: {e}")
        return

    # Process image
    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs.pixel_values.to(device, dtype=model_dtype)
    logger.info(f"Image processed, pixel values shape: {pixel_values.shape}")

    # --- Tokenize Question ---
    logger.info(f"Tokenizing question: \"{args.question}\"")
    # Important: Add a space or indicator if needed, depending on how the model was trained
    # Assuming the training format implies direct concatenation.
    # Often, a special token or prompt structure is used before the question. Check training setup if needed.
    # Let's add a common pattern: "<human>: {question} <assistant>:"
    # Alternatively, use the exact format from training if known. For now, simple question tokenization.
    # We won't add BOS/EOS here as the LLM's generate method usually handles the start.
    question_tokens = llm_tokenizer(
        args.question,
        return_tensors="pt",
        add_special_tokens=False # Usually False, let embedding concatenation handle context
    ).input_ids.to(device)
    logger.info(f"Question tokenized, input_ids shape: {question_tokens.shape}")

    # --- Generate Embeddings ---
    with torch.no_grad(): # Ensure no gradients are calculated
        # 1. Get Visual Features
        vision_outputs = vision_encoder.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False, # Don't need hidden states
            return_dict=True
        )
        # Use patch embeddings (discard class token)
        patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :] # Shape: [B, NumPatches, Dim_Vision]
        logger.info(f"Vision features extracted, shape: {patch_embeddings.shape}")

        # 2. Project Visual Features
        projected_embeds = projection_layer(patch_embeddings) # Shape: [B, NumPatches, Dim_LLM]
        logger.info(f"Projected visual features shape: {projected_embeds.shape}")

        # 3. Embed Question Tokens
        input_embed_layer = llm_model.get_input_embeddings()
        question_embeds = input_embed_layer(question_tokens) # Shape: [B, SeqLen_Q, Dim_LLM]
        logger.info(f"Question embeddings shape: {question_embeds.shape}")

        # 4. Concatenate Embeddings
        # This creates the input sequence for the LLM: [VisualTokens, QuestionTokens]
        inputs_embeds = torch.cat([projected_embeds, question_embeds], dim=1) # Shape: [B, NumPatches + SeqLen_Q, Dim_LLM]
        logger.info(f"Concatenated input embeddings shape: {inputs_embeds.shape}")

        # 5. Generate Answer with LLM
        # Create an attention mask (all ones, since we are providing embeddings)
        # The model needs to know the length of the input sequence.
        batch_size, sequence_length, _ = inputs_embeds.shape
        attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
        logger.info(f"Attention mask shape: {attention_mask.shape}")


        logger.info("Generating answer...")
        # Use generate method with inputs_embeds
        # max_new_tokens needs to be set appropriately.
        # Need eos_token_id for stopping criteria.
        outputs = llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # Provide the mask corresponding to inputs_embeds
            max_new_tokens=args.max_new_tokens, # Control output length
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.pad_token_id,
            # Use values from args
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty, 
            num_beams=args.num_beams,
        )
        logger.info("Generation complete.")

        # --- Decode and Print ---
        # outputs contain the full sequence (input + generated)
        # We need to decode only the generated part
        input_token_len = question_tokens.shape[1] # Length of original question tokens
        # Note: generate() output includes prompt tokens when using inputs_embeds IF they are part of the model's standard input processing.
        # However, when ONLY inputs_embeds are provided, the output usually starts right after. Let's assume the output only contains the *newly* generated tokens beyond the input.
        # A safer way is often to decode the whole sequence and slice off the prompt if needed, but generate with inputs_embeds sometimes behaves differently.
        # Let's decode the whole output first and see.

        full_decoded_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If using inputs_embeds, the `outputs` tensor often *only* contains the generated token IDs.
        # If it contains the input tokens as well, we need to slice.
        # Let's try decoding directly first. If it includes the prompt, we adjust.
        generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # A common pattern is that generate output WITH inputs_embeds may still include the prompt.
        # Let's try slicing based on the combined visual+question embedding length conceptually.
        # This is tricky. The most robust way is usually passing `input_ids` not `inputs_embeds`.
        # Let's stick to the simpler decoding for now.

        print("\\n" + "="*30)
        print("      VQA Inference Result")
        print("="*30)
        print(f"Image Path: {args.image_path}")
        print(f"Question:   {args.question}")
        print(f"Answer:     {generated_text}")
        print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 VQA Inference")

    # --- Model Paths ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Vision encoder name/path")
    parser.add_argument("--llm_path", type=str, default="/mnt/samuel/Siglip/Stage2/trained_vqa_stage2/VQA_Stage2_VD_Class_20_lr2e-5_gemma3_vit-l-384-SFT:lr2e-5-epoch5/final_model/language_model/", help="Path to the fine-tuned LLM directory")
    parser.add_argument("--projector_path", type=str, default="/mnt/samuel/Siglip/Stage2/trained_vqa_stage2/VQA_Stage2_VD_Class_20_lr2e-5_gemma3_vit-l-384-SFT:lr2e-5-epoch5/final_model/projection_layer/", help="Path to the trained projector directory (containing config and weights)")

    # --- Input Data ---
    parser.add_argument("--image_path", type=str, default="/mnt/data/CXR/NIH Chest X-rays_jpg/images_001/images/00000001_000.jpg", help="Path to the input image")
    parser.add_argument("--question", type=str, default="Examine the chest X-ray and write a report discussing any abnormalities or notable features.", help="Question to ask about the image")

    # --- Device --- #
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help=f"Device to use (e.g., cuda:0, cuda:1, cpu). Default: {default_device}")

    # --- Generation Parameters --- (Define arguments)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate for the answer")
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling (otherwise greedy decoding)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (lower is less random)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.8, help="Penalty for repeating tokens (1.0 means no penalty)")
    parser.add_argument("--length_penalty", type=float, default=1.2, help="Penalty applied to sequence length (used with num_beams>1)")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search (1 means no beam search)")

    args = parser.parse_args()

    # --- Validate Paths ---
    if not os.path.isdir(args.llm_path):
        logger.error(f"LLM path not found: {args.llm_path}")
        sys.exit(1)
    if not os.path.isdir(args.projector_path):
        logger.error(f"Projector path not found: {args.projector_path}")
        sys.exit(1)
    if not os.path.exists(args.image_path):
        logger.error(f"Image path not found: {args.image_path}")
        sys.exit(1)


    run_inference(args)