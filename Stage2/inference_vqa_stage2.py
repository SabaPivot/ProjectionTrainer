# Stage2/inference_vqa_stage2.py

import os
import torch
import logging
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import Gemma3ForCausalLM
from peft import PeftModel
from safetensors.torch import load_file
import sys
import json
# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Stage1.projectors import MLPProjector

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(args):
    """Loads models and runs inference on a single image and question."""

    # --- Determine Device and Dtype ---
    device = torch.device(args.device) # Use device from argument
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
    logger.info(f"Loading LLM tokenizer from: {args.base_llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_llm_name, trust_remote_code=True)

    logger.info(f"Loading LLM using Gemma3ForCausalLM from path: {args.base_llm_name}")

    # Check if config.json exists in the LLM path - STILL REQUIRED!
    llm_config_path = os.path.join(args.base_llm_name, "config.json")
    if not os.path.exists(llm_config_path):
        logger.error(f"Critical Error: 'config.json' not found in {args.base_llm_name}")
        logger.error("This is required by Gemma3ForCausalLM.from_pretrained to load the correct model architecture.")
        logger.error("The absence of config.json is the primary reason for loading failures.")
        logger.error("Please ensure the config.json from the original base LLM ('google/gemma-3-1b-it') is saved alongside the fine-tuned weights in this directory.")
        sys.exit(1)

    # Attempt to load directly using the path and the specific Gemma3 class
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            args.base_llm_name,
            torch_dtype=model_dtype,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
    except RuntimeError as e:
        logger.error(f"RuntimeError loading LLM using Gemma3ForCausalLM from {args.base_llm_name}: {e}")
        logger.error("This likely means the weights in model.safetensors (or .bin) do not match the architecture defined in config.json.")
        logger.error("Verify that the config.json belongs to the model whose weights are saved here AND that the weights are compatible with Gemma3ForCausalLM.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load LLM using Gemma3ForCausalLM from {args.base_llm_name}: {e}")
        sys.exit(1)

    # Infer LLM dimension (should be correct if loading succeeded)
    llm_dim = llm.config.hidden_size
    logger.info(f"Inferred LLM dimension: {llm_dim}")

    # Handle padding token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")
        llm.config.pad_token_id = tokenizer.pad_token_id

    # --- Load Projector ---
    logger.info(f"Loading projector from: {args.projector_path}")
    # Instantiate using inferred dimensions
    # Assuming default expansion_factor=10 from projectors.py definition
    projection_layer = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    # Corrected projector weights file name based on list_dir result
    projector_weights_path = os.path.join(args.projector_path, "projector_best.bin") 
    if not os.path.exists(projector_weights_path):
         raise FileNotFoundError(f"Projector weights not found at {projector_weights_path}")
    # Use torch.load for .bin files, map to CPU to avoid GPU memory issues during loading
    proj_state_dict = torch.load(projector_weights_path, map_location="cpu") 
    projection_layer.load_state_dict(proj_state_dict)
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
    question_tokens = tokenizer(
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
        input_embed_layer = llm.get_input_embeddings()
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
        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # Provide the mask corresponding to inputs_embeds
            max_new_tokens=args.max_new_tokens, # Control output length
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        print("\\n" + "="*30)
        print("      VQA Inference Result")
        print("="*30)
        print(f"Image Path: {args.image_path}")
        print(f"Question:   {args.question}")
        print(f"Answer:     {generated_text}")
        print("="*30)

def process_sample(sample, image_root, processor, vision_encoder, projection_layer, tokenizer, llm, device, model_dtype, generation_args, image_root_2=None):
    """Process a single sample for visual question answering."""
    try:
        image_filename = sample.get("image")
        question_text = sample.get("problem")
        
        if not image_filename or not question_text:
            logger.warning(f"Skipping sample due to missing 'image' or 'problem': {sample}")
            return None
        
        # Determine which image root to use based on the image path format
        if image_filename.startswith("p") and "/" in image_filename and image_root_2:
            # Second format: "p10012261/s50349409"
            image_path = os.path.join(image_root_2, image_filename)
            
            # Handle MIMIC-CXR directory structure - the path might be a directory containing .jpg files
            if os.path.isdir(image_path):
                # Find all jpg files in the directory
                jpg_files = [f for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
                
                if not jpg_files:
                    logger.warning(f"No .jpg files found in directory: {image_path}. Skipping sample.")
                    return None
                
                # Use the first jpg file (there should typically be only one)
                image_path = os.path.join(image_path, jpg_files[0])
        else:
            # Original format: "images_002/images/00001836_057.jpg"
            image_path = os.path.join(image_root, image_filename)
        
        # --- Load and Preprocess Image ---
        image = Image.open(image_path).convert('RGB')
        img_size = processor.image_processor.size['height'] if 'height' in processor.image_processor.size else 384
        image = image.resize((img_size, img_size))
        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs.pixel_values.to(device, dtype=model_dtype)

        # --- Tokenize Question ---
        question_tokens = tokenizer(
            question_text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)

        # --- Generate Embeddings ---
        with torch.no_grad():
            vision_outputs = vision_encoder.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            )
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
            projected_embeds = projection_layer(patch_embeddings)
            input_embed_layer = llm.get_input_embeddings()
            question_embeds = input_embed_layer(question_tokens)
            inputs_embeds = torch.cat([projected_embeds, question_embeds], dim=1)
            batch_size, sequence_length, _ = inputs_embeds.shape
            attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)

            # --- Generate Answer ---
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_args # Pass generation kwargs dict
            )

            # --- Decode ---
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}. Skipping sample.")
        return None
    except Exception as e:
        logger.error(f"Error processing sample for image {image_path}: {e}", exc_info=True)
        return None


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 VQA Inference on a JSON file")

    # --- Model Paths ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Vision encoder name/path")
    parser.add_argument("--base_llm_name", type=str, required=True, help="Name or path of the base LLM (e.g., 'Qwen/Qwen3-8B').")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained QLoRA adapter directory.")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the trained projector directory (containing config and weights)")

    # --- Input Data ---
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing image/problem/normal_caption triplets")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory where images are located (paths in JSON are relative to this)")
    parser.add_argument("--image_root_2", type=str, default=None, help="Secondary root directory for images with a different path format (e.g., 'p10012261/s50349409')")

    # --- Device ---
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
    if not os.path.isdir(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)
    if not os.path.isdir(args.projector_path):
        logger.error(f"Projector path not found: {args.projector_path}")
        sys.exit(1)
    if not os.path.exists(args.input_json):
        logger.error(f"Input JSON file not found: {args.input_json}")
        sys.exit(1)
    if not os.path.isdir(args.image_root):
        logger.error(f"Image root directory not found: {args.image_root}")
        sys.exit(1)

    # --- Setup Device and Dtype ---
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif device.type == 'cuda':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    logger.info(f"Using device: {device}")
    logger.info(f"Using model dtype: {model_dtype}")

    # --- Load Models and Processor ONCE ---
    logger.info("Loading models and processor...")
    try:
        processor = AutoProcessor.from_pretrained(args.vision_model_name)
        vision_encoder = AutoModel.from_pretrained(
            args.vision_model_name, torch_dtype=model_dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        vision_dim = vision_encoder.config.vision_config.hidden_size

        # --- Load Base LLM --- #
        logger.info(f"Loading base LLM: {args.base_llm_name}")
        llm = AutoModelForCausalLM.from_pretrained(
            args.base_llm_name,
            torch_dtype=model_dtype, # Match training dtype if possible
            device_map='auto', # Load directly to the target device(s)
            trust_remote_code=True,  # Necessary for some models like Qwen
            low_cpu_mem_usage=True,
            # Consider adding quantization config if needed for memory
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16
        ).eval() # Set to eval mode
        logger.info(f"Base LLM '{args.base_llm_name}' loaded.")

        # --- Load Tokenizer --- #
        logger.info(f"Loading tokenizer from adapter path ({args.adapter_path}) or base ({args.base_llm_name})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
            logger.info(f"Tokenizer loaded from adapter path: {args.adapter_path}")
        except Exception:
            logger.warning(f"Could not load tokenizer from adapter path '{args.adapter_path}'. Loading from base model '{args.base_llm_name}'.")
            tokenizer = AutoTokenizer.from_pretrained(args.base_llm_name, trust_remote_code=True)

        # Set padding side for generation
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("pad_token was None. Set to eos_token.")
            llm.config.pad_token_id = tokenizer.pad_token_id # Update model config too

        # --- Load PEFT Adapters --- #
        logger.info(f"Loading PEFT adapters from: {args.adapter_path}")
        try:
            llm = PeftModel.from_pretrained(llm, args.adapter_path)
            # Optional: Merge adapters for potentially faster inference, but requires more memory
            # llm = llm.merge_and_unload()
            # logger.info(f"QLoRA adapters loaded and merged from {args.adapter_path}")
            logger.info(f"QLoRA adapters loaded from {args.adapter_path}")
        except Exception as e:
            logger.error(f"Failed to load PEFT adapters from {args.adapter_path}: {e}", exc_info=True)
            logger.warning("Proceeding with the base LLM without adapters.")
            # exit() # Optional: exit if adapters are mandatory

        llm.eval() # Ensure model is in eval mode after adapter loading
        llm_dim = llm.config.hidden_size
        logger.info(f"LLM dimension after potential adapter load: {llm_dim}")

        # --- Load Projector --- (Use correct llm_dim)
        projection_layer = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim) # Use inferred LLM dimension
        projector_weights_path = os.path.join(args.projector_path, "projector_best.bin")
        if not os.path.exists(projector_weights_path):
             raise FileNotFoundError(f"Projector weights not found at {projector_weights_path}")
        # Use torch.load for .bin files, map to CPU to avoid GPU memory issues during loading
        proj_state_dict = torch.load(projector_weights_path, map_location="cpu") 
        projection_layer.load_state_dict(proj_state_dict)
        projection_layer = projection_layer.to(device, dtype=model_dtype).eval()
        logger.info("Models loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)

    # --- Load JSON Data ---
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {args.input_json}")
    except Exception as e:
        logger.error(f"Failed to load or parse JSON file {args.input_json}: {e}")
        sys.exit(1)

    # --- Prepare Generation Arguments ---
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
    }

    # --- Process Each Sample ---
    logger.info("Starting inference loop...")
    for i, sample in enumerate(data):
        logger.info(f"Processing sample {i+1}/{len(data)}...")
        prediction = process_sample(
            sample=sample,
            image_root=args.image_root,
            image_root_2=args.image_root_2,
            processor=processor,
            vision_encoder=vision_encoder,
            projection_layer=projection_layer,
            tokenizer=tokenizer,
            llm=llm,
            device=device,
            model_dtype=model_dtype,
            generation_args=generation_args
        )

        if prediction is not None:
            image_rel_path = sample.get("image")
            question = sample.get("problem")
            ground_truth = sample.get("normal_caption", "N/A") # Handle missing ground truth

            print("\n" + "="*30 + f" Sample {i+1} " + "="*30)
            print(f"Image Path: {os.path.join(args.image_root, image_rel_path)}")
            print(f"Question:   {question}")
            print(f"Prediction: {prediction}")
            print(f"Answer:     {ground_truth}")
            print("="*70)

    logger.info("Inference loop finished.")