# Stage2/inference_generation.py

import os
import torch
import logging
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file
import sys
import json
from tqdm import tqdm
from collections import Counter
# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Stage1.projectors import MLPProjector

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_sample(sample, image_root, processor, vision_encoder, projection_layer, tokenizer, llm, device, model_dtype, max_length=128):
    """Loads image, runs inference for one sample dict from JSON."""
    try:
        image_rel_path = sample.get("image")
        normal_caption = sample.get("normal_caption", "")
        if not image_rel_path:
            logger.warning(f"Skipping sample due to missing 'image': {sample}")
            return None

        image_path = os.path.join(image_root, image_rel_path)

        # --- Load and Preprocess Image ---
        image = Image.open(image_path).convert('RGB')
        img_size = 384
        image = image.resize((img_size, img_size))
        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs.pixel_values.to(device, dtype=model_dtype)

        # --- Generate Embeddings ---
        with torch.no_grad():
            # 1. Get Visual Features
            vision_outputs = vision_encoder.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            )
            # Use patch embeddings (discard class token)
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
            
            # 2. Project Visual Features
            projected_embeds = projection_layer(patch_embeddings)
            
            # 3. Create prompt for generation
            prompt = "Identify the diseases in this chest X-ray image. Provide your answer in a single word or phrase."
            
            # Tokenize prompt
            prompt_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(device)
            
            # Embed prompt tokens
            input_embed_layer = llm.get_input_embeddings()
            prompt_embeds = input_embed_layer(prompt_tokens)
            
            # Concatenate embeddings
            inputs_embeds = torch.cat([projected_embeds, prompt_embeds], dim=1)
            batch_size, sequence_length, _ = inputs_embeds.shape
            attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
            
            # Generate answer
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                return_dict_in_generate=True
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Check if normal_caption appears in the generated text
            is_correct = normal_caption.lower() in generated_text.lower() if normal_caption else False
            
            return {
                "generated_text": generated_text,
                "normal_caption": normal_caption,
                "is_correct": is_correct
            }

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}. Skipping sample.")
        return None
    except Exception as e:
        logger.error(f"Error processing sample for image {image_path}: {e}", exc_info=True)
        return None

def display_results(result, image_path, verbose=False):
    """Display generation results in a formatted table for a single image."""
    if not verbose:
        return
        
    if image_path:
        print(f"\nGeneration Results for {os.path.basename(image_path)}:")
    else:
        print("\nGeneration Results:")
    
    print("-" * 60)
    print(f"Generated Text: {result['generated_text'][:100]}..." if len(result['generated_text']) > 100 else result['generated_text'])
    
    if result['normal_caption']:
        print(f"Ground Truth: {result['normal_caption']}")
        print(f"Correct: {'Yes' if result['is_correct'] else 'No'}")
    
    print("-" * 60)

def display_summary(all_results, candidate_labels):
    """Display a detailed summary of results, including per-label stats."""
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    total = len(all_results)
    if total == 0:
        print("No results to display")
        return
        
    # Overall accuracy (ground truth label found in generated text)
    overall_correct = sum(1 for r in all_results if r['is_correct'])
    
    print(f"Total images processed: {total}")
    
    # Specific candidate label info
    candidate_label_str = ", ".join(candidate_labels)
    print(f"Images classified as {candidate_label_str}: {total} (100.00%)") # Assuming input was filtered
    
    print(f"\nImages with ground truth labels: {total}") # All should have ground truth
    print(f"Predictions matching ground truth (exact label string): {overall_correct} ({overall_correct/total*100:.2f}%)")
    
    # --- Calculate detailed stats ---
    ground_truth_counts = Counter()
    prediction_counts = Counter()
    label_correct_counts = Counter()
    
    for result in all_results:
        gt_label = result.get('normal_caption')
        generated_text_lower = result.get('generated_text', '').lower()
        
        if gt_label:
            ground_truth_counts[gt_label] += 1
            
            # Check if ground truth label is in generated text
            if gt_label.lower() in generated_text_lower:
                 label_correct_counts[gt_label] += 1 # This contributes to overall_correct
        
        # Check for presence of any candidate label in generated text
        for label in candidate_labels:
            if label.lower() in generated_text_lower:
                prediction_counts[label] += 1
                
    # --- Display Accuracy per Label ---
    print("\nClassification Accuracy distribution (Ground Truth label found in prediction):")
    # Sort labels based on ground_truth_counts for consistent ordering
    sorted_gt_labels = sorted(ground_truth_counts.keys(), key=lambda k: ground_truth_counts[k], reverse=True)
    for label in sorted_gt_labels:
        correct_count = label_correct_counts[label]
        total_count = ground_truth_counts[label]
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        print(f"  {label:<20}: {correct_count}/{total_count} ({accuracy:.2f}%)")
        
    # --- Display Ground Truth Distribution ---
    print("\nGround Truth Distribution:")
    # Use the same sorted order as above
    for label in sorted_gt_labels:
        count = ground_truth_counts[label]
        percentage = (count / total) * 100
        print(f"  {label:<40}: {count} ({percentage:.2f}%)")

    # --- Display Prediction Distribution ---
    print("\nPrediction distribution (Candidate label found in prediction):")
    # Sort predicted labels by frequency
    sorted_pred_labels = sorted(prediction_counts.keys(), key=lambda k: prediction_counts[k], reverse=True)
    for label in sorted_pred_labels:
        count = prediction_counts[label]
        percentage = (count / total) * 100 # Percentage of *total images* where this label was predicted
        print(f"  {label:<20}: {count} ({percentage:.2f}%)")
    
    print("="*60)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Generation Inference on a JSON file")

    # --- Model Paths ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Vision encoder name/path")
    parser.add_argument("--base_llm_name", type=str, required=True, help="Name or path of the base LLM used for adapter training.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained QLoRA adapter directory (language_model folder).")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the trained projector directory (e.g., projection_layer folder containing projector_best.bin).")

    # --- Input Data ---
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing image/normal_caption pairs")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory where images are located (paths in JSON are relative to this)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated text")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results for each image")
    parser.add_argument("--candidate_labels", type=str, required=True, help="Comma-separated string of candidate labels used for this run (e.g., 'Atelectasis,No Finding')")

    # --- Device ---
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help=f"Device to use (e.g., cuda:0, cuda:1, cpu). Default: {default_device}")

    args = parser.parse_args()

    # --- Validate Paths ---
    if not os.path.isdir(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)
    if not os.path.isdir(args.projector_path):
        logger.error(f"Projector path not found: {args.projector_path}")
        sys.exit(1)
    projector_weights_expected_path = os.path.join(args.projector_path, "projector_best.bin")
    if not os.path.exists(projector_weights_expected_path):
        logger.error(f"Projector weights file not found at expected location: {projector_weights_expected_path}")
    if not os.path.exists(args.input_json):
        logger.error(f"Input JSON file not found: {args.input_json}")
        sys.exit(1)
    if not os.path.isdir(args.image_root):
        logger.error(f"Image root directory not found: {args.image_root}")
        sys.exit(1)

    # Parse candidate labels
    candidate_labels_list = [label.strip() for label in args.candidate_labels.split(',')]
    logger.info(f"Using candidate labels for analysis: {candidate_labels_list}")

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

        # --- Load Base LLM ---
        logger.info(f"Loading base LLM: {args.base_llm_name}")
        llm = AutoModelForCausalLM.from_pretrained(
            args.base_llm_name,
            torch_dtype=model_dtype, 
            device_map='auto', 
            trust_remote_code=True, 
            low_cpu_mem_usage=True,
        ).eval() # Set to eval mode
        logger.info(f"Base LLM '{args.base_llm_name}' loaded.")

        # --- Load Tokenizer ---
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

        # --- Load PEFT Adapters ---
        logger.info(f"Loading PEFT adapters from: {args.adapter_path}")
        try:
            llm = PeftModel.from_pretrained(llm, args.adapter_path)
            logger.info(f"QLoRA adapters loaded from {args.adapter_path}")
        except Exception as e:
            logger.error(f"Failed to load PEFT adapters from {args.adapter_path}: {e}", exc_info=True)
            logger.warning("Proceeding with the base LLM without adapters.")

        llm.eval() # Ensure model is in eval mode after adapter loading
        llm_dim = llm.config.hidden_size
        logger.info(f"LLM dimension after potential adapter load: {llm_dim}")

        # --- Load Projector ---
        projection_layer = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim) 
        projector_weights_path = os.path.join(args.projector_path, "projector_best.bin")
        logger.info(f"Loading projector weights from: {projector_weights_path}")
        if not os.path.exists(projector_weights_path):
             # Log error but allow to continue to potentially fail later if needed
             logger.error(f"Projector weights file not found at expected location: {projector_weights_path}") 
        # Use torch.load for .bin files
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

    # --- Process Each Sample ---
    logger.info("Starting inference loop...")
    all_results = []
    processed_count = 0
    skipped_count = 0
    
    logger.info(f"Starting inference on {len(data)} samples from {args.input_json}...")
    for i, sample_data in enumerate(tqdm(data, desc="Processing images")):
        image_rel_path = sample_data.get("image")
        result = process_sample(
            sample_data, 
            args.image_root, 
            processor, 
            vision_encoder, 
            projection_layer, 
            tokenizer, 
            llm, 
            device, 
            model_dtype, 
            max_length=args.max_length
        )
        
        if result:
            all_results.append(result)
            processed_count += 1
            # Log model answer and ground truth for each sample
            logger.info(f"Image: {image_rel_path}, Model Answer: {result['generated_text']}, Ground Truth: {result['normal_caption']}")
            if args.verbose:
                display_results(result, os.path.join(args.image_root, image_rel_path), verbose=True)
        else:
            skipped_count += 1
        
        # Optional: Log progress periodically if needed
        # if (i + 1) % 100 == 0:
        #     logger.info(f"Processed {i + 1}/{len(data)} samples...")

    logger.info(f"Inference completed. Processed: {processed_count}, Skipped: {skipped_count}")

    # --- Display Summary of Results ---
    if processed_count > 0:
        display_summary(all_results, candidate_labels_list)
    else:
        logger.info("No samples were processed successfully.")

    logger.info("Script finished.") 