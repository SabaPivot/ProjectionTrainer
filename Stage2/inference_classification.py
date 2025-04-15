# Stage2/inference_classification.py

import os
import torch
import logging
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from transformers import Gemma3ForCausalLM
from safetensors.torch import load_file
import sys
import json
from tqdm import tqdm
# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Stage1.projectors import MLPProjector

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_ground_truth_labels(normal_caption: str) -> list:
    """
    Extract ground truth labels from normal_caption by splitting on commas.
    
    Args:
        normal_caption: The caption string containing labels separated by commas
        
    Returns:
        List of labels extracted from the caption
    """
    if not normal_caption:
        return []
        
    # Split by comma and strip whitespace
    return [label.strip() for label in normal_caption.split(", ")]

def process_sample(sample, image_root, processor, vision_encoder, projection_layer, llm_tokenizer, llm_model, device, model_dtype, candidate_labels, prompt_template="xray"):
    """Loads image, runs inference for one sample dict from JSON."""
    try:
        image_rel_path = sample.get("image")
        if not image_rel_path:
            logger.warning(f"Skipping sample due to missing 'image': {sample}")
            return None

        image_path = os.path.join(image_root, image_rel_path)

        # --- Load and Preprocess Image ---
        image = Image.open(image_path).convert('RGB')
        img_size = processor.image_processor.size['height'] if 'height' in processor.image_processor.size else 384
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
            
            # 3. Process each candidate label
            results = []
            for label in candidate_labels:
                # Create prompt based on template
                if prompt_template == "photo":
                    prompt = f"This is a photo of {label}."
                elif prompt_template == "none":
                    prompt = label
                else:  # Default to "xray"
                    prompt = f"This X-ray shows {label}."
                
                # Tokenize prompt
                prompt_tokens = llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(device)
                
                # Embed prompt tokens
                input_embed_layer = llm_model.get_input_embeddings()
                prompt_embeds = input_embed_layer(prompt_tokens)
                
                # Concatenate embeddings
                inputs_embeds = torch.cat([projected_embeds, prompt_embeds], dim=1)
                batch_size, sequence_length, _ = inputs_embeds.shape
                attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
                
                # Generate score
                outputs = llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=1,  # We only need a single token for classification
                    eos_token_id=llm_tokenizer.eos_token_id,
                    pad_token_id=llm_tokenizer.pad_token_id,
                    do_sample=False,  # Use greedy decoding for classification
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Get the score for the generated token
                score = outputs.scores[0][0, 0].item()  # Get the score of the first token
                results.append({
                    "label": label,
                    "score": score
                })
            
            # Normalize scores to probabilities
            scores = torch.tensor([r["score"] for r in results])
            probs = torch.nn.functional.softmax(scores, dim=0)
            
            # Update results with probabilities
            for i, result in enumerate(results):
                result["probability"] = round(probs[i].item() * 100, 2)
            
            # Sort by probability (highest first)
            results = sorted(results, key=lambda x: x["probability"], reverse=True)
            
            return results

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}. Skipping sample.")
        return None
    except Exception as e:
        logger.error(f"Error processing sample for image {image_path}: {e}", exc_info=True)
        return None

def display_results(results, image_path, normal_caption, verbose=False):
    """Display classification results in a formatted table for a single image."""
    if not verbose:
        return
        
    if image_path:
        print(f"\nClassification Results for {os.path.basename(image_path)}:")
    else:
        print("\nClassification Results:")
    
    if normal_caption:
        print(f"Ground truth: {normal_caption[:100]}..." if len(normal_caption) > 100 else normal_caption)
        
    print("-" * 45)
    print(f"{'Pathology':<20} {'Probability %':<15}")
    print("-" * 45)
    
    for res in results:
        print(f"{res['label']:<20} {res['probability']}%")
    
    print("\nTop diagnosis:", results[0]['label'], f"({results[0]['probability']}% probability)")

def display_summary(all_results, target_labels):
    """Display a summary of results for all processed images."""
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    total = len(all_results)
    if total == 0:
        print("No results to display")
        return
        
    # Count predictions matching each target label
    correct = sum(1 for r in all_results if r['prediction'] in target_labels)
    
    print(f"Total images processed: {total}")
    print(f"Images classified as {', '.join(target_labels)}: {correct} ({correct/total*100:.2f}%)")
    
    # Analyze accuracy using normal_caption if available
    has_ground_truth = sum(1 for r in all_results if r.get('ground_truth'))
    if has_ground_truth:
        print(f"\nImages with ground truth labels: {has_ground_truth}")
        matches = sum(1 for r in all_results if r.get('correct', False))
        print(f"Predictions matching ground truth: {matches} ({matches/has_ground_truth*100:.2f}%)")
        
        # Calculate per-class accuracy statistics
        print("\nClassification Accuracy distribution:")
        per_class_stats = {}
        
        # Initialize counts for each target label
        for label in target_labels:
            per_class_stats[label] = {"total": 0, "correct": 0}
        
        # Count occurrences and correct predictions for each class
        for result in all_results:
            if not result.get('ground_truth_labels'):
                continue
                
            # For each class in this image's ground truth
            for gt_label in result['ground_truth_labels']:
                if gt_label in target_labels:
                    per_class_stats[gt_label]["total"] += 1
                    
                    # If prediction matches this specific class
                    if result['prediction'] == gt_label:
                        per_class_stats[gt_label]["correct"] += 1
        
        # Display per-class accuracy
        for label, stats in sorted(per_class_stats.items()):
            if stats["total"] > 0:
                accuracy = (stats["correct"] / stats["total"]) * 100
                print(f"  {label:<20}: {stats['correct']}/{stats['total']} ({accuracy:.2f}%)")
            else:
                print(f"  {label:<20}: 0/0 (0.00%)")
        
        # Count and display distribution of normal_caption values
        print("\nGround Truth Distribution:")
        normal_caption_counts = {}
        for result in all_results:
            caption = result.get('ground_truth')
            if caption:
                normal_caption_counts[caption] = normal_caption_counts.get(caption, 0) + 1
        
        # Sort by frequency (highest first)
        for caption, count in sorted(normal_caption_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / has_ground_truth) * 100
            print(f"  {caption:<40}: {count} ({percentage:.2f}%)")
    
    # Display distribution of top predictions
    prediction_counts = {}
    for r in all_results:
        pred = r['prediction']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    for label, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<20}: {count} ({count/total*100:.2f}%)")
    
    print("="*60)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Classification Inference on a JSON file")

    # --- Model Paths ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Vision encoder name/path")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to the fine-tuned LLM directory (must contain config.json)")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the trained projector directory (containing config and weights)")

    # --- Input Data ---
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing image/normal_caption pairs")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory where images are located (paths in JSON are relative to this)")
    parser.add_argument("--candidate_labels", type=str, default="Atelectasis, No Finding", 
                      help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    parser.add_argument("--prompt_template", type=str, default="xray", 
                      choices=["xray", "photo", "none"], 
                      help="Prompt template to use (xray: 'This X-ray shows {label}.', photo: 'This is a photo of {label}.', none: just '{label}')")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results for each image")

    # --- Device ---
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help=f"Device to use (e.g., cuda:0, cuda:1, cpu). Default: {default_device}")

    args = parser.parse_args()

    # --- Validate Paths ---
    if not os.path.isdir(args.llm_path):
        logger.error(f"LLM path not found: {args.llm_path}")
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

        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_config_path = os.path.join(args.llm_path, "config.json")
        if not os.path.exists(llm_config_path):
             logger.error(f"Critical Error: 'config.json' not found in {args.llm_path}")
             sys.exit(1)
        llm_model = Gemma3ForCausalLM.from_pretrained(
            args.llm_path, torch_dtype=model_dtype, low_cpu_mem_usage=True, attn_implementation="eager"
        ).to(device).eval()
        llm_dim = llm_model.config.hidden_size

        projection_layer = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim)
        projector_weights_path = os.path.join(args.projector_path, "model.safetensors")
        if not os.path.exists(projector_weights_path):
             raise FileNotFoundError(f"Projector weights not found at {projector_weights_path}")
        proj_state_dict = load_file(projector_weights_path, device="cpu")
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

    # --- Get Candidate Labels ---
    candidate_labels = [label.strip() for label in args.candidate_labels.split(",")]
    logger.info(f"Using candidate labels: {', '.join(candidate_labels)}")

    # --- Process Each Sample ---
    logger.info("Starting inference loop...")
    all_results = []
    
    for i, sample in enumerate(tqdm(data, desc="Classifying", unit="image")):
        logger.info(f"Processing sample {i+1}/{len(data)}...")
        results = process_sample(
            sample=sample,
            image_root=args.image_root,
            processor=processor,
            vision_encoder=vision_encoder,
            projection_layer=projection_layer,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model,
            device=device,
            model_dtype=model_dtype,
            candidate_labels=candidate_labels,
            prompt_template=args.prompt_template
        )

        if results is not None:
            image_rel_path = sample.get("image")
            normal_caption = sample.get("normal_caption", "")
            
            # Display individual results (if verbose)
            display_results(results, os.path.join(args.image_root, image_rel_path), normal_caption, args.verbose)
            
            # Extract ground truth labels from normal_caption
            ground_truth_labels = extract_ground_truth_labels(normal_caption)
            
            # Store results for summary
            top_prediction = results[0]['label']
            
            # Check if prediction is correct (appears in comma-separated normal_caption)
            is_correct = top_prediction in ground_truth_labels if ground_truth_labels else False
            
            result_entry = {
                'image_path': os.path.join(args.image_root, image_rel_path),
                'prediction': top_prediction,
                'probability': results[0]['probability'],
                'ground_truth': normal_caption if normal_caption else None,
                'ground_truth_labels': ground_truth_labels,
                'correct': is_correct,
                'metadata': sample
            }
            all_results.append(result_entry)

    # Display summary of all results
    if all_results:
        display_summary(all_results, candidate_labels)
        
        # Count correct predictions
        correct_count = sum(1 for r in all_results if r['correct'])
        total_count = len(all_results)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\nFINAL CLASSIFICATION ACCURACY: {correct_count}/{total_count} ({accuracy:.2f}%)")

    logger.info("Inference loop finished.") 