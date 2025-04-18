import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import os
import json
import argparse
from typing import List, Tuple, Dict, Optional, Any, Union

def parse_args():
    """Parse command line arguments for X-ray classification."""
    parser = argparse.ArgumentParser(description="Classify X-ray images with SigLIP")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--json_file", type=str, required=True, help="JSON file containing image filenames")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results for each image")
    parser.add_argument("--candidate_labels", type=str, default="Atelectasis, No Finding", 
                      help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    parser.add_argument("--prompt_template", type=str, default="xray", 
                      choices=["xray", "photo", "none"], 
                      help="Prompt template to use (xray: 'This X-ray shows {label}.', photo: 'This is a photo of {label}.', none: just '{label}')")
    return parser.parse_args()

def setup_device():
    """Setup and return the appropriate device (CPU/GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to use {device}")
    return device

def load_model(model_name, device):
    """Load SigLIP model and processor."""
    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name).to(device)
    return processor, model

def load_image_from_json(json_file: str, image_root: str) -> List[Tuple[Optional[Image.Image], str, Dict[str, Any]]]:
    """
    Load all image filenames from JSON and return a list of loaded images with metadata.
    
    Returns:
        List of tuples, each containing (image object, image path, metadata dictionary)
        If an image can't be loaded, image will be None for that entry
        
    Note:
        The metadata dictionary will contain all fields from the JSON entry,
        including 'image' and 'normal_caption' if available.
    """
    print(f"Loading images from: {json_file}")
    images_data = []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        if not isinstance(data, list):
            print("JSON file format not recognized. Expected a list of image entries.")
            return []
        
        print(f"Found {len(data)} entries in JSON file")
        
        for i, item in enumerate(data):
            if 'image' not in item:
                print(f"Skipping entry {i}: No 'image' field found")
                continue
                
            image_filename = item['image']
            image_path = os.path.join(image_root, image_filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                images_data.append((None, image_path, item))
                continue
            
            try:
                # Load image from local path
                image = Image.open(image_path).convert("RGB")
                images_data.append((image, image_path, item))
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                images_data.append((None, image_path, item))
    
    if not images_data:
        print("No valid entries found in the JSON file")
    else:
        print(f"Successfully loaded {sum(1 for img, _, _ in images_data if img is not None)} images")
        
    return images_data

def extract_ground_truth_labels(normal_caption: str) -> List[str]:
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

def get_candidate_labels(args) -> List[str]:
    """
    Return the list of candidate X-ray pathology labels.
    
    Args:
        args: Command line arguments containing candidate_labels
        
    Returns:
        List of candidate labels for classification
    """
    # Parse the comma-separated candidate labels from command line arguments
    # Strip whitespace to handle both "Label1, Label2" and "Label1,Label2" formats
    candidate_labels = [label.strip() for label in args.candidate_labels.split(",")]
    print(f"Using candidate labels: {', '.join(candidate_labels)}")
    return candidate_labels

def process_image(image, candidate_labels, processor, model, device, prompt_template="xray"):
    """Process image with SigLIP model and return classification results."""
    # Create prompt templates based on the specified template type
    if prompt_template == "photo":
        texts = [f"This is a photo of {label}." for label in candidate_labels] # Siglip default
    elif prompt_template == "none":
        texts = [label for label in candidate_labels]
    else:  # Default to "xray"
        texts = [f"This X-ray shows {label}." for label in candidate_labels] #
    # There is a {label}. 
    # Process all labels at once for efficiency
    inputs = processor(
        text=texts,
        images=image, 
        return_tensors="pt", 
        padding="max_length"
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits_per_image
    logits_per_image = outputs.logits_per_image[0]  # Shape: (num_labels,)
    
    # Calculate softmax probabilities
    probs = torch.nn.functional.softmax(logits_per_image, dim=0)
    
    # Create results
    results = []
    for i, label in enumerate(candidate_labels):
        results.append({
            "probability": round(probs[i].item() * 100, 2),
            "label": label
        })
    
    # Sort by probability (highest first)
    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return results

def display_results(results: List[Dict], image_path: str = None, normal_caption: str = None, verbose: bool = False) -> None:
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

def display_summary(all_results: List[Dict[str, Any]], target_labels: List[str]) -> None:
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