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
            
            # Extract normal_caption if available
            normal_caption = item.get('normal_caption', '')
            if normal_caption:
                print(f"Entry {i} has normal_caption: {normal_caption[:50]}..." if len(normal_caption) > 50 else normal_caption)
            else:
                print(f"Entry {i} has no normal_caption")
            
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
        print(f"With normal_caption: {sum(1 for _, _, meta in images_data if 'normal_caption' in meta)}")
        
    return images_data

def get_candidate_labels() -> List[str]:
    """Return the list of candidate X-ray pathology labels."""
    # FIXME: TO CHANGE TARGET CLASSES, CHANGE THIS LIST
    candidate_labels = ["Atelectasis", "Effusion", "Cardiomegaly"]
    print(f"Using candidate labels: {', '.join(candidate_labels)}")
    return candidate_labels

def process_image(image, candidate_labels, processor, model, device):
    """Process image with SigLIP model and return classification results."""
    # Create prompt templates for X-ray specific context
    texts = [f"This X-ray shows {label}." for label in candidate_labels]
    
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

def display_results(results: List[Dict], image_path: str = None, normal_caption: str = None) -> None:
    """Display classification results in a formatted table for a single image."""
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
    correct = sum(1 for r in all_results if r['prediction'] in target_labels)
    
    print(f"Total images processed: {total}")
    print(f"Images classified as {', '.join(target_labels)}: {correct} ({correct/total*100:.2f}%)")
    
    # Analyze accuracy using normal_caption if available
    has_ground_truth = sum(1 for r in all_results if r.get('ground_truth'))
    if has_ground_truth:
        print(f"\nImages with ground truth labels: {has_ground_truth}")
        matches = sum(1 for r in all_results if r.get('ground_truth') and r['prediction'] in r.get('ground_truth_labels', []))
        print(f"Predictions matching ground truth: {matches} ({matches/has_ground_truth*100:.2f}%)")
    
    # Display distribution of top predictions
    prediction_counts = {}
    for r in all_results:
        pred = r['prediction']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    for label, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<20}: {count} ({count/total*100:.2f}%)")
    
    print("="*60)