import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import os
import json
import argparse

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

def load_image_from_json(json_file, image_root):
    """Load image filename from JSON and return the loaded image."""
    print(f"Loading image filenames from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
        # Assuming JSON file has at least one entry with an 'image' key
        if isinstance(data, list) and len(data) > 0 and 'image' in data[0]:
            image_filename = data[0]['image']
        else:
            print("JSON file format not recognized. Please provide a valid file.")
            return None, None
    
    # Create image path
    image_path = os.path.join(image_root, image_filename)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None
    
    # Load image from local path
    print(f"Analyzing X-ray image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image, image_path

def get_candidate_labels():
    """Return the list of candidate X-ray pathology labels."""
    candidate_labels = ["cardiomegaly", "atelectasis", "pneumonia", "effusion", "nodule", "mass", "no finding"]
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

def display_results(results):
    """Display classification results in a formatted table."""
    print("\nClassification Results:")
    print("-" * 45)
    print(f"{'Pathology':<20} {'Probability %':<15}")
    print("-" * 45)
    
    for res in results:
        print(f"{res['label']:<20} {res['probability']}%")
    
    print("\nTop diagnosis:", results[0]['label'], f"({results[0]['probability']}% probability)") 