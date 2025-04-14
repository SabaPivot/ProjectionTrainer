import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Classify X-ray images with SigLIP")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--json_file", type=str, required=True, help="JSON file containing image filenames")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to use {device}")
    
    # Load model and processor
    model_name = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name).to(device)
    
    # Load image filename from JSON
    print(f"Loading image filenames from: {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        # Assuming JSON file has at least one entry with an 'image' key
        if isinstance(data, list) and len(data) > 0 and 'image' in data[0]:
            image_filename = data[0]['image']
        else:
            print("JSON file format not recognized. Please provide a valid file.")
            return
    
    # Create image path
    image_path = os.path.join(args.image_root, image_filename)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load image from local path
    print(f"Analyzing X-ray image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Define candidate labels for chest X-ray pathologies
    candidate_labels = ["cardiomegaly", "atelectasis", "pneumonia", "effusion", "nodule", "mass", "no finding"]
    print(f"Using candidate labels: {', '.join(candidate_labels)}")
    
    # Create prompt templates for X-ray specific context
    texts = [f"This X-ray shows {label}." for label in candidate_labels]
    
    # Process all labels at once for efficiency
    # Note: Using padding="max_length" as recommended in SigLIP documentation
    inputs = processor(
        text=texts,
        images=image, 
        return_tensors="pt", 
        padding="max_length"
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits_per_image (Comparing each label to one image)
    logits_per_image = outputs.logits_per_image[0]  # Shape: (num_labels,)
    
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
    
    # Display results
    print("\nClassification Results:")
    print("-" * 45)
    print(f"{'Pathology':<20} {'Probability %':<15}")
    print("-" * 45)
    
    for res in results:
        print(f"{res['label']:<20} {res['probability']}%")
    
    print("\nTop diagnosis:", results[0]['label'], f"({results[0]['probability']}% probability)")

if __name__ == "__main__":
    main()