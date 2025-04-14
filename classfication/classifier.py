import torch
from PIL import Image
import requests
from transformers import SiglipProcessor, SiglipModel
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load model and processor
model_name = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
processor = SiglipProcessor.from_pretrained(model_name)
model = SiglipModel.from_pretrained(model_name).to(device)

# Load image
url = 'https://upload.wikimedia.org/wikipedia/commons/7/7a/Cardiomegally.PNG'
print(f"Analyzing X-ray image: {url}")
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

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

# For multi-class classification (determining the most likely single condition),
# softmax provides a more interpretable probability distribution
probs = torch.nn.functional.softmax(logits_per_image, dim=0)

# Create results
results = []
for i, label in enumerate(candidate_labels):
    results.append({
        "score": round(logits_per_image[i].item(), 4),
        "probability": round(probs[i].item() * 100, 2),
        "label": label
    })

# Sort by score (highest first)
results = sorted(results, key=lambda x: x["score"], reverse=True)

# Display results
print("\nClassification Results:")
print("-" * 60)
print(f"{'Pathology':<15} {'Raw Score':<15} {'Probability %':<15}")
print("-" * 60)

for res in results:
    print(f"{res['label']:<15} {res['score']:<15} {res['probability']}%")

print("\nTop diagnosis:", results[0]['label'], f"({results[0]['probability']}% probability)")
print("\nNote:")
print("- Raw scores are logits (unnormalized predictions)")
print("- Probability % is calculated using softmax for multi-class interpretation")