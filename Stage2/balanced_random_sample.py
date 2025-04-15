import json
import argparse
import os
import random
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_labels", type=str, required=True, help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    parser.add_argument("--output_path", type=str, default="/mnt/samuel/Siglip/filtered_formatted_Class_QA.json", 
                      help="Path to save the filtered JSON file")
    parser.add_argument("--sample_size", type=int, default=100, help="Total number of samples to randomly select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Load the original dataset
    with open("/mnt/WHY/VLM/Deepseek/VLM-R1/QA_DATASET/Test/SQA/formatted_Class_QA.json", "r") as f:
        data = json.load(f)
    
    # Split candidate labels into a list, stripping whitespace
    candidate_labels = [label.strip() for label in args.candidate_labels.split(',')]
    print(f"Filtering for EXACT matches to: {candidate_labels}")
    
    # Filter data to only include samples with exact matches to candidate labels
    filtered_data = []
    for item in data:
        if item["normal_caption"] in candidate_labels:
            filtered_data.append(item)
    
    print(f"Found {len(filtered_data)} entries with exact matches to candidate labels")
    
    # Group samples by label
    label_groups = defaultdict(list)
    for item in filtered_data:
        label_groups[item["normal_caption"]].append(item)
    
    # Calculate samples per label for balanced distribution
    num_labels = len(candidate_labels)
    samples_per_label = args.sample_size // num_labels
    remainder = args.sample_size % num_labels
    
    # Randomly sample from each label group
    balanced_samples = []
    for i, label in enumerate(candidate_labels):
        if label in label_groups:
            # Add one extra sample to some labels if there's a remainder
            num_samples = samples_per_label + (1 if i < remainder else 0)
            # Ensure we don't try to sample more than available
            num_samples = min(num_samples, len(label_groups[label]))
            sampled = random.sample(label_groups[label], num_samples)
            balanced_samples.extend(sampled)
            print(f"Sampled {num_samples} images for label '{label}'")
        else:
            print(f"Warning: No samples found for label '{label}'")
    
    # Shuffle the final sample
    random.shuffle(balanced_samples)
    
    print(f"Total balanced sample size: {len(balanced_samples)}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to the specified output path
    with open(args.output_path, 'w') as f:
        print(f"Saving balanced random sample to: {args.output_path}")
        json.dump(balanced_samples, f, indent=4)

if __name__ == "__main__":
    main() 