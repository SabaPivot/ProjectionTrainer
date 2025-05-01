import json
import argparse
import os
import random # Added for sampling
from collections import defaultdict # Added for grouping

def parse_args():
    parser = argparse.ArgumentParser()
    # Add argument for input file path
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON dataset file")
    parser.add_argument("--candidate_labels", type=str, required=True, help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    parser.add_argument("--output_path", type=str, default="./filtered_data.json", # Changed default to current dir
                      help="Path to save the filtered JSON file")
    # Add argument for random seed for reproducibility during sampling
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()

args = parse_args()

# Load the original dataset from the provided input_path
try:
    with open(args.input_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file {args.input_path} not found.")
    exit(1)
except json.JSONDecodeError:
     print(f"Error: Could not decode JSON from input file {args.input_path}.")
     exit(1)


# Set the random seed
random.seed(args.seed)

# Split candidate labels into a set for efficient lookup, stripping whitespace
candidate_labels_set = {label.strip() for label in args.candidate_labels.split(',')}
print(f"Target labels for filtering and balancing: {sorted(list(candidate_labels_set))}")

# 1. Group samples by their normal_caption if the caption is in candidate_labels_set
grouped_samples = defaultdict(list)
for item in data:
    caption = item.get("normal_caption", "").strip()
    if caption in candidate_labels_set:
        grouped_samples[caption].append(item)

# 2. Calculate counts for each found label
label_counts = {label: len(samples) for label, samples in grouped_samples.items()}
print("Initial counts for found labels:")
for label, count in label_counts.items():
    print(f"  - {label}: {count}")

# Check if any target labels were found
if not label_counts:
    print("Warning: No samples found matching any of the candidate labels. Output file will be empty.")
    min_count = 0
else:
    # 3. Find the minimum count among the labels that were actually found
    min_count = min(label_counts.values())
    print(f"Minimum sample count found: {min_count}. Balancing all included labels to this count.")

# 4. Subsample N items for each label, where N is min_count
balanced_lines = []
for label, samples in grouped_samples.items():
    if len(samples) >= min_count:
        # Randomly sample min_count samples from the list for this label
        balanced_samples_for_label = random.sample(samples, min_count)
        balanced_lines.extend(balanced_samples_for_label)
        print(f"  - Subsampled {label} from {len(samples)} down to {min_count}")
    # else: # This case shouldn't happen if min_count is derived correctly, but good for safety
        # print(f"  - Warning: Label {label} has {len(samples)} samples, which is less than min_count {min_count}. Including all.")
        # balanced_lines.extend(samples)


# Optional: Shuffle the final combined list for better distribution if needed later
random.shuffle(balanced_lines) 
print(f"Total number of samples after balancing: {len(balanced_lines)}")


# 5. Combine and Save
# Ensure the output directory exists
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
    try:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        exit(1)


# Save to the specified output path
try:
    with open(args.output_path, 'w') as f:
        print(f"Saving balanced data to: {args.output_path}")
        json.dump(balanced_lines, f, indent=4)
except IOError as e:
    print(f"Error writing to output file {args.output_path}: {e}")
    exit(1)

print("Filtering and balancing complete.")
