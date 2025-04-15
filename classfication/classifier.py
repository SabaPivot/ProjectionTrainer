import os
from utils import (
    parse_args,
    setup_device,
    load_model,
    load_image_from_json,
    get_candidate_labels,
    process_image,
    display_results,
    display_summary,
    extract_ground_truth_labels
)
from tqdm import tqdm

def main():
    args = parse_args()
    device = setup_device()
    
    model_name = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
    processor, model = load_model(model_name, device)
    
    # Load all images from JSON file
    images_data = load_image_from_json(args.json_file, args.image_root)
    if not images_data:
        print("No images to process. Exiting.")
        return
    
    # Get candidate labels from command-line arguments
    candidate_labels = get_candidate_labels(args)
    
    # Target labels for counting correct predictions
    target_labels = candidate_labels  # Using all candidate labels as targets
    print(f"Using labels for classification: {', '.join(target_labels)}")
    
    print(f"Processing {len(images_data)} images...")
    
    # Process all images
    all_results = []
    # Using tqdm for progress visualization
    for i, (image, image_path, metadata) in enumerate(tqdm(images_data, desc="Classifying", unit="image")):
        if image is None:
            if args.verbose:
                print(f"\nSkipping image {i+1}/{len(images_data)}: Could not load")
            continue
        
        if args.verbose:
            print(f"\nProcessing image {i+1}/{len(images_data)}: {os.path.basename(image_path)}")
        
        # Process the image
        results = process_image(image, candidate_labels, processor, model, device, args.prompt_template)
        
        # Get normal_caption if available
        normal_caption = metadata.get('normal_caption', '')
        
        # Display individual results (if verbose)
        display_results(results, image_path, normal_caption, args.verbose)
        
        # Extract ground truth labels from normal_caption using comma separation
        ground_truth_labels = extract_ground_truth_labels(normal_caption)
        
        if args.verbose and ground_truth_labels:
            print(f"Ground truth labels from caption: {', '.join(ground_truth_labels)}")
        
        # Store results for summary
        top_prediction = results[0]['label']
        
        # Check if prediction is correct (appears in comma-separated normal_caption)
        is_correct = top_prediction in ground_truth_labels if ground_truth_labels else False
        
        result_entry = {
            'image_path': image_path,
            'prediction': top_prediction,
            'probability': results[0]['probability'],
            'ground_truth': normal_caption if normal_caption else None,
            'ground_truth_labels': ground_truth_labels,
            'correct': is_correct,
            'metadata': metadata
        }
        all_results.append(result_entry)
    
    # Display summary of all results
    if all_results:
        display_summary(all_results, target_labels)
        
        # Count correct predictions
        correct_count = sum(1 for r in all_results if r['correct'])
        total_count = len(all_results)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\nFINAL CLASSIFICATION ACCURACY: {correct_count}/{total_count} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()