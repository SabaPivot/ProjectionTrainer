import os
from utils import (
    parse_args,
    setup_device,
    load_model,
    load_image_from_json,
    get_candidate_labels,
    process_image,
    display_results,
    display_summary
)

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
    
    candidate_labels = get_candidate_labels()
    
    # Target labels for counting correct predictions
    target_labels = candidate_labels  # Using all candidate labels as targets
    print(f"Using labels for classification: {', '.join(target_labels)}")
    
    # Process all images
    all_results = []
    for i, (image, image_path, metadata) in enumerate(images_data):
        if image is None:
            print(f"Skipping image {i+1}/{len(images_data)}: Could not load")
            continue
        
        print(f"\nProcessing image {i+1}/{len(images_data)}: {os.path.basename(image_path)}")
        
        # Process the image
        results = process_image(image, candidate_labels, processor, model, device)
        
        # Get normal_caption if available
        normal_caption = metadata.get('normal_caption', '')
        
        # Display individual results
        display_results(results, image_path, normal_caption)
        
        # Extract ground truth labels from normal_caption
        ground_truth_labels = []
        if normal_caption:
            # Look for each target label in the normal_caption
            for label in target_labels:
                if label.lower() in normal_caption.lower():
                    ground_truth_labels.append(label)
            
            if ground_truth_labels:
                print(f"Found ground truth labels in caption: {', '.join(ground_truth_labels)}")
        
        # Store results for summary
        top_prediction = results[0]['label']
        result_entry = {
            'image_path': image_path,
            'prediction': top_prediction,
            'probability': results[0]['probability'],
            'ground_truth': normal_caption if normal_caption else None,
            'ground_truth_labels': ground_truth_labels,
            'correct': top_prediction in ground_truth_labels if ground_truth_labels else top_prediction in target_labels,
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
        
        print(f"\nClassification Results:")
        print(f"Correct predictions: {correct_count}/{total_count} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()