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
    target_labels = ["cardiomegaly", "atelectasis", "effusion"]
    print(f"Counting matches for target labels: {', '.join(target_labels)}")
    
    # Process all images
    all_results = []
    for i, (image, image_path, metadata) in enumerate(images_data):
        if image is None:
            print(f"Skipping image {i+1}/{len(images_data)}: Could not load")
            continue
        
        print(f"\nProcessing image {i+1}/{len(images_data)}: {os.path.basename(image_path)}")
        
        # Process the image
        results = process_image(image, candidate_labels, processor, model, device)
        
        # Display individual results
        display_results(results, image_path)
        
        # Store results for summary
        top_prediction = results[0]['label']
        result_entry = {
            'image_path': image_path,
            'prediction': top_prediction,
            'probability': results[0]['probability'],
            'correct': top_prediction.lower() in target_labels,
            'metadata': metadata
        }
        all_results.append(result_entry)
    
    # Display summary of all results
    if all_results:
        display_summary(all_results, target_labels)
        
        # Count correct predictions
        correct_count = sum(1 for r in all_results if r['correct'])
        print(f"\nCount of correct predictions (matching {', '.join(target_labels)}): {correct_count}/{len(all_results)}")

if __name__ == "__main__":
    main()