import os
from utils import (
    parse_args,
    setup_device,
    load_model,
    load_image_from_json,
    get_candidate_labels,
    process_image,
    display_results
)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup device (CPU/GPU)
    device = setup_device()
    
    # Load SigLIP model and processor
    model_name = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
    processor, model = load_model(model_name, device)
    
    # Load image from JSON file
    image, _ = load_image_from_json(args.json_file, args.image_root)
    if image is None:
        return
    
    # Get candidate labels for classification
    candidate_labels = get_candidate_labels()
    
    # Process image with SigLIP model
    results = process_image(image, candidate_labels, processor, model, device)
    
    # Display classification results
    display_results(results)

if __name__ == "__main__":
    main()