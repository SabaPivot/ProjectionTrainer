import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipVisionModel, AutoConfig, AutoProcessor
from PIL import Image
import os
import json
import logging
import argparse
from tqdm import tqdm
# import numpy as np - No longer needed directly here
# from sklearn.model_selection import train_test_split - No longer needed directly here

# --- Import Utilities --- #
# Use relative imports assuming train.py is run within the soombit package context
# or the parent directory is in PYTHONPATH
from .train_utils import (
    setup_environment,
    setup_model_processor,
    load_and_prepare_data,
    setup_optimizer,
    run_training_loop
)
# --- Import Model Components --- #
# Use relative imports
from .models import (
    XrayClassificationDataset,
    classification_collate_fn,
    AbnormalityClassifier
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Default Configuration (can be overridden by args) ---
DEFAULT_DATA_JSON = "/mnt/samuel/Siglip/soombit/single_label_dataset.json"
DEFAULT_IMAGE_ROOT = "/mnt/data/CXR/NIH Chest X-rays_jpg"
DEFAULT_IMAGE_ROOT_2 = "/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
DEFAULT_VISION_MODEL_NAME = "/mnt/samuel/Siglip/soombit/checkpoint/epoch_16" # StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
DEFAULT_IMG_SIZE = 384
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_WORKERS = 4
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BACKBONE_LEARNING_RATE = 1e-8
DEFAULT_OUTPUT_BASE_DIR = "./soombit/checkpoints"
ALL_POSSIBLE_CLASSES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion"]

# === Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train X-ray Abnormality Classifier")
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID (e.g., EXP01)')
    parser.add_argument('--class_names', type=str, required=True,
                        help='Comma-separated string of class names to use (e.g., "No Finding,Atelectasis")')
    parser.add_argument('--freeze_mode', type=str, required=True, choices=['Freeze', 'Unfreeze', '1EpochUnfreeze'],
                        help='Vision encoder freezing strategy')
    parser.add_argument('--handle_abnormal', action='store_true',
                        help='Map Atelectasis, Cardiomegaly, Effusion to a single "Abnormal" class')
    parser.add_argument('--filter_no_finding', action='store_true',
                        help='Filter out "No Finding" samples before training')
    parser.add_argument('--device_id', type=int, default=None,
                        help='CUDA device ID to use for training (e.g., 0, 1). If not specified, defaults to cuda:0')

    # Paths and Model Defaults (Allow Overrides)
    parser.add_argument('--data_json', type=str, default=DEFAULT_DATA_JSON, help='Path to data JSON file')
    parser.add_argument('--image_root', type=str, default=DEFAULT_IMAGE_ROOT, help='Path to primary image root')
    parser.add_argument('--image_root_2', type=str, default=DEFAULT_IMAGE_ROOT_2, help='Path to secondary image root')
    parser.add_argument('--vision_model_name', type=str, default=DEFAULT_VISION_MODEL_NAME, help='Vision model name')
    parser.add_argument('--output_base_dir', type=str, default=DEFAULT_OUTPUT_BASE_DIR, help='Base directory for output checkpoints')

    # Hyperparameters (Allow Overrides)
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help='Image size')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for the head')
    parser.add_argument('--bb_lr', type=float, default=DEFAULT_BACKBONE_LEARNING_RATE, help='Learning rate for the backbone (if unfrozen)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='Number of dataloader workers')
    parser.add_argument('--dropout_rate', type=float, default=DEFAULT_DROPOUT_RATE, help='Dropout rate for the classifier head')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Weight decay for AdamW')

    args = parser.parse_args()

    # --- Derive specific config from args ---
    args.target_class_names = [name.strip() for name in args.class_names.split(',')]
    if args.handle_abnormal:
        if "No Finding" in args.target_class_names:
             args.effective_class_names = ["No Finding", "Abnormal"]
        else:
             logger.warning("Handling abnormal but 'No Finding' not in target classes. Using only 'Abnormal'.")
             args.effective_class_names = ["Abnormal"]
        args.abnormal_source_classes = [cls for cls in ALL_POSSIBLE_CLASSES if cls != "No Finding"]
    else:
        args.effective_class_names = args.target_class_names
        args.abnormal_source_classes = []

    if args.freeze_mode == 'Freeze':
        args.freeze_vision_encoder = True
        args.train_vision_encoder_first_epoch = False
    elif args.freeze_mode == 'Unfreeze':
        args.freeze_vision_encoder = False
        args.train_vision_encoder_first_epoch = False
    elif args.freeze_mode == '1EpochUnfreeze':
        args.freeze_vision_encoder = False
        args.train_vision_encoder_first_epoch = True

    args.output_dir = os.path.join(args.output_base_dir, args.exp_id)
    return args


# === Main Function ===
def main(args):
    """Orchestrates the training process."""
    # 2. Setup Environment (Device, Logging)
    device = setup_environment(args)

    # 3. Initialize Model and Processor
    model, processor = setup_model_processor(args)

    model.to(device) # Move model to device

    # 4. Load and Prepare Data (Load, Filter, Split, Dataloaders)
    train_loader, val_loader = load_and_prepare_data(args, processor)

    # 5. Setup Optimizer (handles initial freezing state)
    optimizer = setup_optimizer(model, args)

    # 6. Setup Loss Function
    criterion = nn.CrossEntropyLoss().to(device)
    logger.info("Using CrossEntropyLoss.")

    # 7. Run Training Loop (includes validation and checkpointing)
    run_training_loop(model, train_loader, val_loader, criterion, optimizer, device, args)

    # 8. Final Message
    logger.info(f"--- Experiment {args.exp_id} Finished ---")

# === Main Execution Block ===
if __name__ == '__main__':
    args = parse_args()
    main(args)
