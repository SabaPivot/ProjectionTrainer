# soombit/train_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for softmax in AUC calculation
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import os
import json
import logging
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score # Added roc_auc_score
from sklearn.preprocessing import label_binarize # Added for one-hot encoding
import glob

# Need to import classes/functions defined in models.py and train.py
try:
    # Import classes from models.py
    from .models import AbnormalityClassifier, XrayClassificationDataset, classification_collate_fn
    # Import parse_args from train.py (if needed, though likely not needed in utils)
    # from .train import parse_args
except ImportError:
    try:
        from models import AbnormalityClassifier, XrayClassificationDataset, classification_collate_fn
        # from train import parse_args
    except ImportError as e:
        print(f"ERROR in train_utils.py: Could not import dependencies. Error: {e}")
        pass

logger = logging.getLogger(__name__)

# === Evaluation Function ===
def evaluate(model, data_loader, criterion, device, class_names):
    """Evaluates the model on the given data loader and returns accuracy, loss, and AUC."""
    model.eval()
    total_loss = 0.0
    all_target_indices = []
    all_pred_indices = []
    all_pred_probs = [] # Added to store probabilities for AUC
    num_classes = len(class_names)

    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            pixel_values = batch["pixel_values"].to(device)
            target_indices = batch["target_indices"].to(device)
            logits = model(pixel_values)
            loss = criterion(logits, target_indices)
            total_loss += loss.item()
            
            # Store predictions and targets
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
            all_target_indices.append(target_indices.cpu().numpy())
            all_pred_indices.append(pred_indices)
            all_pred_probs.append(probs) # Store probabilities

    if not all_target_indices: # Handle empty dataloader case
        logger.warning("Evaluation dataloader was empty or yielded no valid batches.")
        return 0.0, 0.0, None # Return None for AUC

    all_target_indices = np.concatenate(all_target_indices)
    all_pred_indices = np.concatenate(all_pred_indices)
    all_pred_probs = np.concatenate(all_pred_probs)

    num_samples = len(all_target_indices)
    correct_predictions = np.sum(all_pred_indices == all_target_indices)
    accuracy = correct_predictions / num_samples if num_samples > 0 else 0.0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    # --- Calculate AUC --- #
    auc_score = None
    if num_classes > 1 and len(np.unique(all_target_indices)) > 1:
        try:
            all_targets_one_hot = label_binarize(all_target_indices, classes=range(num_classes))
            if all_targets_one_hot.shape[1] == 1: # Handle binary case returned by label_binarize
                all_targets_one_hot = np.hstack((1 - all_targets_one_hot, all_targets_one_hot))
            
            if all_pred_probs.shape[1] != all_targets_one_hot.shape[1]:
                 logger.warning(f"AUC Calc: Shape mismatch Probs {all_pred_probs.shape}, Targets {all_targets_one_hot.shape}. Skipping AUC.")
            else:
                 auc_score = roc_auc_score(all_targets_one_hot, all_pred_probs, average="macro", multi_class="ovr")
        except ValueError as e:
            logger.warning(f"Could not calculate Val AUC: {e}. Maybe only one class present?")
        except Exception as e:
            logger.error(f"Unexpected error calculating Val AUC: {e}", exc_info=True)
    elif num_classes <= 1:
        logger.warning("Skipping Val AUC calculation because num_classes <= 1")
    else:
        logger.warning("Skipping Val AUC calculation because only one class present in validation targets.")
    # ------------------ #

    return accuracy, avg_loss, auc_score # Return AUC as well

# === Helper Functions Moved from Main Script ===

def setup_environment(args):
    """Sets up device and logs initial configuration."""
    # Use specified device ID if provided, otherwise use cuda:0
    if hasattr(args, 'device_id') and args.device_id is not None:
        device_str = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        if torch.cuda.is_available():
            logger.info(f"Using specified CUDA device: {device_str}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"No device_id specified, defaulting to: {device}")
    
    logger.info(f"--- Experiment: {args.exp_id} ---")
    logger.info(f"Using device: {device}")
    logger.info(f"Effective Class Names: {args.effective_class_names}")
    logger.info(f"Freeze Mode: {args.freeze_mode} (VE Frozen: {args.freeze_vision_encoder}, Train VE 1st Epoch: {args.train_vision_encoder_first_epoch})")
    logger.info(f"Handle Abnormal: {args.handle_abnormal}")
    logger.info(f"Filter No Finding: {args.filter_no_finding}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Learning Rate (Head): {args.lr}, (Backbone): {args.bb_lr}")
    logger.info(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"Weight Decay: {args.weight_decay}, Dropout: {args.dropout_rate}")
    return device

def setup_model_processor(args):
    """Initializes the model and processor."""
    # Ensure AbnormalityClassifier is available (either via import or defined locally if import fails)
    if 'AbnormalityClassifier' not in globals():
        raise NameError("AbnormalityClassifier not imported or defined.")

    model = AbnormalityClassifier(
        vision_model_name=args.vision_model_name,
        class_names=args.effective_class_names,
        dropout_rate=args.dropout_rate
    )
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    return model, processor

def load_and_prepare_data(args, processor):
    """Loads JSON, filters, splits, and creates DataLoaders."""
    if 'XrayClassificationDataset' not in globals() or 'classification_collate_fn' not in globals():
         raise NameError("XrayClassificationDataset or classification_collate_fn not imported or defined.")

    logger.info(f"Loading data from {args.data_json}...")
    try:
        with open(args.data_json, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        logger.info(f"Loaded {len(all_samples)} total samples initially.")
    except Exception as e:
        logger.error(f"Error loading data JSON {args.data_json}: {e}", exc_info=True)
        exit(1)

    if args.filter_no_finding:
        original_count = len(all_samples)
        all_samples = [s for s in all_samples if s.get("normal_caption", "").strip() != "No Finding"]
        filtered_count = len(all_samples)
        logger.info(f"Filtered out 'No Finding' samples. Kept {filtered_count}/{original_count} samples.")
        if filtered_count == 0:
            logger.error("No samples remaining after filtering 'No Finding'. Exiting.")
            exit(1)
        if "No Finding" in args.target_class_names:
             logger.warning("Filtering 'No Finding' but it was included in --class_names. Ensure this is intended.")

    temp_class_to_idx = {name: i for i, name in enumerate(args.effective_class_names)}
    labels = []
    samples_for_split = []
    for sample in all_samples:
        original_label = sample.get("normal_caption", "").strip()
        target_label = original_label
        if args.handle_abnormal and original_label in args.abnormal_source_classes:
            target_label = "Abnormal"
        label_idx = temp_class_to_idx.get(target_label, -1)
        if label_idx != -1:
            labels.append(label_idx)
            samples_for_split.append(sample)

    if len(samples_for_split) < len(all_samples):
        logger.info(f"Keeping {len(samples_for_split)} samples relevant to effective classes: {args.effective_class_names}")
    if len(samples_for_split) < 2:
        logger.error(f"Not enough relevant samples ({len(samples_for_split)}) for effective classes {args.effective_class_names} to perform train/validation split.")
        exit(1)

    try:
        train_samples, val_samples = train_test_split(
            samples_for_split, labels, test_size=0.1, random_state=42, stratify=labels
        )
        logger.info(f"Split data into {len(train_samples)} training and {len(val_samples)} validation samples (stratified).")
    except ValueError as e:
        logger.warning(f"Could not stratify split: {e}. Performing random split.")
        train_samples, val_samples = train_test_split(
            samples_for_split, test_size=0.1, random_state=42
        )
        logger.info(f"Split data randomly into {len(train_samples)} training and {len(val_samples)} validation samples.")

    train_dataset = XrayClassificationDataset(
        samples=train_samples, image_root=args.image_root, class_names=args.effective_class_names,
        processor=processor, img_size=args.img_size, image_root_2=args.image_root_2,
        handle_abnormal=args.handle_abnormal, abnormal_source_classes=args.abnormal_source_classes
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=classification_collate_fn
    )

    val_loader = None
    if val_samples:
        val_dataset = XrayClassificationDataset(
            samples=val_samples, image_root=args.image_root, class_names=args.effective_class_names,
            processor=processor, img_size=args.img_size, image_root_2=args.image_root_2,
            handle_abnormal=args.handle_abnormal, abnormal_source_classes=args.abnormal_source_classes
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=classification_collate_fn
        )
        logger.info(f"Loaded {len(val_dataset)} validation samples.")
    else:
        logger.warning("Validation set is empty after splitting.")

    return train_loader, val_loader

def setup_optimizer(model, args):
    """Configures the optimizer with parameter groups."""
    if args.freeze_vision_encoder:
        model.vision_model.requires_grad_(False)
    else:
        model.vision_model.requires_grad_(True)
    # Ensure head components are always trainable
    model.abnormality_queries.requires_grad_(True)
    model.mha.requires_grad_(True)
    model.classification_head.requires_grad_(True)

    head_params = ([model.abnormality_queries] +
                   list(model.mha.parameters()) +
                   list(model.classification_head.parameters()))
    num_head_params = sum(p.numel() for p in head_params if p.requires_grad)

    if args.freeze_vision_encoder:
        logger.info("Optimizer: Vision Encoder FROZEN. Only head parameters included.")
        optimizer_params = [{"params": head_params, "lr": args.lr}]
        total_trainable_params = num_head_params
    else:
        logger.info(f"Optimizer: Vision Encoder TRAINABLE (initially). Using discriminative LRs.")
        backbone_params = list(model.vision_model.parameters())
        num_backbone_params = sum(p.numel() for p in backbone_params if p.requires_grad)
        optimizer_params = [
            {"params": backbone_params, "lr": args.bb_lr},
            {"params": head_params, "lr": args.lr}
        ]
        total_trainable_params = num_head_params + num_backbone_params
        logger.info(f"  - Vision Backbone: {num_backbone_params} params (LR={args.bb_lr})")

    logger.info(f"  - Head: {num_head_params} params (LR={args.lr})")
    logger.info(f"Total Trainable Parameters (initial): {total_trainable_params}")

    actual_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_trainable_params != actual_total_params:
        logger.warning(f"Parameter count mismatch! Calculated: {total_trainable_params}, Actual requires_grad: {actual_total_params}")

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)
    logger.info(f"Using AdamW optimizer with Weight Decay={args.weight_decay}")
    return optimizer

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, device, args):
    """Executes the main training loop, including VE freezing, eval, and checkpointing."""
    logger.info("Starting training loop...")
    best_val_accuracy = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {os.path.abspath(args.output_dir)}")

    # --- Initialize results.tsv --- #
    results_tsv_path = os.path.join(args.output_dir, 'results.tsv')
    try:
        # Write header only if file doesn't exist
        if not os.path.exists(results_tsv_path):
             with open(results_tsv_path, 'w', encoding='utf-8') as f_tsv:
                 f_tsv.write("Epoch\tTrain Loss\tVal Loss\tVal Accuracy\tVal AUC\n")
             logger.info(f"Created results file: {results_tsv_path}")
        else:
             logger.info(f"Appending results to existing file: {results_tsv_path}")
    except IOError as e:
         logger.error(f"Could not create or open results file {results_tsv_path}: {e}. Metrics will not be saved to TSV.")
         results_tsv_path = None # Disable saving if file cannot be opened
    # ----------------------------- #

    for epoch in range(args.epochs):
        model.train()

        # Dynamic Vision Encoder Freezing/Unfreezing
        if not args.freeze_vision_encoder:
            is_first_epoch = (epoch == 0)
            if args.train_vision_encoder_first_epoch:
                if is_first_epoch:
                    logger.info(f"Epoch {epoch+1}: Training Vision Encoder (first epoch only).")
                    model.vision_model.requires_grad_(True)
                    model.vision_model.train()
                else:
                    if model.vision_model.training: # Check if it was training
                         if epoch == 1: logger.info(f"Epoch {epoch+1}: Freezing Vision Encoder for subsequent epochs.")
                         model.vision_model.requires_grad_(False)
                         model.vision_model.eval()
            else: # Unfreeze mode (train VE throughout)
                 if epoch == 0: logger.info(f"Epoch {epoch+1}: Vision Encoder is trainable throughout.")
                 if not model.vision_model.training: # Ensure it's in train mode
                      model.vision_model.requires_grad_(True)
                      model.vision_model.train()
        else: # Freeze mode
             if epoch == 0: logger.info("Epoch 1: Vision encoder is permanently frozen.")
             if model.vision_model.training: # Ensure it's in eval mode
                  model.vision_model.requires_grad_(False)
                  model.vision_model.eval()

        epoch_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        # Training step
        for batch in train_progress_bar:
            if batch is None: continue
            pixel_values = batch["pixel_values"].to(device)
            target_indices = batch["target_indices"].to(device)

            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, target_indices)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Average Training Loss: {avg_epoch_train_loss:.4f}")

        # Validation step
        val_accuracy = 0.0
        avg_epoch_val_loss = float('inf')
        val_auc = None # Initialize val_auc for the epoch
        if val_loader:
            # Get accuracy, loss, and AUC from evaluate function
            val_accuracy, avg_epoch_val_loss, val_auc = evaluate(model, val_loader, criterion, device, args.effective_class_names)
            val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_epoch_val_loss:.4f}, Validation AUC: {val_auc_str}")

            # Save best checkpoint based on validation accuracy
            if val_accuracy > best_val_accuracy: # Prioritize accuracy
                best_val_accuracy = val_accuracy # Update best accuracy value
                # Find and remove previous best model checkpoint(s) to keep only the latest best
                previous_best_files = glob.glob(os.path.join(args.output_dir, "best_model_epoch_*.pth"))
                for f_path in previous_best_files:
                    try:
                        os.remove(f_path)
                        logger.debug(f"Removed previous best checkpoint: {f_path}")
                    except OSError as e:
                        logger.warning(f"Error removing previous best checkpoint {f_path}: {e}")

                # Save the new best checkpoint
                save_path = os.path.join(args.output_dir, f"best_model_epoch_{epoch+1}.pth")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_val_loss,
                    'accuracy': val_accuracy,
                    'auc': val_auc,
                    'class_names': args.effective_class_names,
                    'args': vars(args)
                }
                torch.save(checkpoint, save_path)
                logger.info(f"Saved new best model checkpoint to {save_path} (Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc_str})")
        else:
             logger.info(f"Epoch {epoch+1}/{args.epochs} - Skipping validation loop.")

        # --- Append results to TSV --- #
        if results_tsv_path:
             try:
                 with open(results_tsv_path, 'a', encoding='utf-8') as f_tsv:
                     # Format NaN for AUC if it's None
                     auc_str_tsv = f"{val_auc:.6f}" if val_auc is not None else "NaN"
                     f_tsv.write(f"{epoch+1}\t{avg_epoch_train_loss:.6f}\t{avg_epoch_val_loss:.6f}\t{val_accuracy:.6f}\t{auc_str_tsv}\n")
             except IOError as e:
                  logger.error(f"Could not write to results file {results_tsv_path}: {e}")
        # ----------------------------- #

        # Save periodic checkpoint
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            periodic_save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            periodic_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_train_loss,
                'val_loss': avg_epoch_val_loss,
                'val_accuracy': val_accuracy,
                'val_auc': val_auc,
                'class_names': args.effective_class_names,
                'args': vars(args)
            }
            torch.save(periodic_checkpoint, periodic_save_path)
            logger.info(f"Saved periodic checkpoint to {periodic_save_path} (Epoch {epoch+1})")

    logger.info(f"Training loop finished for experiment {args.exp_id}.") 