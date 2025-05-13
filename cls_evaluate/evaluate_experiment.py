import torch
import torch.nn as nn
import torch.nn.functional as F # For softmax
from torch.utils.data import DataLoader
import os
import json
import logging
import argparse # Added
import glob # Added for finding checkpoints
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score # Added roc_auc_score
from sklearn.preprocessing import label_binarize # Added for one-hot encoding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import pandas as pd # Added for reading results.tsv

# --- Import necessary components from train.py ---
# Assuming train.py is in the same directory or Python path
try:
    # Use a relative import path assuming evaluate_experiment.py and train.py are in the same dir
    from .train import AbnormalityClassifier, XrayClassificationDataset, classification_collate_fn, parse_args as train_parse_args
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback to absolute import if relative fails (e.g., running script directly)
        from train import AbnormalityClassifier, XrayClassificationDataset, classification_collate_fn, parse_args as train_parse_args
    except (ImportError, ModuleNotFoundError):
        print("Error: Ensure train.py is in the same directory or accessible in the Python path.")
        print("Attempted relative import from .train and absolute import from train.")
        exit(1)

# === Configuration via Arguments ===
def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in an experiment directory.")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory containing checkpoints and results.tsv.')
    parser.add_argument('--test_json', type=str, required=True,
                        help='Path to the test JSON file (base file, will be filtered).')
    parser.add_argument('--eval_class_names', type=str, required=True,
                        help='Comma-separated string of class names for this evaluation task (used for filtering test_json).')
    # Allow overriding some params if not found in checkpoint args, but primarily rely on checkpoint
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--image_root', type=str, default=None, help='(Optional) Override image root 1')
    parser.add_argument('--image_root_2', type=str, default=None, help='(Optional) Override image root 2')
    parser.add_argument('--device_id', type=int, default=None, 
                        help='CUDA device ID to use for evaluation (e.g., 0, 1). If not specified, defaults to first available CUDA device.')

    args = parser.parse_args()
    return args

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Evaluation Function for a Single Checkpoint ===
def evaluate_checkpoint(checkpoint_path, test_json_path, target_eval_classes, eval_args):
    """Loads a model checkpoint and evaluates its performance (Accuracy, AUC) on filtered data."""
    logger.info(f"Evaluating checkpoint: {os.path.basename(checkpoint_path)}")
    logger.info(f"  Target evaluation classes: {target_eval_classes}")
    # Load to CPU first to potentially avoid OOM during load itself
    device_load = torch.device("cpu") 
    
    # Use specified device ID if provided, otherwise use first available CUDA device
    if hasattr(eval_args, 'device_id') and eval_args.device_id is not None:
        device_str = f"cuda:{eval_args.device_id}" if torch.cuda.is_available() else "cpu"
        device_eval = torch.device(device_str)
        if torch.cuda.is_available():
            logger.info(f"Using specified CUDA device: {device_str}")
    else:
        device_eval = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"No device_id specified, defaulting to: {device_eval}")

    model = None # Ensure model is defined in outer scope for finally block
    checkpoint = None

    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint file not found!")
        return None
    if not os.path.exists(test_json_path):
        logger.error(f"Test JSON file not found: {test_json_path}")
        return None

    try:
        # Load checkpoint to CPU
        checkpoint = torch.load(checkpoint_path, map_location=device_load, weights_only=False)
        train_args_dict = checkpoint.get('args')
        if not train_args_dict:
            logger.error(f"Checkpoint {checkpoint_path} does not contain saved 'args'. Cannot infer parameters.")
            return None

        # --- Extract args from checkpoint --- #
        ckpt_class_names = train_args_dict.get('effective_class_names')
        ckpt_vision_model = train_args_dict.get('vision_model_name')
        ckpt_img_size = train_args_dict.get('img_size')
        ckpt_handle_abnormal = train_args_dict.get('handle_abnormal', False)
        ckpt_abnormal_source = train_args_dict.get('abnormal_source_classes', [])
        ckpt_epoch = checkpoint.get('epoch', -1) # Get epoch number
        ckpt_image_root = train_args_dict.get('image_root')
        ckpt_image_root_2 = train_args_dict.get('image_root_2')
        image_root_to_use = eval_args.image_root or ckpt_image_root
        image_root_2_to_use = eval_args.image_root_2 or ckpt_image_root_2

        if not all([ckpt_class_names, ckpt_vision_model, ckpt_img_size, image_root_to_use]):
             logger.error(f"Missing critical args (class_names, vision_model, img_size, image_root) in checkpoint {checkpoint_path} or eval args.")
             return None
        logger.info(f"  Checkpoint Params: Classes={ckpt_class_names}, Model={ckpt_vision_model}, ImgSize={ckpt_img_size}, HandleAbnormal={ckpt_handle_abnormal}")

        # --- Initialize Model on CPU --- #
        model = AbnormalityClassifier(
            vision_model_name=ckpt_vision_model,
            class_names=ckpt_class_names,
        )

        # --- Load Model Weights (still on CPU) --- #
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # --- Move model to evaluation device --- #
        model.to(device_eval)
        model.eval()

        # --- Load Processor --- #
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(ckpt_vision_model)

        # --- Prepare Test Data (Load, Filter based on target_eval_classes) --- #
        with open(test_json_path, 'r', encoding='utf-8') as f:
            all_test_samples = json.load(f)

        # Filter samples based on target_eval_classes, considering handle_abnormal from checkpoint
        temp_class_to_idx_ckpt = {name: i for i, name in enumerate(ckpt_class_names)} # mapping used by checkpoint
        relevant_test_samples = []
        # No need for test_labels_for_eval here, dataset handles it
        logger.info(f"Filtering test samples from {test_json_path} for classes: {target_eval_classes}")
        for sample in all_test_samples:
            original_label = sample.get("normal_caption", "").strip()
            effective_label_for_eval = original_label
            # Determine the label as the *checkpoint* model sees it (for handle_abnormal mapping)
            label_checkpoint_sees = original_label
            if ckpt_handle_abnormal and original_label in ckpt_abnormal_source:
                label_checkpoint_sees = "Abnormal"
            
            # Check if this effective label is one of the classes we *want* to evaluate in this run
            if label_checkpoint_sees in target_eval_classes:
                # Check if the checkpoint model actually knows about this class
                label_idx_ckpt = temp_class_to_idx_ckpt.get(label_checkpoint_sees, -1)
                if label_idx_ckpt != -1:
                    relevant_test_samples.append(sample) # Keep the original sample


        if not relevant_test_samples:
            logger.warning(f"No relevant test samples found after filtering for classes {target_eval_classes} based on checkpoint's classes {ckpt_class_names}. Skipping eval.")
            return None

        logger.info(f"  Evaluating on {len(relevant_test_samples)} filtered test samples.")

        # Create dataset using the *checkpoint's* parameters and the filtered samples
        test_dataset = XrayClassificationDataset(
            samples=relevant_test_samples,
            image_root=image_root_to_use,
            class_names=ckpt_class_names, # IMPORTANT: Use checkpoint's classes for dataset init
            processor=processor,
            img_size=ckpt_img_size,
            image_root_2=image_root_2_to_use,
            handle_abnormal=ckpt_handle_abnormal,
            abnormal_source_classes=ckpt_abnormal_source
        )
        if len(test_dataset) != len(relevant_test_samples):
             logger.error("Dataset length mismatch after filtering. Check XrayClassificationDataset logic.")
             return None
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_args.batch_size,
            shuffle=False,
            num_workers=eval_args.num_workers,
            pin_memory=True,
            collate_fn=classification_collate_fn
        )

        # --- Evaluation Loop --- #
        all_target_indices = []
        all_pred_indices = []
        all_pred_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval Epoch {ckpt_epoch}", leave=False):
                if batch is None: continue
                # Move batch data to evaluation device
                pixel_values = batch["pixel_values"].to(device_eval)
                target_indices_batch = batch["target_indices"].cpu().numpy()
                logits = model(pixel_values)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                pred_indices = np.argmax(probs, axis=1)
                all_target_indices.append(target_indices_batch)
                all_pred_indices.append(pred_indices)
                all_pred_probs.append(probs)

        if not all_target_indices:
            logger.warning("No valid batches processed for checkpoint. Cannot calculate metrics.")
            return None
        all_target_indices = np.concatenate(all_target_indices, axis=0)
        all_pred_indices = np.concatenate(all_pred_indices, axis=0)
        all_pred_probs = np.concatenate(all_pred_probs, axis=0)

        # --- Calculate Metrics --- #
        accuracy = accuracy_score(all_target_indices, all_pred_indices)
        num_classes = len(ckpt_class_names)
        auc_score = None
        if num_classes > 1 and len(np.unique(all_target_indices)) > 1:
            try:
                all_targets_one_hot = label_binarize(all_target_indices, classes=range(num_classes))
                if all_targets_one_hot.shape[1] == 1:
                     all_targets_one_hot = np.hstack((1-all_targets_one_hot, all_targets_one_hot))
                if all_pred_probs.shape[1] != all_targets_one_hot.shape[1]:
                     logger.warning(f"Shape mismatch for AUC: Probs {all_pred_probs.shape}, Targets {all_targets_one_hot.shape}. Skipping AUC.")
                else:
                     auc_score = roc_auc_score(all_targets_one_hot, all_pred_probs, average="macro", multi_class="ovr")
            except ValueError as e:
                 logger.warning(f"Could not calculate AUC: {e}. Maybe only one class present in targets?")
            except Exception as e:
                 logger.error(f"Unexpected error calculating AUC: {e}", exc_info=True)
        elif num_classes <= 1:
             logger.warning("Skipping AUC calculation because num_classes <= 1")
        else:
             logger.warning(f"Skipping AUC calculation because only one class present in targets for checkpoint {os.path.basename(checkpoint_path)}. ")

        logger.info(f"  Epoch {ckpt_epoch} - Accuracy: {accuracy:.4f}, AUC: {auc_score if auc_score is not None else 'N/A'}")

        return {
            'epoch': ckpt_epoch,
            'accuracy': accuracy,
            'auc': auc_score,
            'checkpoint_path': checkpoint_path
        }

    except torch.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory during evaluation of {checkpoint_path}: {e}")
        # Try to clear cache and continue if possible, otherwise return None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None # Mark this checkpoint as failed
    except Exception as e:
        logger.error(f"Failed to evaluate checkpoint {checkpoint_path}: {e}", exc_info=True)
        return None
    finally:
        # --- Explicit Cleanup --- #
        del checkpoint # Remove reference to loaded data
        del model # Remove reference to model instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear CUDA cache

# === Main Execution Block ===
if __name__ == '__main__':
    eval_args = parse_eval_args()

    # Parse the target classes for this specific evaluation run
    target_eval_classes = [name.strip() for name in eval_args.eval_class_names.split(',') if name.strip()]
    if not target_eval_classes:
        logger.error("Evaluation class names (--eval_class_names) cannot be empty.")
        exit(1)
    logger.info(f"Starting evaluation in {eval_args.exp_dir}")
    logger.info(f"Target classes for this evaluation: {target_eval_classes}")
    logger.info(f"Loading test data from: {eval_args.test_json}")

    # Find all checkpoints in the experiment directory
    checkpoint_files = sorted(glob.glob(os.path.join(eval_args.exp_dir, '*.pth')))
    if not checkpoint_files:
        logger.error(f"No checkpoint files ('*.pth') found in {eval_args.exp_dir}")
        exit(1)

    logger.info(f"Found {len(checkpoint_files)} checkpoints to evaluate.")

    all_results = []
    for ckpt_path in checkpoint_files:
        # Pass the target classes needed for filtering to the evaluation function
        result = evaluate_checkpoint(ckpt_path, eval_args.test_json, target_eval_classes, eval_args)
        if result:
            all_results.append(result)

    if not all_results:
        logger.error("No checkpoints could be successfully evaluated.")
        exit(1)

    # --- Determine the best checkpoint based on validation metrics from training --- #
    # Assumes a results.tsv file was saved during training in eval_args.exp_dir
    results_tsv_path = os.path.join(eval_args.exp_dir, 'results.tsv')
    best_result = None
    best_val_metric = -1.0 # Initialize with a low value
    metric_to_optimize = 'Val AUC' # Prioritize AUC
    fallback_metric = 'Val Accuracy'

    if os.path.exists(results_tsv_path):
        try:
            train_results_df = pd.read_csv(results_tsv_path, sep='\t')
            logger.info(f"Read training results from {results_tsv_path}")

            # Check if primary metric exists
            if metric_to_optimize not in train_results_df.columns:
                logger.warning(f"'{metric_to_optimize}' not found in {results_tsv_path}. Trying '{fallback_metric}'.")
                metric_to_optimize = fallback_metric

            if metric_to_optimize in train_results_df.columns:
                 # Find the row with the highest validation metric
                best_train_epoch_row = train_results_df.loc[train_results_df[metric_to_optimize].idxmax()]
                best_train_epoch = best_train_epoch_row['Epoch']
                best_val_metric_value = best_train_epoch_row[metric_to_optimize]
                logger.info(f"Best validation performance found at Epoch {best_train_epoch} ({metric_to_optimize}={best_val_metric_value:.4f}) based on {results_tsv_path}")

                # Find the corresponding evaluation result for that epoch
                for res in all_results:
                    if res['epoch'] == best_train_epoch:
                        best_result = res
                        break
                if best_result:
                     logger.info(f"Found matching evaluation result for best epoch {best_train_epoch}.")
                else:
                     logger.warning(f"Could not find evaluation result for the best training epoch {best_train_epoch}. This might happen if the checkpoint for that epoch failed evaluation or was missing.")
            else:
                logger.warning(f"Neither 'Val AUC' nor 'Val Accuracy' found in {results_tsv_path}. Cannot determine best epoch from training logs.")

        except Exception as e:
            logger.error(f"Error reading or processing {results_tsv_path}: {e}. Cannot determine best epoch from training logs.")

    # If best epoch couldn't be determined from results.tsv, fall back to best *evaluated* checkpoint on the test set
    if best_result is None:
        logger.warning("Falling back to selecting the best checkpoint based on *test* set performance (highest Accuracy, then AUC).")
        # Sort by Accuracy (descending), then AUC (descending, handle None) as a tie-breaker
        all_results.sort(key=lambda x: (x.get('accuracy', -1), x.get('auc', -1) if x.get('auc') is not None else -1), reverse=True)
        if all_results:
            best_result = all_results[0]
            logger.info(f"Selected best checkpoint based on test performance: Epoch {best_result['epoch']} (Acc: {best_result['accuracy']:.4f}, AUC: {best_result.get('auc', 'N/A') if best_result.get('auc') is not None else 'N/A'})")

    # --- Print Final Best Result --- #
    if best_result:
        best_epoch = best_result['epoch']
        best_acc = best_result['accuracy']
        best_auc = best_result.get('auc') # May be None
        best_ckpt_path = best_result['checkpoint_path']
        # Format the final output line for the shell script
        print(f"BEST_RESULT\t{best_epoch}\t{best_acc:.4f}\t{best_auc if best_auc is not None else 'NaN'}\t{best_ckpt_path}")
        logger.info(f"Best Checkpoint Path: {best_ckpt_path}")
    else:
        logger.error("Could not determine the best checkpoint after evaluating all available checkpoints.")
        # Optionally print a placeholder error line for the TSV file
        print("BEST_RESULT\tERROR\tERROR\tERROR\tERROR")
        exit(1)

    # --- Process and Save Results --- #
    if all_results:
        # Sort results by epoch for plotting
        all_results.sort(key=lambda x: x['epoch']) # Ensure epoch is present and numeric

        # Find best result based on accuracy
        best_result = max(all_results, key=lambda x: x['accuracy']) # Find max accuracy
        # Corrected f-string for logging:
        auc_str_log = f"{best_result['auc']:.4f}" if best_result['auc'] is not None else 'N/A'
        logger.info(f"Best Test Accuracy: {best_result['accuracy']:.4f} (Epoch {best_result['epoch']}, AUC: {auc_str_log}) from {os.path.basename(best_result['checkpoint_path'])}")

        # --- Plotting Results --- #
        logger.info("Generating performance vs. epoch plot...")
        epochs = [r['epoch'] for r in all_results if r['epoch'] != -1] # Filter out invalid epochs
        accuracies = [r['accuracy'] for r in all_results if r['epoch'] != -1]
        # Handle None AUCs gracefully for plotting
        aucs = [r['auc'] if r['auc'] is not None else np.nan for r in all_results if r['epoch'] != -1]

        if not epochs:
             logger.warning("No valid epochs found in results, cannot generate plot.")
        else:
             plt.figure(figsize=(12, 7))
             
             # Plot Accuracy
             plt.plot(epochs, accuracies, marker='o', linestyle='-', label='Test Accuracy')
             
             # Plot AUC if available
             valid_auc_indices = [i for i, auc_val in enumerate(aucs) if not np.isnan(auc_val)]
             if valid_auc_indices:
                  plt.plot([epochs[i] for i in valid_auc_indices], [aucs[i] for i in valid_auc_indices], marker='s', linestyle='--', label='Test AUC (Macro OVR')
             else:
                  logger.info("No valid AUC scores to plot.")

             plt.xlabel("Epoch")
             plt.ylabel("Metric Value")
             plt.title(f"Test Performance vs. Epoch\nExp: {os.path.basename(eval_args.exp_dir)}")
             plt.xticks(range(min(epochs), max(epochs)+1)) # Show integer epochs
             plt.legend()
             plt.grid(True)
             plt.ylim(bottom=0) # Metrics like Acc/AUC are >= 0
             plt.tight_layout()

             # Save the plot
             plot_filename = os.path.join(eval_args.exp_dir, "performance_vs_epoch.png")
             try:
                 plt.savefig(plot_filename)
                 logger.info(f"Performance plot saved to {plot_filename}")
             except Exception as e:
                 logger.error(f"Failed to save performance plot: {e}")
             plt.close()

        # --- Delete all checkpoints after successful evaluation --- #
        logger.info(f"Evaluation successful. Skipping checkpoint cleanup (cleanup disabled)...")


    else:
        logger.warning("No results collected from checkpoint evaluations. Cannot determine best result or plot.")

    logger.info(f"Evaluation script finished for {eval_args.exp_dir}.") 