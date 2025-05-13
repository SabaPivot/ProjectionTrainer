import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib

# Import necessary components from train.py
try:
    from train import AbnormalityClassifier, XrayClassificationDataset, classification_collate_fn
except ImportError:
    print("Error: Ensure train.py is in the same directory or accessible in the Python path.")
    exit()

# --- Configuration --- (MUST BE SET BY USER) ---
# *** USER ACTION REQUIRED: Set the correct path to the model chefrom sklearn.metrics import classification_report, roc_auc_score, accuracy_scoreckpoint ***
CHECKPOINT_PATH = "/mnt/samuel/Siglip/soombit/soombit/checkpoints/best_model_epoch_5.pth" # e.g., './best_model_epoch_8.pth' or './checkpoint_epoch_10.pth'
# *** USER ACTION REQUIRED: Set the correct name of the test JSON file ***
TEST_JSON_FILENAME = "filtered_Atelectasis_Cardiomegaly_Effusion_No_Finding.json" # Assuming a file named 'test.json' exists in the directory
TEST_JSON_DIR = "/mnt/samuel/Siglip/balanced_samples"
IMAGE_ROOT = "/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2 = "/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
# CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion"] # Must match training - REMOVED: Will be loaded from checkpoint
VISION_MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" # Must match training
IMG_SIZE = 384
BATCH_SIZE = 32 # Adjust based on GPU memory for inference
NUM_WORKERS = 4
PREDICTION_THRESHOLD = 0.5 # Threshold for converting probabilities to binary predictions
# --------------------------------------------------

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(checkpoint_path, test_json_path, image_root, image_root_2, vision_model_name, img_size, batch_size, num_workers, threshold):
    """Loads a model checkpoint and evaluates it on the test set.

    Returns:
        tuple: (overall_accuracy, macro_auc) or (None, None) if evaluation fails.
    """

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    if not os.path.exists(test_json_path):
        logger.error(f"Test JSON file not found: {test_json_path}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Checkpoint First to Get Parameters --- 
    logger.info(f"Loading checkpoint to extract model parameters: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Extract parameters from checkpoint
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            # Get class names from checkpoint
            if 'effective_class_names' in checkpoint_args:
                class_names = checkpoint_args['effective_class_names']
                logger.info(f"Using class names from checkpoint: {class_names}")
            else:
                logger.error("Checkpoint doesn't contain 'effective_class_names'. Cannot evaluate.")
                return None, None
                
            # Extract handle_abnormal flag and abnormal_source_classes
            handle_abnormal = checkpoint_args.get('handle_abnormal', False)
            abnormal_source_classes = checkpoint_args.get('abnormal_source_classes', [])
            logger.info(f"Using handle_abnormal={handle_abnormal} from checkpoint")
            if handle_abnormal:
                logger.info(f"Abnormal source classes: {abnormal_source_classes}")
                
            # Optionally get vision model name from checkpoint if available
            if 'vision_model_name' in checkpoint_args:
                vision_model_name = checkpoint_args['vision_model_name']
                logger.info(f"Using vision model name from checkpoint: {vision_model_name}")
        else:
            logger.error("Checkpoint doesn't contain 'args'. Cannot extract parameters.")
            return None, None
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return None, None

    # --- Load Model Architecture ---
    logger.info(f"Loading model architecture ({vision_model_name})...")
    model = AbnormalityClassifier(
        vision_model_name=vision_model_name,
        class_names=class_names, # Use class names from checkpoint
    ).to(device)

    # --- Load Model Weights ---
    logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
    try:
        # Checkpoint is already loaded above, just apply state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded state dict from epoch {checkpoint.get('epoch', 'N/A')}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded state dict (from 'state_dict' key).")
        else:
            # Assume the checkpoint file *is* the state_dict
            model.load_state_dict(checkpoint)
            logger.info("Loaded state dict directly from file.")
        model.eval() # Set model to evaluation mode
    except Exception as e:
        logger.error(f"Error loading model weights: {e}", exc_info=True)
        return None, None

    # --- Load Processor ---
    # Need processor for the dataset
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(vision_model_name)
    except ImportError:
         logger.error("transformers library not found. Please install it.")
         return None, None
    except Exception as e:
         logger.error(f"Error loading processor for {vision_model_name}: {e}")
         return None, None


    # --- Prepare Test Data ---
    logger.info(f"Loading test data from: {test_json_path}")
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        logger.info(f"Loaded {len(test_samples)} samples from {test_json_path}.")
    except FileNotFoundError:
        logger.error(f"Test JSON file not found at {test_json_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading test JSON {test_json_path}: {e}")
        return None, None

    test_dataset = XrayClassificationDataset(
        samples=test_samples, # Pass the loaded samples list
        image_root=image_root,
        class_names=class_names,
        processor=processor,
        img_size=img_size,
        image_root_2=image_root_2,
        handle_abnormal=handle_abnormal,
        abnormal_source_classes=abnormal_source_classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No shuffling for test set
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=classification_collate_fn
    )
    logger.info(f"Loaded {len(test_dataset)} test samples.")

    # --- Evaluation Loop ---
    all_single_target_indices = [] # Store the true single target index
    all_preds_probs = []
    all_preds_single_indices = [] # Store the predicted single target index

    logger.info("Starting evaluation...")
    # Create mapping for test evaluation
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            if batch is None: continue # Skip potentially empty batches from collation errors

            pixel_values = batch["pixel_values"].to(device)
            # Get target indices directly from batch (key defined in train.py's collate_fn)
            target_indices_batch = batch["target_indices"].cpu().numpy()

            # Get model outputs (logits)
            logits = model(pixel_values)

            # Calculate probabilities
            # For CrossEntropyLoss, logits are direct outputs. Use Softmax for probabilities if needed,
            # but only argmax is needed for the single label accuracy.
            probs = torch.softmax(logits, dim=1).cpu() # Use softmax for probabilities

            # === Single Target Prediction ===
            # Get the index of the highest probability class
            preds_single_indices = torch.argmax(logits, dim=1).cpu().numpy() # Argmax on logits is fine

            # Store batch results
            all_single_target_indices.append(target_indices_batch)
            all_preds_probs.append(probs)
            all_preds_single_indices.append(preds_single_indices)

    # Concatenate results from all batches
    if not all_single_target_indices:
        logger.error("No valid batches were processed. Cannot calculate metrics.")
        return None, None

    all_single_target_indices = np.concatenate(all_single_target_indices, axis=0)
    all_preds_probs = torch.cat(all_preds_probs, dim=0).numpy()
    all_preds_single_indices = np.concatenate(all_preds_single_indices, axis=0)

    logger.info("Evaluation finished. Calculating metrics...")

    # --- Calculate and Print Metrics ---
    macro_auc = None # Initialize macro_auc
    try:
        # --- NEW: Single Accuracy Calculation ---
        # Compare predicted index with true single target index
        # Exclude samples where the true single target index is -1 (could not be determined)
        valid_indices = (all_single_target_indices != -1)
        if np.sum(valid_indices) == 0:
            logger.warning("Could not determine any valid single target labels from the dataset. Cannot calculate Single Accuracy.")
            single_accuracy = 0.0
        else:
            correct_single_preds = (all_preds_single_indices[valid_indices] == all_single_target_indices[valid_indices])
            single_accuracy = np.mean(correct_single_preds)
        logger.info(f"Single Label Accuracy (Highest Prob matching Target): {single_accuracy:.4f} (evaluated on {np.sum(valid_indices)}/{len(all_single_target_indices)} samples)")
        # ---------------------------------------

        # --- Optional: Add Confusion Matrix --- #
        try:
            if np.sum(valid_indices) > 0: # Only if we have valid samples
                true_labels = all_single_target_indices[valid_indices]
                pred_labels = all_preds_single_indices[valid_indices]
                labels_present = np.arange(len(class_names))

                cm = confusion_matrix(true_labels, pred_labels, labels=labels_present)
                logger.info("\nConfusion Matrix:")
                logger.info(f"Labels: {class_names}")
                logger.info(f"\n{cm}")
                logger.info("\nPer-Class Metrics:")
                # Calculate metrics per class from CM
                for i, name in enumerate(class_names):
                    TP = cm[i, i]
                    FP = cm[:, i].sum() - TP # Sum of column i (all predicted as i) - TP
                    FN = cm[i, :].sum() - TP # Sum of row i (all truly i) - TP
                    TN = cm.sum() - (TP + FP + FN)

                    # Recall (Sensitivity) = TP / (TP + FN)
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    # Specificity = TN / (TN + FP)
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                    # Precision = TP / (TP + FP)
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    logger.info(f"- {name}:")
                    logger.info(f"    Recall (Sensitivity): {recall:.4f} (TP={TP}, FN={FN})")
                    logger.info(f"    Precision:            {precision:.4f} (TP={TP}, FP={FP})")
                    logger.info(f"    Specificity:          {specificity:.4f} (TN={TN}, FP={FP})")
                    logger.info(f"    F1-Score:             {f1_score:.4f}")

            else:
                logger.warning("Skipping confusion matrix as no valid samples were found.")
        except Exception as e:
            logger.error(f"Error calculating or printing confusion matrix: {e}", exc_info=True)
        # -------------------------------------- #

        # --- Re-implemented: ROC/AUC Calculation (One-vs-Rest) --- #
        logger.info("\nArea Under Curve (AUC) - One-vs-Rest:")
        auc_scores = {}
        fpr = {}
        tpr = {}

        # Convert true labels to one-hot encoding for micro-average AUC calculation
        y_true_one_hot = np.eye(len(class_names))[all_single_target_indices]

        # Calculate ROC/AUC for each class
        for i, name in enumerate(class_names):
            try:
                # Create binary true labels for the current class (OvR)
                y_true_binary = (all_single_target_indices == i).astype(int)
                y_score = all_preds_probs[:, i] # Probabilities for the current class

                # Check if we have both positive and negative samples for this class
                if len(np.unique(y_true_binary)) < 2:
                    logger.warning(f"Skipping ROC/AUC for class \'{name}\' because it only has one class present in the true labels.")
                    fpr[name], tpr[name], auc_scores[name] = None, None, None
                    continue

                fpr[name], tpr[name], _ = roc_curve(y_true_binary, y_score)
                auc_value = auc(fpr[name], tpr[name])
                # Alternative using roc_auc_score:
                # auc_value = roc_auc_score(y_true_binary, y_score)
                auc_scores[name] = auc_value
                logger.info(f"- {name}: {auc_value:.4f}")

            except Exception as e:
                logger.warning(f"Could not calculate ROC/AUC for class \'{name}\': {e}")
                fpr[name], tpr[name], auc_scores[name] = None, None, None # Mark as invalid

        # Calculate average AUCs
        valid_aucs = [v for v in auc_scores.values() if v is not None]
        if valid_aucs:
            macro_auc = np.mean(valid_aucs)
            logger.info(f"- Macro Average: {macro_auc:.4f}")
        else:
            logger.warning("No valid per-class AUC scores found for Macro Average.")

        # Calculate Micro Average AUC globally
        try:
            # Use one-hot encoded true labels and predicted probabilities
            micro_auc = roc_auc_score(y_true_one_hot, all_preds_probs, average="micro", multi_class="ovr")
            logger.info(f"- Micro Average: {micro_auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate Micro Average AUC: {e}")

        # --- Plot ROC Curves --- #
        logger.info("\nGenerating ROC curve plot...")
        plt.figure(figsize=(10, 8))
        for name in class_names:
            if fpr.get(name) is not None and tpr.get(name) is not None and auc_scores.get(name) is not None:
                plt.plot(fpr[name], tpr[name], label=f'{name} (AUC = {auc_scores[name]:.3f})')
            else:
                 logger.warning(f"Skipping ROC plot for class \'{name}\' due to previous calculation error or lack of variance.")

        plt.plot([0, 1], [0, 1], 'k--', label='Chance') # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05]) # Add a little space at the top
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curves per Class (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True)

        # Save the plot
        plot_filename = "roc_curves_ovr.png" # Changed filename
        try:
            plt.savefig(plot_filename)
            logger.info(f"ROC curve plot saved to {os.path.abspath(plot_filename)}")
        except Exception as e:
            logger.error(f"Failed to save ROC curve plot: {e}")
        plt.close() # Close the plot figure to free memory
        # ---------------------------------------------------------- #

    except Exception as e:
        logger.error(f"Error calculating metrics or plotting ROC: {e}", exc_info=True)

    return single_accuracy, macro_auc # Return the collected metrics

if __name__ == '__main__':
    test_json_full_path = os.path.join(TEST_JSON_DIR, TEST_JSON_FILENAME)

    if CHECKPOINT_PATH == "./best_model_epoch_X.pth": # Check against placeholder
         logger.warning("CHECKPOINT_PATH is set to the default placeholder.")
         logger.warning("Please edit soombit/test.py and set CHECKPOINT_PATH to your actual model file.")
    elif not os.path.exists(CHECKPOINT_PATH):
         logger.error(f"Specified CHECKPOINT_PATH does not exist: {CHECKPOINT_PATH}")
    else:
         evaluate_model(
             checkpoint_path=CHECKPOINT_PATH,
             test_json_path=test_json_full_path,
             image_root=IMAGE_ROOT,
             image_root_2=IMAGE_ROOT_2,
             vision_model_name=VISION_MODEL_NAME,
             img_size=IMG_SIZE,
             batch_size=BATCH_SIZE,
             num_workers=NUM_WORKERS,
             threshold=PREDICTION_THRESHOLD
         )

    logger.info("Script finished.")

    # Note: The script does not automatically revert itself. 
    # The next step is to manually revert the changes or use the assistant to do so. 