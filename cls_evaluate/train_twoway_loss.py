import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for logsumexp
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipVisionModel, AutoConfig, AutoProcessor
from PIL import Image
import os
import json
import logging
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_JSON = "/mnt/samuel/Siglip/soombit/single_label_dataset.json"
IMAGE_ROOT = "/mnt/data/CXR/NIH Chest X-rays_jpg"
IMAGE_ROOT_2 = "/mnt/data/CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
CLASS_NAMES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion"]
VISION_MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
IMG_SIZE = 384
BATCH_SIZE = 16 # Note: Class-wise loss works on the full batch/bucket size
LEARNING_RATE = 5e-6
NUM_EPOCHS = 10
NUM_WORKERS = 4
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 0.01
BACKBONE_LEARNING_RATE = 1e-8
FREEZE_VISION_ENCODER = True
OUTPUT_DIR = "./soombit/checkpoints"
TRAIN_VISION_ENCODER_FIRST_EPOCH = False
LOSS_TEMP_P = 4.0 # Positive temperature for TwoWayMultiLabelLoss [cite: 146]
LOSS_TEMP_N = 1.0 # Negative temperature for TwoWayMultiLabelLoss [cite: 146]

# === Step 1: Data Loading ===

class XrayClassificationDataset(Dataset):
    """Dataset for X-ray image classification, modified for multi-hot labels."""
    def __init__(self, samples, image_root, class_names, processor, img_size, image_root_2=None):
        self.image_root = image_root
        self.image_root_2 = image_root_2
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.processor = processor
        self.img_size = img_size
        self.samples = samples
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_filename = sample.get("image")
        caption = sample.get("normal_caption", "") # Label(s) are expected here

        # *** Create Multi-Hot Target Vector ***
        target_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        # --- Modification Point ---
        # Assuming caption *is* the single label (like original code)
        # If caption contains multiple labels (e.g., "Atelectasis,Cardiomegaly"),
        # you need to parse it here.
        target_label_str = caption.strip()
        if target_label_str in self.class_to_idx:
            target_index = self.class_to_idx[target_label_str]
            target_vector[target_index] = 1.0
        elif target_label_str: # Only warn if caption is not empty but not found
             logger.warning(f"Sample {idx}: Ground truth label \"{target_label_str}\" in normal_caption"
                           f" not found in CLASS_NAMES: {self.class_names}. Creating zero vector.")
        # If no valid label found, target_vector remains all zeros.
        # --- End Modification Point ---


        if not image_filename:
            logger.warning(f"Sample {idx} missing image filename. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # --- Determine Image Path ---
        if image_filename.startswith("p") and "/" in image_filename and self.image_root_2:
            image_path = os.path.join(self.image_root_2, image_filename)
            # Handle potential directory structure (simplified)
            if os.path.isdir(image_path):
                 try:
                     jpg_files = [f for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
                     if not jpg_files:
                         logger.warning(f"No .jpg files found in directory: {image_path}. Skipping sample {idx}.")
                         return self.__getitem__((idx + 1) % len(self))
                     image_path = os.path.join(image_path, jpg_files[0]) # Use the first JPG found
                 except Exception as e:
                      logger.error(f"Error processing directory {image_path} for sample {idx}: {e}", exc_info=True)
                      return self.__getitem__((idx + 1) % len(self))
        else:
            image_path = os.path.join(self.image_root, image_filename)
        # -----------------------------

        try:
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.pixel_values.squeeze(0) # Remove batch dim

            # Check if any label was assigned
            # if target_vector.sum() == 0 and target_label_str: # Only skip if a label was expected but not found
            #     logger.warning(f"Sample {idx} has no valid labels assigned based on caption '{target_label_str}'. Skipping.")
            #     return self.__getitem__((idx + 1) % len(self))

            return {
                "pixel_values": pixel_values,
                "target_vector": target_vector # Return multi-hot vector
            }

        except FileNotFoundError:
            logger.warning(f"Image file not found for sample {idx}: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({image_path}): {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))

# --- Collate Function (Modified) ---
def classification_collate_fn(batch):
    """Stacks pixel_values and target_vectors."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    target_vectors = torch.stack([item["target_vector"] for item in batch]) # Stack the vectors

    return {
        "pixel_values": pixel_values,
        "target_vectors": target_vectors # Tensor of multi-hot vectors
    }

# === Step 2 & 3: Model Definition, Freezing, Loss ===

class AbnormalityClassifier(nn.Module):
    # (No changes needed in the model architecture itself)
    def __init__(self, vision_model_name, class_names, embed_dim=1024, num_heads=16, dropout_rate=0.1):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.embed_dim = embed_dim
        self.vision_model = SiglipVisionModel.from_pretrained(vision_model_name)
        self.abnormality_queries = nn.Parameter(torch.randn(1, self.num_classes, self.embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.classification_head = nn.Linear(self.embed_dim, 1) # Output is 1 logit per query

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        image_features = vision_outputs.last_hidden_state
        batch_queries = self.abnormality_queries.repeat(batch_size, 1, 1)
        attn_output, _ = self.mha(batch_queries, image_features, image_features)
        attn_output_dropout = self.dropout(attn_output)
        # Logits shape before squeeze: (batch_size, num_classes, 1)
        logits = self.classification_head(attn_output_dropout)
        # Logits shape after squeeze: (batch_size, num_classes)
        logits = logits.squeeze(-1)
        return logits

# === Two-way Multi-Label Loss Implementation ===
class TwoWayMultiLabelLoss(nn.Module):
    """
    Implements the Two-way Multi-Label Loss function as described in:
    Kobayashi, T. (2023). Two-way Multi-Label Loss. CVPR. [cite: 4]
    """
    def __init__(self, T_p=4.0, T_n=1.0, base_loss_func=F.softplus):
        """
        Args:
            T_p (float): Temperature for positive logits[cite: 74, 146].
            T_n (float): Temperature for negative logits[cite: 74, 146].
            base_loss_func (callable): The base softplus-like function.
        """
        super().__init__()
        if T_p <= 0 or T_n <= 0:
             raise ValueError("Temperatures T_p and T_n must be positive.") # Based on Prop. 2 [cite: 73]
        self.T_p = T_p
        self.T_n = T_n
        self.base_loss_func = base_loss_func
        self.eps = 1e-6 # Small epsilon for numerical stability in logsumexp

    def _single_loss(self, logits, labels, T_p, T_n):
        """Calculates the core multi-label loss for a single sample or class."""
        # logits: Tensor of size [N] (either classes for a sample or samples for a class)
        # labels: Binary tensor of size [N] (0 or 1)
        # T_p, T_n: Temperatures

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        # Check if there are any positive or negative labels
        has_pos = torch.any(pos_mask)
        has_neg = torch.any(neg_mask)

        # Handle cases with no positives or no negatives
        if not has_pos or not has_neg:
            # If no positives or no negatives, loss contribution is zero
            # or handle as per specific requirements (e.g., log a warning)
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        # Calculate log-sum-exp terms (Eq. 14) [cite: 74]
        # Term for negatives: T_n * log(sum(exp(x_n / T_n)))
        logsumexp_neg = T_n * torch.logsumexp(neg_logits / T_n, dim=0)

        # Term for positives: T_p * log(sum(exp(-x_p / T_p)))
        logsumexp_pos = T_p * torch.logsumexp(-pos_logits / T_p, dim=0)

        # Combine terms inside the base loss function (e.g., softplus)
        loss_val = self.base_loss_func(logsumexp_neg + logsumexp_pos) # [cite: 65, 74]

        return loss_val

    def forward(self, logits, targets):
        """
        Calculates the two-way multi-label loss.
        Args:
            logits (torch.Tensor): Logits from the model. Shape (batch_size, num_classes).
            targets (torch.Tensor): Multi-hot encoded target labels. Shape (batch_size, num_classes).
        Returns:
            torch.Tensor: The final computed loss (scalar).
        """
        batch_size, num_classes = logits.shape

        if logits.shape != targets.shape:
            raise ValueError(f"Logits shape {logits.shape} must match targets shape {targets.shape}")

        # --- 1. Sample-wise Loss (Eq. 15) --- [cite: 86]
        sample_losses = []
        for i in range(batch_size):
            sample_loss = self._single_loss(logits[i, :], targets[i, :], self.T_p, self.T_n)
            sample_losses.append(sample_loss)
        # Average sample-wise loss
        avg_sample_loss = torch.mean(torch.stack(sample_losses))

        # --- 2. Class-wise Loss (Eq. 16) --- [cite: 90]
        # Uses the full batch as the bucket as in paper's implementation [cite: 138]
        class_losses = []
        for c in range(num_classes):
            # Note: In class-wise, the roles of pos/neg samples are based on targets[:, c]
            # The loss is calculated on the logits for that class across the batch
            # We need to swap pos/neg definition compared to _single_loss interpretation
            # Or rather, feed pos/neg logits directly based on targets for this class
            class_logits_c = logits[:, c]
            class_targets_c = targets[:, c]

            pos_mask_c = (class_targets_c == 1)
            neg_mask_c = (class_targets_c == 0)

            has_pos_c = torch.any(pos_mask_c)
            has_neg_c = torch.any(neg_mask_c)

            if not has_pos_c or not has_neg_c:
                 class_losses.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))
                 continue # Skip if no positive or negative samples for this class in the batch

            pos_logits_c = class_logits_c[pos_mask_c] # Samples positive for class c
            neg_logits_c = class_logits_c[neg_mask_c] # Samples negative for class c

            # Calculate log-sum-exp terms for class-wise loss
            # Term for negative SAMPLES (logits from samples where target=0 for class c)
            # Note: In Eq 16, the sum is over i where y_ic=0, using x_ic
            logsumexp_neg_samples = self.T_n * torch.logsumexp(neg_logits_c / self.T_n, dim=0)

            # Term for positive SAMPLES (logits from samples where target=1 for class c)
            # Note: In Eq 16, the sum is over j where y_jc=1, using -x_jc/T
            logsumexp_pos_samples = self.T_p * torch.logsumexp(-pos_logits_c / self.T_p, dim=0)

            class_loss = self.base_loss_func(logsumexp_neg_samples + logsumexp_pos_samples)
            class_losses.append(class_loss)

        # Average class-wise loss
        avg_class_loss = torch.mean(torch.stack(class_losses))


        # --- 3. Combine Losses (Eq. 17) --- [cite: 96]
        # Simple average, can add weighting factors alpha, beta if needed
        final_loss = (avg_sample_loss + avg_class_loss) / 2.0

        return final_loss


# === Evaluation Function (Modified) ===
def evaluate(model, data_loader, criterion, device, class_names):
    """Evaluates the model on the given data loader and returns loss."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            num_batches += 1
            pixel_values = batch["pixel_values"].to(device)
            # Use target_vectors now
            target_vectors = batch["target_vectors"].to(device)

            # Forward pass
            logits = model(pixel_values)

            # Calculate loss using the provided criterion (TwoWayMultiLabelLoss)
            loss = criterion(logits, target_vectors)
            total_loss += loss.item()

            # --- Accuracy Calculation Removed ---
            # Standard accuracy (argmax) is not suitable for multi-label.
            # Metrics like mAP, Precision, Recall per class would be needed here.
            # For simplicity, we only return the loss.

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Return only loss, as accuracy calculation needs significant change
    return avg_loss

# === Main Training Script ===
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = AbnormalityClassifier(
        vision_model_name=VISION_MODEL_NAME,
        class_names=CLASS_NAMES,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)

    # --- Apply Freezing Strategy ---
    if FREEZE_VISION_ENCODER:
        logger.info("Setting up trainable parameters (Vision Encoder FROZEN)...")
        model.vision_model.requires_grad_(False)
    else:
        logger.info("Setting up trainable parameters (Vision Encoder TRAINABLE)...")
        model.vision_model.requires_grad_(True)

    model.abnormality_queries.requires_grad_(True)
    model.mha.requires_grad_(True)
    model.dropout.requires_grad_(True) # Dropout itself doesn't have params, but good practice
    model.classification_head.requires_grad_(True)
    logger.info("Queries, MHA, and Classification Head are trainable.")

    # --- Load and Split Data ---
    logger.info(f"Loading data from {DATA_JSON}...")
    try:
        with open(DATA_JSON, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        logger.info(f"Loaded {len(all_samples)} total samples.")
    except Exception as e:
        logger.error(f"Error loading data JSON {DATA_JSON}: {e}", exc_info=True)
        exit()

    # --- Perform Train/Val Split ---
    # Stratification based on single primary label might be less ideal for true multi-label.
    # If data is multi-label, consider iterative stratification or random split.
    # Sticking to original split logic for now.
    temp_class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    # Extract *first* label found for stratification purposes (approximation)
    labels_for_strat = []
    valid_indices_for_strat = []
    for i, sample in enumerate(all_samples):
        caption = sample.get("normal_caption", "").strip()
        lbl_idx = temp_class_to_idx.get(caption, -1)
        # Only include samples that have at least one valid label for stratification
        if lbl_idx != -1:
            labels_for_strat.append(lbl_idx)
            valid_indices_for_strat.append(i)

    if len(valid_indices_for_strat) < len(all_samples):
         logger.warning(f"Removed {len(all_samples) - len(valid_indices_for_strat)} samples without a recognized primary label before splitting.")
         stratify_samples = [all_samples[i] for i in valid_indices_for_strat]
    else:
         stratify_samples = all_samples

    if len(stratify_samples) < 2:
        logger.error("Not enough valid samples for train/validation split.")
        exit()

    try:
        train_samples, val_samples = train_test_split(
            stratify_samples,
            test_size=0.05,
            random_state=42,
            stratify=labels_for_strat # Stratify based on the primary label found
        )
        logger.info(f"Split data into {len(train_samples)} training samples and {len(val_samples)} validation samples (stratified by primary label).")
    except ValueError as e:
        logger.warning(f"Could not stratify split (maybe too few samples per class?): {e}. Performing random split instead.")
        train_samples, val_samples = train_test_split(
            all_samples, # Use all samples for random split
            test_size=0.05,
            random_state=42
        )
        logger.info(f"Split data randomly into {len(train_samples)} training samples and {len(val_samples)} validation samples.")


    # --- Create Datasets and DataLoaders ---
    train_dataset = XrayClassificationDataset(
        samples=train_samples, image_root=IMAGE_ROOT, class_names=CLASS_NAMES,
        processor=processor, img_size=IMG_SIZE, image_root_2=IMAGE_ROOT_2
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=classification_collate_fn
    )

    if val_samples:
        val_dataset = XrayClassificationDataset(
             samples=val_samples, image_root=IMAGE_ROOT, class_names=CLASS_NAMES,
             processor=processor, img_size=IMG_SIZE, image_root_2=IMAGE_ROOT_2
        )
        val_loader = DataLoader(
             val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
             pin_memory=True, collate_fn=classification_collate_fn
        )
        logger.info(f"Loaded {len(val_dataset)} validation samples.")
    else:
         logger.warning("Validation set is empty. Proceeding without validation loop.")
         val_loader = None

    # --- Optimizer --- (Parameter grouping logic remains the same)
    head_params = ([model.abnormality_queries] +
                   list(model.mha.parameters()) +
                   list(model.classification_head.parameters()))
    num_head_params = sum(p.numel() for p in head_params if p.requires_grad)

    if FREEZE_VISION_ENCODER:
        logger.info("Optimizer will only include head parameters.")
        optimizer_params = [{"params": head_params, "lr": LEARNING_RATE}]
        total_trainable_params = num_head_params
    else:
        logger.info("Optimizer will include head and vision backbone parameters with different LRs.")
        backbone_params = list(model.vision_model.parameters())
        num_backbone_params = sum(p.numel() for p in backbone_params if p.requires_grad)
        optimizer_params = [
            {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_params, "lr": LEARNING_RATE}
        ]
        total_trainable_params = num_head_params + num_backbone_params
        logger.info(f"  - Vision Backbone: {num_backbone_params} trainable parameters (LR={BACKBONE_LEARNING_RATE})")

    logger.info(f"  - Head (Queries, MHA, Classif): {num_head_params} trainable parameters (LR={LEARNING_RATE})")
    logger.info(f"Total Trainable Parameters: {total_trainable_params}")

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)
    logger.info(f"Using AdamW optimizer with Weight Decay={WEIGHT_DECAY}")

    # --- Loss Function (Two-way Multi-Label Loss) ---
    criterion = TwoWayMultiLabelLoss(T_p=LOSS_TEMP_P, T_n=LOSS_TEMP_N).to(device)
    logger.info(f"Using TwoWayMultiLabelLoss with T_p={LOSS_TEMP_P}, T_n={LOSS_TEMP_N}.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    # --- Training Loop ---
    logger.info("Starting training...")
    best_val_loss = float('inf') # Track best validation loss now

    # Initial VE state setup
    if FREEZE_VISION_ENCODER:
        model.vision_model.requires_grad_(False)
        logger.info("Vision encoder is permanently frozen.")
    else:
        model.vision_model.requires_grad_(True)
        logger.info(f"Vision encoder is initially trainable. First epoch only train: {TRAIN_VISION_ENCODER_FIRST_EPOCH}")

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Dynamic VE freezing/unfreezing logic
        if not FREEZE_VISION_ENCODER:
             is_first_epoch = (epoch == 0)
             if TRAIN_VISION_ENCODER_FIRST_EPOCH:
                 if is_first_epoch:
                     logger.info(f"Epoch {epoch+1}: Training Vision Encoder (first epoch only).")
                     model.vision_model.requires_grad_(True)
                     model.vision_model.train()
                 else:
                     if epoch == 1: logger.info(f"Epoch {epoch+1}: Freezing Vision Encoder for subsequent epochs.")
                     model.vision_model.requires_grad_(False)
                     model.vision_model.eval()
             else: # Fine-tune throughout
                  if epoch == 0: logger.info(f"Epoch {epoch+1}: Vision Encoder is trainable throughout.")
                  model.vision_model.requires_grad_(True)
                  model.vision_model.train()
        else: # Permanently frozen
             model.vision_model.eval()

        epoch_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)

        for batch in train_progress_bar:
            if batch is None: continue
            pixel_values = batch["pixel_values"].to(device)
            # Use target_vectors now
            target_vectors = batch["target_vectors"].to(device)

            optimizer.zero_grad()
            logits = model(pixel_values)
            # Use the new loss function
            loss = criterion(logits, target_vectors)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Training Loss: {avg_epoch_train_loss:.4f}")

        # --- Validation Phase ---
        if val_loader:
            # Evaluate function now returns only loss
            avg_epoch_val_loss = evaluate(model, val_loader, criterion, device, CLASS_NAMES)
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {avg_epoch_val_loss:.4f}")

            # --- Save Best Checkpoint (Based on Validation Loss) ---
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                save_path = os.path.join(OUTPUT_DIR, f"best_model_epoch_{epoch+1}.pth")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss, # Saving best validation loss
                    # 'accuracy': best_val_accuracy, # Accuracy removed
                    'class_names': CLASS_NAMES
                }
                torch.save(checkpoint, save_path)
                logger.info(f"Saved new best model checkpoint to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            avg_epoch_val_loss = float('inf') # Placeholder if no validation
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Skipping validation loop.")

        # --- Periodic Checkpoint Saving ---
        if (epoch + 1) % 2 == 0:
             periodic_save_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
             periodic_checkpoint = {
                 'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'train_loss': avg_epoch_train_loss,
                 'val_loss': avg_epoch_val_loss,
                 # 'val_accuracy': val_accuracy, # Accuracy removed
                 'class_names': CLASS_NAMES
             }
             torch.save(periodic_checkpoint, periodic_save_path)
             logger.info(f"Saved periodic checkpoint to {periodic_save_path} (Epoch {epoch+1})")

    logger.info("Training finished.")