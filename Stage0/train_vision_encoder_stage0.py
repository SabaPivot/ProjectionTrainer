"""
Stage 0: Fine-tuning Vision Encoder using Image-Text Contrastive Loss (SigLIP-style).

This script adapts a pre-trained vision encoder (like XraySigLIP)
by fine-tuning it on paired image-text data (using 'normal_caption' as text).
The goal is to make the vision encoder produce embeddings that are better aligned
with the semantics of the text descriptions, potentially leading to better
separation of concepts in downstream tasks or visualizations like t-SNE.

It uses a contrastive loss similar to CLIP/SigLIP.
The text encoder is kept frozen during this stage.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F # For cosine similarity & loss
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import math
from transformers import get_cosine_schedule_with_warmup, AutoProcessor, AutoModel, AutoTokenizer
import json
from PIL import Image
import time
import argparse
import pandas as pd
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerator_setup import setup_accelerator_and_logging # Import the setup function

logger = get_logger(__name__, log_level="INFO") # Set default level

# --- Dataset Class ---
class ImageTextContrastiveDataset(Dataset):
    """Dataset for Image-Text Contrastive Learning."""
    def __init__(self, json_path, image_root, processor, tokenizer, max_text_len=77):
        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        logger.info(f"Loading data from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        initial_count = len(df)
        logger.info(f"Initial dataset size: {initial_count}")

        # --- Remove Filtering and Balancing Logic ---
        self.data = df.copy()
        # --- End Remove Filtering and Balancing Logic ---

        # Ensure text data is string
        self.data['normal_caption'] = self.data['normal_caption'].astype(str)

        # Filter out rows with effectively empty captions after stripping
        initial_count = len(self.data)
        self.data = self.data[self.data['normal_caption'].str.strip().str.len() > 0]
        filtered_count = len(self.data)
        if initial_count > filtered_count:
            logger.warning(f"Filtered out {initial_count - filtered_count} rows with empty/whitespace captions.")

        if len(self.data) == 0:
            logger.warning("Dataset is empty after filtering/processing!")

        # Ensure 'normal_caption' exists for tokenization
        if 'normal_caption' not in self.data.columns:
            raise ValueError("Dataset must contain a 'normal_caption' column for text data.")
        # Ensure text data is string
        self.data['normal_caption'] = self.data['normal_caption'].astype(str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, item['image'])
        caption = item['normal_caption']

        try:
            # Process Image
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)

            # Process Text
            # Add a check for empty string just in case, though filtering should handle it
            if not caption or not caption.strip():
                 logger.warning(f"Item {idx} has empty caption after load. Skipping.")
                 return None

            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
                return_attention_mask=True # Explicitly request attention mask
            )
            # Check if keys exist BEFORE accessing them
            if 'input_ids' not in text_inputs or 'attention_mask' not in text_inputs:
                 logger.error(f"Tokenizer failed for item {idx}. Caption: '{caption}'. Tokenizer output keys: {text_inputs.keys()}. Skipping.")
                 return None

            input_ids = text_inputs['input_ids'].squeeze(0)
            attention_mask = text_inputs['attention_mask'].squeeze(0)

        except Exception as e:
            # Log the specific caption causing the error
            logger.error(f"Error loading/processing item {idx} (Image: {image_path}, Caption: '{caption}'): {e}. Skipping.", exc_info=True)
            return None

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


# --- SigLIP Loss ---
# Reference: https://arxiv.org/abs/2303.15343, https://github.com/google-research/big_vision/blob/main/big_vision/losses/contrastive.py
def siglip_loss(image_features, text_features, logit_scale, logit_bias=None):
    """Calculates the SigLIP loss.

    Args:
        image_features: Features from the image encoder (N, D).
        text_features: Features from the text encoder (N, D).
        logit_scale: Temperature parameter (scalar tensor).
        logit_bias: Bias parameter (scalar tensor, optional).

    Returns:
        The SigLIP loss (scalar tensor).
    """
    # Normalize features (important for cosine similarity)
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    # Calculate cosine similarity
    logits = torch.matmul(image_features, text_features.t()) * logit_scale.exp()
    if logit_bias is not None:
        logits += logit_bias

    # Create labels: 1 for positive pairs (diagonal), 0 for negative pairs
    n = logits.size(0)
    # SigLIP uses pairwise sigmoid loss. Binary cross entropy with logits expects 0/1 labels.
    labels_01 = torch.eye(n, device=logits.device)

    # Calculate loss using binary cross entropy with logits (combines sigmoid and BCE)
    loss = F.binary_cross_entropy_with_logits(logits, labels_01, reduction='sum') / n

    return loss

# --- Trainer Class ---
class VisionEncoderTrainerStage0:
    def __init__(
        self,
        accelerator,
        model_name,
        processor, # Image processor
        tokenizer, # Text tokenizer
        train_dataset,
        output_dir="./trained_vision_encoder_stage0_contrastive", # Updated dir name
        batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        num_epochs=5,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        freeze_layers_ratio=0.0,
        freeze_text_encoder=True, # Freeze text encoder by default
        freeze_logit_scale=True,  # Freeze logit_scale by default
        wandb_project="vision_encoder_contrastive_finetuning", # Updated project name
        save_every_n_epochs=1,
        logging_steps=100 # Add logging_steps parameter
    ):
        self.accelerator = accelerator
        self.processor = processor
        self.tokenizer = tokenizer # Store tokenizer
        self.device = accelerator.device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.batch_size = batch_size # Store batch_size for logging
        self.logging_steps = logging_steps # Store logging_steps

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )

        # --- Load Model (Vision + Text + Logit Scale) ---
        logger.info(f"Loading model with vision and text parts: {model_name}")
        # Load the base model (e.g., SigLIP has vision_model, text_model, logit_scale)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # --- Layer Freezing ---
        # Freeze Text Encoder (optional)
        if freeze_text_encoder:
            if hasattr(self.model, 'text_model'):
                for param in self.model.text_model.parameters():
                    param.requires_grad = False
                logger.info("Froze Text Encoder parameters.")
            else:
                logger.warning("Model does not have 'text_model' attribute. Cannot freeze text encoder.")

        # Freeze Logit Scale (optional)
        if freeze_logit_scale:
            if hasattr(self.model, 'logit_scale'):
                self.model.logit_scale.requires_grad = False
                logger.info("Froze logit_scale parameter.")
            else:
                 logger.warning("Model does not have 'logit_scale' attribute. Cannot freeze logit_scale.")

        # Freeze Vision Encoder Layers (optional)
        if freeze_layers_ratio > 0:
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder') and hasattr(self.model.vision_model.encoder, 'layers'):
                 num_total_layers = len(self.model.vision_model.encoder.layers)
                 num_freeze = int(num_total_layers * freeze_layers_ratio)
                 if num_freeze > 0:
                     logger.info(f"Freezing the first {num_freeze}/{num_total_layers} layers of the vision encoder.")
                     if hasattr(self.model.vision_model, 'embeddings'):
                         for param in self.model.vision_model.embeddings.parameters():
                             param.requires_grad = False
                     for i, layer in enumerate(self.model.vision_model.encoder.layers):
                         if i < num_freeze:
                             for param in layer.parameters():
                                 param.requires_grad = False
                         else:
                             for param in layer.parameters():
                                 param.requires_grad = True # Ensure rest are trainable if needed
                 else:
                     logger.info("freeze_layers_ratio > 0 but calculated num_freeze is 0. Training all vision layers.")
            else:
                logger.warning(f"Could not determine vision encoder layers structure for freezing. Training all available vision layers.")

        # --- Parameters to Optimize ---
        params_to_optimize = []
        trainable_param_count = 0
        total_param_count = 0
        for name, param in self.model.named_parameters():
            total_param_count += param.numel()
            if param.requires_grad:
                params_to_optimize.append(param)
                trainable_param_count += param.numel()
                # logger.debug(f" Parameter to optimize: {name}") # Uncomment for detailed debugging

        logger.info(f"Total parameters: {total_param_count:,}")
        logger.info(f"Trainable parameters: {trainable_param_count:,}")
        if trainable_param_count == 0:
             raise ValueError("No parameters set to trainable. Check freezing logic.")


        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # --- Scheduler ---
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        self.max_train_steps = num_epochs * num_update_steps_per_epoch
        num_warmup_steps = math.ceil(warmup_ratio * self.max_train_steps)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        logger.info(f"Max train steps: {self.max_train_steps}, Warmup steps: {num_warmup_steps}")

        # --- Prepare with Accelerator ---
        # Important: Prepare the *entire* model, even if parts are frozen
        self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler
        )

    def train(self):
        """Fine-tune the vision encoder using contrastive loss"""
        logger.info(f"Process {self.accelerator.process_index}: Starting Stage 0 contrastive training for {self.num_epochs} epochs.")
        
        # Use the stored batch_size from init args instead of accessing prepared loader's attribute
        total_batch_size = self.accelerator.num_processes * self.batch_size * self.accelerator.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_loader.dataset)}")
        logger.info(f"  Num Epochs = {self.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, dist. & accum.) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        global_step = 0
        # Ensure text encoder is in eval mode if frozen
        if hasattr(self.model.module, 'text_model'): # Access via .module
            self.model.module.text_model.eval()

        for epoch in range(self.num_epochs):
            if hasattr(self.model.module, 'vision_model'): # Access via .module
                self.model.module.vision_model.train() # Only vision part needs train mode (unless freezing all)
            else:
                logger.warning("Could not find vision_model on model.module. Setting entire model to train mode.")
                self.model.train()
            
            epoch_train_loss = 0.0
            processed_batches = 0

            progress_bar = tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1}/{self.num_epochs} [Stage 0 Contrastive]",
                disable=not self.accelerator.is_main_process
            )

            for step, batch in enumerate(self.train_loader):
                if batch is None:
                    logger.warning(f"Skipping step {step} due to image loading errors.")
                    continue
                
                with self.accelerator.accumulate(self.model): # Accumulate gradients on the main model
                    pixel_values = batch["pixel_values"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    # === Forward Pass ===
                    try:
                        # Get embeddings from the model
                        outputs = self.model(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            return_loss=False, # We calculate loss manually
                            return_dict=True
                        )
                        # Extract features (adjust keys based on your model's output structure)
                        # Common keys: image_embeds, text_embeds OR vision_model_output, text_model_output
                        # Use getattr for safer access
                        image_features = getattr(outputs, 'image_embeds', getattr(outputs.vision_model_output, 'pooler_output', None))
                        text_features = getattr(outputs, 'text_embeds', getattr(outputs.text_model_output, 'pooler_output', None))

                        if image_features is None or text_features is None:
                            raise ValueError("Could not extract image or text features from model output.")

                        # Get logit_scale (and optional logit_bias)
                        # Need to handle potential DDP wrapping
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        if not hasattr(unwrapped_model, 'logit_scale'):
                            raise ValueError("Model does not have 'logit_scale'. Cannot compute SigLIP loss.")
                        logit_scale = unwrapped_model.logit_scale
                        logit_bias = getattr(unwrapped_model, 'logit_bias', None) # Optional bias

                        # === Calculate SigLIP Loss ===
                        loss = siglip_loss(image_features, text_features, logit_scale, logit_bias)

                    except Exception as e:
                        logger.error(f"Error during model forward pass or loss calculation: {e}", exc_info=True)
                        # Optional: Add debugging for feature shapes if error persists
                        # logger.error(f"Image features shape: {image_features.shape if 'image_features' in locals() else 'N/A'}")
                        # logger.error(f"Text features shape: {text_features.shape if 'text_features' in locals() else 'N/A'}")
                        continue # Skip batch if forward pass or loss calculation fails

                    epoch_train_loss += loss.item()
                    processed_batches += 1

                    # --- Backpropagation ---
                    self.accelerator.backward(loss)

                    # --- Optimizer Step ---
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # --- Logging ---
                    if global_step % self.logging_steps == 0: # Use self.logging_steps
                        if self.accelerator.is_main_process:
                             # Gather loss across processes for accurate logging
                            # --- TEMP: Disable gather for debugging stall --- 
                            # avg_loss_step = self.accelerator.gather(loss).mean().item()
                            avg_loss_step = loss.item() # Log only main process loss
                            logger.info(f"[Rank {self.accelerator.process_index}] Step {global_step} Local Loss: {avg_loss_step:.4f}") # Add rank info
                            # --- End TEMP --- 

                            log_dict = {
                                "train/siglip_loss": avg_loss_step, # Note: This is now local loss
                                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                                "step": global_step
                            }
                            # Add logit scale if it exists and isn't frozen
                            unwrapped_model_log = self.accelerator.unwrap_model(self.model)
                            if hasattr(unwrapped_model_log, 'logit_scale'):
                                if unwrapped_model_log.logit_scale.requires_grad:
                                     log_dict["train/logit_scale"] = unwrapped_model_log.logit_scale.item()
                                else:
                                     log_dict["train/logit_scale_frozen"] = unwrapped_model_log.logit_scale.item()
                                
                            self.accelerator.log(log_dict, step=global_step)
                            progress_bar.set_postfix({"loss": f"{avg_loss_step:.4f}", "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"})
                
                progress_bar.update(1) # Update progress bar per batch


            # --- End of Epoch ---
            progress_bar.close()
            avg_epoch_train_loss = 0.0
            # --- TEMP: Disable gather for debugging stall --- 
            # if processed_batches > 0:
            #      # Gather epoch loss across processes
            #     epoch_train_loss_tensor = torch.tensor(epoch_train_loss, device=self.device)
            #     gathered_epoch_losses = self.accelerator.gather(epoch_train_loss_tensor)
            #     # Gather processed batches count
            #     processed_batches_tensor = torch.tensor(processed_batches, device=self.device)
            #     gathered_processed_batches = self.accelerator.gather(processed_batches_tensor)

            #     # Calculate average loss on main process
            #     if self.accelerator.is_main_process:
            #         total_epoch_loss = gathered_epoch_losses.sum().item()
            #         total_processed_batches = gathered_processed_batches.sum().item()
            #         if total_processed_batches > 0:
            #              avg_epoch_train_loss = total_epoch_loss / total_processed_batches
            #         else:
            #              logger.warning(f"Epoch {epoch+1} had no processed batches across all processes.")
            if self.accelerator.is_main_process and processed_batches > 0:
                 avg_epoch_train_loss = epoch_train_loss / processed_batches # Use local main process avg
                 logger.info(f"[Rank {self.accelerator.process_index}] Epoch {epoch+1} Local Avg Loss: {avg_epoch_train_loss:.4f}")
            elif self.accelerator.is_main_process:
                 logger.warning(f"Epoch {epoch+1} had no processed batches on main process.")
            # --- End TEMP --- 

            if self.accelerator.is_main_process:
                epoch_log_dict = {
                    "train/epoch_siglip_loss": avg_epoch_train_loss, # Note: Now local avg
                    "epoch": epoch + 1
                }
                self.accelerator.log(epoch_log_dict, step=global_step)
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed. Avg Train Loss: {avg_epoch_train_loss:.4f}")

                # --- Periodic Saving ---
                if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_model(epoch=epoch+1)

        # --- End of Training ---
        logger.info("Stage 0 SigLIP Training finished.")
        self.save_model(epoch=self.num_epochs) # Save final model
        logger.info(f"Final fine-tuned vision encoder saved to {self.output_dir}")

    def save_model(self, epoch=None):
        """Save the fine-tuned vision encoder model state."""
        # Ensure model saving happens only on the main process
        if self.accelerator.is_main_process:
            save_dir_name = f"epoch_{epoch}" if epoch is not None else "final_model"
            full_save_dir = os.path.join(self.output_dir, save_dir_name)
            os.makedirs(full_save_dir, exist_ok=True)

            # Unwrap the main model to access sub-components
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # --- Save the Full Model using save_pretrained --- 
            try:
                logger.info(f"Saving full model to {full_save_dir}")
                unwrapped_model.save_pretrained(full_save_dir)
                logger.info(f"Full model saved successfully.")

                # --- Also save Processor and Tokenizer --- 
                if hasattr(self, 'processor') and self.processor:
                     self.processor.save_pretrained(full_save_dir)
                     logger.info(f"Processor config saved to {full_save_dir}")
                if hasattr(self, 'tokenizer') and self.tokenizer:
                     self.tokenizer.save_pretrained(full_save_dir)
                     logger.info(f"Tokenizer config saved to {full_save_dir}")

            except Exception as e:
                 logger.error(f"Error saving full model/processor/tokenizer to {full_save_dir}: {e}", exc_info=True)

            # --- (Optional) Keep code for saving only vision encoder if needed --- 
            # # Save only the vision_encoder part
            # if hasattr(unwrapped_model, 'vision_model'):
            #     vision_encoder_save_path = os.path.join(full_save_dir, "vision_encoder.bin") # Or .safetensors
            #     try:
            #         # Save state dict directly
            #         torch.save(unwrapped_model.vision_model.state_dict(), vision_encoder_save_path)
            #         logger.info(f"Vision encoder state_dict saved to {vision_encoder_save_path}")
                    
            #         # Save processor alongside the vision encoder weights
            #         if hasattr(self, 'processor') and self.processor:
            #              self.processor.save_pretrained(full_save_dir)
            #              logger.info(f"Processor config saved to {full_save_dir}")
            #         # Save tokenizer if it's different or needs specific config
            #         if hasattr(self, 'tokenizer') and self.tokenizer:
            #              self.tokenizer.save_pretrained(full_save_dir)
            #              logger.info(f"Tokenizer config saved to {full_save_dir}")

            #     except Exception as e:
            #          logger.error(f"Error saving vision encoder state/processor/tokenizer to {full_save_dir}: {e}", exc_info=True)
            # else:
            #     logger.error("Could not find 'vision_model' attribute in unwrapped model. Cannot save vision encoder separately.")

        # Remove wait_for_everyone() here, let accelerate handle synchronization implicitly
        # self.accelerator.wait_for_everyone()

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a vision encoder using SigLIP loss (Stage 0)")
    # --- Essential Arguments ---
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name (MUST support vision and text, e.g., SigLIP)")
    parser.add_argument("--train_json", type=str, required=True, help="Path to the training data JSON file (must contain 'image' and 'normal_caption')")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for image files")
    parser.add_argument("--output_dir", type=str, default="./trained_vision_encoder_stage0_contrastive", help="Directory to save checkpoints and logs")

    # --- Data Handling Arguments ---
    parser.add_argument("--max_text_len", type=int, default=77, help="Maximum sequence length for text tokenizer")

    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate for vision encoder fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")

    # --- Freezing Options ---
    parser.add_argument("--freeze_layers_ratio", type=float, default=0.0, help="Ratio of initial vision encoder layers to freeze (0.0 trains all vision layers)")
    # Default is True (frozen)
    parser.add_argument("--freeze_text_encoder", action=argparse.BooleanOptionalAction, default=True, help="Freeze the text encoder weights.") 
    parser.add_argument("--freeze_logit_scale", action=argparse.BooleanOptionalAction, default=True, help="Freeze the logit_scale parameter.")
    # Add trust_remote_code argument
    parser.add_argument("--trust_remote_code", action='store_true', help="Allow loading models with custom code from Hugging Face Hub.")


    # --- Other Arguments ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every N epochs (0 saves only at end)")
    parser.add_argument("--wandb_project", type=str, default="vision_encoder_siglip_stage0", help="WandB project name")
    parser.add_argument("--log_with", type=str, default="wandb", help="Tracker(s) to log with (e.g., 'wandb', 'tensorboard', 'all', 'none')")
    # Add logging_steps argument
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every N global steps.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (defaults to W&B generated name)")
    parser.add_argument("--mixed_precision", type=str, default="no", help="Mixed precision type ('no', 'fp16', 'bf16')")
    # Add disable_wandb argument needed by setup script
    parser.add_argument("--disable_wandb", action='store_true', help="Disable WandB logging.")


    return parser.parse_args()

# --- Main Function ---
def main():
    args = parse_args()

    # --- Setup Accelerator and Logging ---
    # Pass args to the setup function
    # Accelerator setup should handle logging config and WandB init
    accelerator = setup_accelerator_and_logging(args)

    # --- Set Seed (after accelerator is initialized for consistency) ---
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed for reproducibility: {args.seed}")

    # --- Load Processor and Tokenizer ---
    logger.info(f"Loading processor and tokenizer from: {args.model_name}")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_name, 
            trust_remote_code=args.trust_remote_code
        )
        # SigLIP processor often includes tokenizer
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
        else:
            # Fallback if tokenizer isn't directly attached
            logger.warning("Processor does not have 'tokenizer' attribute, attempting to load AutoTokenizer separately.")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, 
                trust_remote_code=args.trust_remote_code
            )
        logger.info(f"Loaded Processor and Tokenizer from {args.model_name}")
    except Exception as e:
        logger.error(f"Failed to load processor/tokenizer for {args.model_name}: {e}", exc_info=True)
        exit(1)

    # --- Load Dataset ---
    try:
        train_dataset = ImageTextContrastiveDataset(
            json_path=args.train_json,
            image_root=args.image_root,
            processor=processor, # Pass the loaded image processor
            tokenizer=tokenizer, # Pass the separately loaded tokenizer
            max_text_len=args.max_text_len
        )
        if len(train_dataset) == 0:
             logger.error("Training dataset is empty after processing. Please check data paths and filtering.")
             exit(1)
        logger.info(f"Dataset loaded with {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        exit(1)


    # --- Initialize Trainer ---
    try:
        trainer = VisionEncoderTrainerStage0(
            accelerator=accelerator, # Pass the initialized accelerator
            model_name=args.model_name, # Trainer still loads the main model
            processor=processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            freeze_layers_ratio=args.freeze_layers_ratio,
            # Pass freezing args directly
            freeze_text_encoder=args.freeze_text_encoder,
            freeze_logit_scale=args.freeze_logit_scale,
            wandb_project=args.wandb_project,
            save_every_n_epochs=args.save_every_n_epochs,
            logging_steps=args.logging_steps # Pass logging_steps
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        exit(1)

    # --- Start Training ---
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        exit(1)
    finally:
        # Accelerator might handle tracker ending automatically, but explicit call is safe
        if accelerator.is_main_process and accelerator.trackers:
           try:
               accelerator.end_training()
               logger.info("Ended trackers.")
           except Exception as e:
               logger.error(f"Error ending trackers: {e}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main() 