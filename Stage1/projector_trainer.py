import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import math
from transformers import get_cosine_schedule_with_warmup
import json
from PIL import Image # Added for validation image loading
import time

# Setup logger for this module
logger = logging.getLogger(__name__)

class ProjectionTrainerStage1:
    def __init__(
        self,
        accelerator,
        vision_encoder, # Frozen
        language_model, # Frozen
        projection_layer, # Trained
        processor, # Image processor
        tokenizer, # LLM tokenizer
        train_dataset,
        output_dir="./trained_projection_stage1",
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=10,
        gradient_accumulation_steps=1,
        warmup_ratio=0.0,
        wandb_project="xray_projection_training",
        save_every_n_epochs=0 # Added: Default 0 means only save at end
    ):
        self.accelerator = accelerator
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projection_layer = projection_layer
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = accelerator.device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.wandb_project = wandb_project
        self.save_every_n_epochs = save_every_n_epochs # Store the value

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        # Optimizer targets only the projection layer parameters
        self.optimizer = optim.AdamW(
            self.projection_layer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Calculate total training steps for the scheduler
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        self.max_train_steps = num_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {self.max_train_steps}")

        # Calculate warmup steps based on ratio
        num_warmup_steps = math.ceil(warmup_ratio * self.max_train_steps)
        logger.info(f"Warmup ratio: {warmup_ratio}, Calculated num_warmup_steps: {num_warmup_steps}")

        # Learning rate scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps, # Use calculated value
            num_training_steps=self.max_train_steps,
        )
        logger.info(f"Initialized cosine LR scheduler with {num_warmup_steps} warmup steps.")

        # Prepare components with Accelerator - SEPARATELY for DeepSpeed
        # Prepare trainable layer, optimizer, dataloader, scheduler together
        self.projection_layer, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.projection_layer, self.optimizer, self.train_loader, self.lr_scheduler
        )
        
        # Prepare frozen models individually using prepare_model for device placement/wrapping only
        self.vision_encoder = self.accelerator.prepare_model(self.vision_encoder, device_placement=True)
        self.language_model = self.accelerator.prepare_model(self.language_model, device_placement=True)

        # --- Freezing Strategy --- (Apply after prepare)
        # Freeze vision encoder and language model completely
        self.vision_encoder.requires_grad_(False)
        self.language_model.requires_grad_(False)
        # Ensure the projection layer is trainable
        self.projection_layer.requires_grad_(True)
        logger.info("Froze vision encoder and language model parameters.")
        logger.info("Projection layer parameters are trainable (Stage 1 Alignment).")

        # Note: Loss function is implicitly handled by the LLM's forward pass when labels are provided
        # self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # Not explicitly needed

    def train(self):
        """Train the projection layer for feature alignment (Stage 1)"""
        logger.info(f"Process {self.accelerator.process_index}: Starting Stage 1 training for {self.num_epochs} epochs on device {self.device}")

        global_step = 0
        best_val_loss = float('inf') # Initialize best validation loss tracking

        for epoch in range(self.num_epochs):
            self.projection_layer.train() # Set projector to train mode
            self.vision_encoder.eval()    # Keep vision encoder in eval mode
            self.language_model.eval()    # Keep LLM in eval mode

            epoch_train_loss = 0.0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} [Stage 1 Train]",
                disable=not self.accelerator.is_main_process
            )

            for batch in progress_bar:
                pixel_values = batch["pixel_values"]
                token_ids = batch["token_ids"] # Get original token IDs
                labels = batch["labels"]       # Get labels with -100

                # === Get Visual Features ===
                with torch.no_grad(): # Ensure no gradients for vision encoder
                    vision_dtype = next(self.vision_encoder.parameters()).dtype
                    try:
                        # Handle DDP wrapping
                        if self.accelerator.num_processes > 1 and hasattr(self.vision_encoder, 'module'):
                            vision_tower = self.vision_encoder.module.vision_model
                        else:
                            vision_tower = self.vision_encoder.vision_model

                        vision_outputs = vision_tower(
                            pixel_values=pixel_values.to(vision_dtype),
                            output_hidden_states=False,
                            return_dict=True
                        )
                        # Assuming SigLIP-like output, discard CLS token embedding
                        patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                    except Exception as e:
                        logger.error(f"Error getting vision embeddings: {e}", exc_info=True)
                        continue # Skip batch on error

                # === Project Visual Features ===
                # This is the part being trained
                projected_embeds = self.projection_layer(patch_embeddings)

                # === Prepare LLM Inputs ===
                with torch.no_grad(): # Ensure no gradients for LLM embedding layer
                     # Handle DDP wrapping
                    if self.accelerator.num_processes > 1 and hasattr(self.language_model, 'module'):
                       word_embeddings_module = self.language_model.module.get_input_embeddings()
                    else:
                       word_embeddings_module = self.language_model.get_input_embeddings()

                    # --- Get embeddings using original token_ids (MUST be non-negative) --- 
                    label_embeds = word_embeddings_module(token_ids) # Shape: [B, SeqLen_Text, Dim_LLM]

                # Concatenate projected visual embeddings and target caption embeddings
                # Shape: [B, SeqLen_Vision + SeqLen_Text, Dim_LLM]
                combined_embeds = torch.cat([projected_embeds, label_embeds], dim=1)

                # === Prepare Labels and Attention Mask for LLM ===
                num_visual_tokens = projected_embeds.size(1)

                # Attention mask: 1s for visual tokens, 1s for non-padding caption tokens, 0s for padding
                visual_attention_mask = torch.ones(
                    (labels.shape[0], num_visual_tokens),
                    dtype=torch.long, device=self.device
                )
                # Mask for text tokens (1 if not padding, 0 if padding)
                # Use token_ids here as labels might have -100 where pad token was
                text_attention_mask = (token_ids != self.tokenizer.pad_token_id).long()
                if self.tokenizer.pad_token_id is None: # Handle cases without explicit pad token
                     text_attention_mask = torch.ones_like(token_ids, dtype=torch.long, device=self.device)

                # Combine masks: [B, SeqLen_Vision + SeqLen_Text]
                extended_attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)

                # Labels for LM loss: -100 for visual tokens, actual token IDs for caption tokens
                visual_labels = torch.full(
                     (labels.shape[0], num_visual_tokens), fill_value=-100, dtype=labels.dtype, device=self.device
                )
                # Combine labels: [B, SeqLen_Vision + SeqLen_Text]
                # Use the 'labels' tensor which already has -100 for padding
                lm_labels = torch.cat([visual_labels, labels], dim=1)


                # === Forward Pass through Frozen LLM ===
                # Calculate loss for predicting caption tokens based on visual input
                # Gradients will only flow back to the projector layer
                outputs = self.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=extended_attention_mask,
                    labels=lm_labels, # Use the combined labels with -100 for visual tokens
                    return_dict=True,
                    output_hidden_states=False
                )
                loss = outputs.loss # This is the causal LM loss

                # --- Backpropagation (updates only projector) ---
                scaled_loss = loss / self.accelerator.gradient_accumulation_steps
                self.accelerator.backward(scaled_loss)

                # --- Optimizer Step ---
                if self.accelerator.sync_gradients:
                    # Optional gradient clipping (uncomment if needed)
                    # TODO: Experiment with gradient clipping if training becomes unstable.
                    self.accelerator.clip_grad_norm_(self.projection_layer.parameters(), 5.0) # Enabled gradient clipping
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # --- Accumulate loss ONLY on optimizer steps ---
                    # Gather loss across all GPUs for the batches contributing to this step
                    # Note: We still use the 'loss' from the last micro-batch for logging simplicity,
                    #       a more precise average would require storing/averaging losses over accumulation steps.
                    avg_loss_step = self.accelerator.gather(loss).mean().item() 
                    epoch_train_loss += avg_loss_step

                # --- Logging and Metric Accumulation (Log status after each micro-batch) ---
                # Gather loss across all GPUs for the current micro-batch for logging purposes
                current_avg_micro_batch_loss = self.accelerator.gather(loss).mean().item()

                if self.accelerator.is_main_process and self.accelerator.sync_gradients: # Log only when gradients are synced
                    current_step_for_logging = global_step # Use global_step AFTER incrementing
                    log_dict = {
                        # Log the loss from the specific step that just completed
                        "train/batch_loss": avg_loss_step, 
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step": current_step_for_logging
                    }
                    self.accelerator.log(log_dict, step=current_step_for_logging)
                    # Show the current micro-batch loss in progress bar for more frequent updates
                    progress_bar.set_postfix({"micro_loss": current_avg_micro_batch_loss, "lr": self.lr_scheduler.get_last_lr()[0]})
                elif self.accelerator.is_main_process:
                    # Update progress bar even if not logging step, showing micro-batch loss
                    progress_bar.set_postfix({"micro_loss": current_avg_micro_batch_loss, "lr": self.lr_scheduler.get_last_lr()[0]})

            # --- End of Epoch ---
            # Calculate average epoch loss based on optimizer steps
            num_optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
            avg_epoch_train_loss = epoch_train_loss / num_optimizer_steps_per_epoch

            # --- Epoch Logging ---
            if self.accelerator.is_main_process:
                epoch_log_dict = {
                    "train/epoch_loss": avg_epoch_train_loss,
                    "epoch": epoch + 1
                }
                self.accelerator.log(epoch_log_dict, step=global_step) # Log against final step of epoch
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed. Avg Train Loss: {avg_epoch_train_loss:.4f}")

                # --- Periodic Saving --- # Added
                if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_projection(epoch=epoch+1)

            # --- Validation Step (on main process only) ---
            if self.accelerator.is_main_process:
                logger.info(f"Running validation inference for Epoch {epoch+1}...")
                self.projection_layer.eval()
                self.language_model.eval() # Ensure LLM is in eval for generation
                # Vision encoder should already be in eval mode

                val_image_path = "/home/compu/samuel/Siglip/images/001.png"
                max_new_tokens_val = 128 # Limit generated tokens for validation output

                try:
                    with torch.no_grad():
                        # --- Load and Process Validation Image ---
                        image = Image.open(val_image_path).convert('RGB')
                        if hasattr(self.processor, 'size'):
                            img_size = self.processor.size.get("shortest_edge", self.processor.size.get("height", 384))
                        else:
                            logger.warning("Cannot determine processor size reliably, assuming 384x384 for validation.")
                            img_size = 384
                        image = image.resize((img_size, img_size))
                        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                        pixel_values_val = image_inputs.pixel_values

                        # --- Validation Inference (Generate Text) ---
                        vision_dtype = next(self.vision_encoder.parameters()).dtype
                        if hasattr(self.vision_encoder, 'module'):
                             vision_tower = self.vision_encoder.module.vision_model
                        else:
                            vision_tower = self.vision_encoder.vision_model
                        vision_outputs = vision_tower(
                            pixel_values=pixel_values_val.to(vision_dtype),
                            output_hidden_states=False, return_dict=True
                        )
                        patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                        projected_embeds = self.projection_layer(patch_embeddings)

                        # Prepare attention mask for generation (all ones for visual input)
                        inputs_embeds = projected_embeds
                        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

                        # Generate text using LLM
                        # Use simple generation parameters for validation output
                        outputs = self.language_model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens_val,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            do_sample=False # Use greedy decoding for consistent validation output
                        )
                        
                        # Decode generated tokens (generate returns only new tokens when using inputs_embeds)
                        generated_ids = outputs[0] 
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                        # Log the generated text
                        logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Validation Image ({os.path.basename(val_image_path)}) Generated Text:\n{generated_text}")
                        
                        # Log to WandB if enabled
                        if self.accelerator.trackers:
                             try:
                                 # Log as text to WandB table
                                 val_table_data = [[epoch+1, global_step, os.path.basename(val_image_path), generated_text]]
                                 # Check if tracker exists before logging
                                 if "wandb" in self.accelerator.trackers:
                                     wandb_tracker = self.accelerator.get_tracker("wandb")
                                     if wandb_tracker is not None:
                                         # Ensure wandb is imported if using wandb directly
                                         import wandb 
                                         val_table = wandb.Table(columns=["Epoch", "Step", "Image", "Generated Text"], data=val_table_data)
                                         wandb_tracker.log({"validation_generations": val_table})
                                 else:
                                     logger.warning("Wandb tracker not found, cannot log validation table.")
                                
                             except Exception as log_e:
                                 logger.warning(f"Could not log validation text to WandB: {log_e}")

                except FileNotFoundError:
                     logger.error(f"Validation image not found at {val_image_path}. Skipping validation.")
                except Exception as e:
                     logger.error(f"Error during validation inference step: {e}", exc_info=True)
                finally:
                    # Ensure model is back in training mode
                    self.projection_layer.train()

        # --- End of Training --- # (Save final model)
        logger.info("Stage 1 Training finished.")
        self.save_projection(epoch=self.num_epochs) # Save final projector
        logger.info(f"Final trained projector saved to {self.output_dir}")

    def save_projection(self, epoch=None):
        """Save the trained projection layer state dictionary."""
        # Simplified save approach that avoids wait_for_everyone() and long timeouts
        
        # Only the main process saves the model
        if self.accelerator.is_main_process:
            # Allow a short pause (1-2 seconds) for other processes to sync up naturally
            # This avoids immediate saving while other processes might still be finishing tasks
            time.sleep(2)
            
            # Unwrap the model to get the raw nn.Module for saving
            unwrapped_projection_layer = self.accelerator.unwrap_model(self.projection_layer)
            
            # Determine filename
            if epoch is not None:
                save_filename = f"projector_epoch_{epoch}.bin"
            else:
                save_filename = "projector_final.bin"
                
            save_path = os.path.join(self.output_dir, save_filename)
            
            # Save the state dict
            try:
                torch.save(unwrapped_projection_layer.state_dict(), save_path)
                logger.info(f"Projection layer state dict saved to {save_path}")
                
                # --- Manual Config Saving (Robustness) ---
                try:
                    # --- Updated logic for nn.Sequential MLP ---
                    if hasattr(unwrapped_projection_layer, 'model') and isinstance(unwrapped_projection_layer.model, nn.Sequential):
                        seq_model = unwrapped_projection_layer.model
                        if len(seq_model) > 0 and isinstance(seq_model[0], nn.Linear):
                            # Get input dim from first linear layer
                            vision_dim = seq_model[0].in_features
                            # Get output dim from last linear layer
                            last_linear_layer = None
                            for layer in reversed(seq_model):
                                if isinstance(layer, nn.Linear):
                                    last_linear_layer = layer
                                    break
                            if last_linear_layer:
                                llm_dim = last_linear_layer.out_features
                                logger.info(f"Determined MLP dims: vision_dim={vision_dim}, llm_dim={llm_dim}")
                            else:
                                logger.warning(f"Could not find last Linear layer in unwrapped_model.model. Using fallbacks.")
                                vision_dim, llm_dim = 0, 0 # Fallback
                        else:
                            logger.warning(f"Unwrapped model's '.model' attribute is not a Sequential or doesn't start with Linear. Using fallbacks.")
                            vision_dim, llm_dim = 0, 0 # Fallback
                    else: # Fallback if structure is unexpected
                        logger.warning(f"Could not reliably determine projector dimensions from unwrapped model structure type: {type(unwrapped_projection_layer)}. Using fallbacks.")
                        # Attempt to get from config if available, otherwise use 0
                        vision_dim = getattr(unwrapped_projection_layer.config, 'vision_dim', 0) if hasattr(unwrapped_projection_layer, 'config') else 0
                        llm_dim = getattr(unwrapped_projection_layer.config, 'llm_dim', 0) if hasattr(unwrapped_projection_layer, 'config') else 0
                
                    # Config no longer includes num_tokens for MLP projector
                    config = { "vision_dim": vision_dim, "llm_dim": llm_dim }
                    # Save config to the output directory, not inside the checkpoint file path
                    config_save_path = os.path.join(self.output_dir, "projector_config.json")
                    with open(config_save_path, 'w') as f:
                        json.dump(config, f, indent=4)
                    logger.info(f"Manual projector config saved to {config_save_path}")
                
                except Exception as e:
                    logger.error(f"Error saving config: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Error saving model: {e}", exc_info=True) 