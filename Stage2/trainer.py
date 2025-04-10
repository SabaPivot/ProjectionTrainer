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
from torch.nn.utils.rnn import pad_sequence
from functools import partial

logger = logging.getLogger(__name__)

# --- Custom Collate Function ---
def vqa_collate_fn(batch, pad_token_id):
    """Collate function for VQADataset.

    Pads question_input_ids and answer_input_ids to the max length in the batch. ==> Dynamic padding
    
    Pros: 
    - Computational Efficiency
    - Memory Efficiency

    Stacks pixel_values.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Pad question_input_ids
    question_input_ids_list = [torch.tensor(item["question_input_ids"], dtype=torch.long) for item in batch]
    question_input_ids_padded = pad_sequence(
        question_input_ids_list,
        batch_first=True,
        padding_value=pad_token_id
    )

    # Pad answer_input_ids
    answer_input_ids_list = [torch.tensor(item["answer_input_ids"], dtype=torch.long) for item in batch]
    answer_input_ids_padded = pad_sequence(
        answer_input_ids_list,
        batch_first=True,
        padding_value=pad_token_id
    )

    return {
        "pixel_values": pixel_values,
        "question_input_ids": question_input_ids_padded,
        "answer_input_ids": answer_input_ids_padded,
    }

class VQATrainerStage2:
    """Trainer for Stage 2 VQA fine-tuning.

    This stage fine-tunes the LLM (and potentially the projection layer)
    while keeping the vision encoder frozen.
    It uses image, question, and answer triplets.
    """
    def __init__(
        self,
        accelerator,
        vision_encoder,
        language_model,
        projection_layer,
        tokenizer,
        train_dataset,
        output_dir: str,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        freeze_vision_encoder: bool,
        freeze_projection_layer: bool,
        freeze_llm: bool,
        train_ve_first_epoch: bool,
        wandb_project: str
    ):
        self.accelerator = accelerator
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projection_layer = projection_layer
        self.tokenizer = tokenizer
        self.device = accelerator.device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.wandb_project = wandb_project
        self.train_ve_first_epoch = train_ve_first_epoch

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        # --- Use functools.partial to pass pad_token_id to collate_fn ---
        # collator in DataLoader receives only the batch argument
        collate_fn_partial = partial(vqa_collate_fn, pad_token_id=self.tokenizer.pad_token_id)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_partial
        )

        # --- Setup Trainable Parameters and Optimizer ---
        trainable_params = self._setup_trainable_parameters(
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_projection_layer=freeze_projection_layer,
            freeze_llm=freeze_llm
        )
        self.optimizer = optim.AdamW(
            trainable_params,
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
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        logger.info(f"Initialized cosine LR scheduler with {num_warmup_steps} warmup steps.")

        # Prepare components with Accelerator
        # Order matters for requires_grad flags to be respected by prepare
        if freeze_llm:
            self.language_model = self.accelerator.prepare_model(self.language_model, device_placement=True)
        if freeze_projection_layer:
            self.projection_layer = self.accelerator.prepare_model(self.projection_layer, device_placement=True)
        if freeze_vision_encoder:
            self.vision_encoder = self.accelerator.prepare_model(self.vision_encoder, device_placement=True)

        # Prepare trainable models and optimizer/dataloaders/scheduler
        if not freeze_llm:
             self.language_model = self.accelerator.prepare(self.language_model)
        if not freeze_projection_layer:
             self.projection_layer = self.accelerator.prepare(self.projection_layer)
        if not freeze_vision_encoder:
             self.vision_encoder = self.accelerator.prepare(self.vision_encoder)

        self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.train_loader, self.lr_scheduler
        )

    def _setup_trainable_parameters(self, freeze_vision_encoder: bool, freeze_projection_layer: bool, freeze_llm: bool) -> list:
        """Applies freezing strategy and collects parameters for the optimizer."""
        models = {
            'vision_encoder': (self.vision_encoder, freeze_vision_encoder),
            'projection_layer': (self.projection_layer, freeze_projection_layer),
            'language_model': (self.language_model, freeze_llm)
        }

        trainable_params = []
        for name, (model, should_freeze) in models.items():
            if should_freeze:
                model.requires_grad_(False)
                logger.info(f"Freezing {name} parameters.")
            else:
                model.requires_grad_(True)
                logger.info(f"{name} parameters are trainable.")
                # Collect parameters only if the model is not frozen
                trainable_params.extend(list(filter(lambda p: p.requires_grad, model.parameters())))

        if not trainable_params:
            raise ValueError("No trainable parameters found. Check freezing configuration.")

        logger.info(f"Collected {len(trainable_params)} trainable parameters for the optimizer.")
        return trainable_params

    def train(self):
        logger.info(f"Process {self.accelerator.process_index}: Starting Stage 2 training for {self.num_epochs} epochs on device {self.device}")

        global_step = 0

        for epoch in range(self.num_epochs):
            # Set modes: Train for trainable components, Eval for frozen ones
            # Unwrap model to check requires_grad of its parameters
            llm_unwrapped = self.accelerator.unwrap_model(self.language_model)
            # Check requires_grad on the parameters
            is_llm_trainable = any(p.requires_grad for p in llm_unwrapped.parameters())
            self.language_model.train(mode=is_llm_trainable)

            proj_unwrapped = self.accelerator.unwrap_model(self.projection_layer)
            is_proj_trainable = any(p.requires_grad for p in proj_unwrapped.parameters())
            self.projection_layer.train(mode=is_proj_trainable)

            ve_unwrapped = self.accelerator.unwrap_model(self.vision_encoder)

            # --- Dynamic Vision Encoder Freezing --- #
            # Train VE only during the first epoch, freeze afterwards, following CheXagent paper
            is_first_epoch = (epoch == 0)
            should_attempt_ve_train = self.train_ve_first_epoch and is_first_epoch
            # Check if VE params are actually in the optimizer (depends on initial freeze_vision_encoder flag)
            ve_params_in_optimizer = any(id(p) in {id(opt_p) for opt_p in self.optimizer.param_groups[0]["params"]} for p in ve_unwrapped.parameters())
            
            if should_attempt_ve_train and ve_params_in_optimizer:
                logger.info(f"Epoch {epoch+1}: Unfreezing Vision Encoder.")
                self.vision_encoder.requires_grad_(True)
                self.vision_encoder.train()
            else:
                # Log state change or reason for not training VE
                if is_first_epoch and not self.train_ve_first_epoch:
                    logger.info(f"Epoch {epoch+1}: --train_ve_first_epoch=False. Keeping Vision Encoder frozen.")
                elif is_first_epoch and not ve_params_in_optimizer:
                    logger.warning(f"Epoch {epoch+1}: Cannot unfreeze Vision Encoder - parameters not found in optimizer (was freeze_vision_encoder=True initially?). Keeping frozen.")
                elif epoch == 1 and ve_params_in_optimizer and self.train_ve_first_epoch: # Log only once when freezing happens after first epoch
                    logger.info(f"Epoch {epoch+1}: Freezing Vision Encoder for subsequent epochs.")
                    
                self.vision_encoder.requires_grad_(False)
                self.vision_encoder.eval()
            # ----------------------------------------- #

            epoch_train_loss = 0.0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} [Stage 2 Train]",
                disable=not self.accelerator.is_main_process
            )

            for batch in progress_bar:
                # Accumulate gradients only for trainable models
                # Need to determine the top-level model to pass to accumulate
                # Check requires_grad on the *parameters* of the unwrapped model
                llm_is_trainable = any(p.requires_grad for p in self.accelerator.unwrap_model(self.language_model).parameters())
                model_to_accumulate = self.language_model if llm_is_trainable else self.projection_layer

                with self.accelerator.accumulate(model_to_accumulate):
                    pixel_values = batch["pixel_values"]
                    question_input_ids = batch["question_input_ids"] # Unpadded question tokens
                    answer_input_ids = batch["answer_input_ids"] # Padded answer tokens

                    # === Get Visual Features (Vision Encoder: Trainable only on epoch 1) ===
                    # Use torch.set_grad_enabled for conditional gradient computation
                    with torch.set_grad_enabled(self.vision_encoder.training):
                        vision_dtype = next(self.vision_encoder.parameters()).dtype
                        # Handle DDP wrapping
                        if hasattr(self.vision_encoder, 'module'):
                            vision_tower = self.vision_encoder.module.vision_model
                        else:
                            vision_tower = self.vision_encoder.vision_model
                        try:
                            vision_outputs = vision_tower(
                                pixel_values=pixel_values.to(vision_dtype),
                                output_hidden_states=False,
                                return_dict=True
                            )
                            # Discard CLS token
                            # [batch_size, sequence_length, hidden_size]
                            # Aligns with the Stage 1 training script
                            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                        except Exception as e:
                            logger.error(f"Error getting vision embeddings: {e}", exc_info=True)
                            continue

                    # === Project Visual Features (Potentially Trainable) ===
                    # Use torch.set_grad_enabled for conditional gradient computation
                    with torch.set_grad_enabled(self.projection_layer.training):
                        projected_embeds = self.projection_layer(patch_embeddings)

                    # === Prepare LLM Inputs (Embeddings) ===
                    # Embed Question and Answer tokens using LLM's embedding layer
                    # Handle DDP wrapping for LLM
                    llm_model_unwrapped = self.accelerator.unwrap_model(self.language_model)
                    input_embed_layer = llm_model_unwrapped.get_input_embeddings()

                    # Special handling for Gemma3 model to prevent in-place operations
                    # Generated by gemini-2.5-flash
                    if 'gemma3' in llm_model_unwrapped.__class__.__name__.lower():
                        # Ensure requires_grad is False for embedding lookup if LLM is frozen, but computation can still happen
                        with torch.set_grad_enabled(self.language_model.training):
                            # Manual embedding to avoid the in-place .to() operation in Gemma3 embedding layer
                            question_embeds = input_embed_layer.weight[question_input_ids]
                            if hasattr(input_embed_layer, 'embed_scale'):
                                # Create a non-inplace copy of the embed_scale 
                                embed_scale = input_embed_layer.embed_scale.clone().detach().requires_grad_(input_embed_layer.embed_scale.requires_grad)
                                question_embeds = question_embeds * embed_scale
                            
                            answer_embeds = input_embed_layer.weight[answer_input_ids]
                            if hasattr(input_embed_layer, 'embed_scale'):
                                # Use the same non-inplace copy
                                answer_embeds = answer_embeds * embed_scale
                    else:
                        # Original code for other model types
                        with torch.set_grad_enabled(self.language_model.training):
                            question_embeds = input_embed_layer(question_input_ids)
                            answer_embeds = input_embed_layer(answer_input_ids)
                            # Shapes: [B, SeqLen_Q, Dim_LLM], [B, SeqLen_A, Dim_LLM]

                    # === Concatenate Inputs ===
                    # [B, NumVisual + SeqLen_Q + SeqLen_A, Dim_LLM]
                    inputs_embeds = torch.cat([projected_embeds, question_embeds, answer_embeds], dim=1)

                    # === Prepare Labels and Attention Mask ===
                    batch_size = projected_embeds.shape[0]
                    num_visual_tokens = projected_embeds.shape[1]
                    q_len = question_input_ids.shape[1]

                    # --- Attention Mask --- #
                    # Mask: 1 for visual, 1 for question, 1 for non-pad answer, 0 for pad answer
                    visual_attn_mask = torch.ones((batch_size, num_visual_tokens), dtype=torch.long, device=self.device)
                    # Question mask depends on padding (dynamically padded)
                    question_attn_mask = (question_input_ids != self.tokenizer.pad_token_id).long()
                    # Answer mask depends on padding
                    answer_attn_mask = (answer_input_ids != self.tokenizer.pad_token_id).long()

                    attention_mask = torch.cat([visual_attn_mask, question_attn_mask, answer_attn_mask], dim=1)

                    # --- Labels --- #
                    # Loss is calculated only on the answer tokens.
                    # Labels: -100 for visual, -100 for question, actual token_id for answer (or -100 if padding)
                    visual_labels = torch.full((batch_size, num_visual_tokens), fill_value=-100, dtype=torch.long, device=self.device)
                    question_labels = torch.full((batch_size, q_len), fill_value=-100, dtype=torch.long, device=self.device)
                    # Answer labels: Use answer_input_ids, but replace pad_token_id with -100
                    answer_labels = answer_input_ids.clone()
                    answer_labels[answer_labels == self.tokenizer.pad_token_id] = -100

                    labels = torch.cat([visual_labels, question_labels, answer_labels], dim=1)

                    # === Forward Pass through LLM ===
                    # Gradients flow based on requires_grad settings of LLM and Projector
                    outputs = self.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=None, # Don't pass labels here, calculate loss manually
                        return_dict=True,
                        output_hidden_states=False
                    )
                    # --- Cast logits to float32 for stable loss calculation ---
                    logits = outputs.logits.to(torch.float32)

                    # --- Calculate loss manually ---
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Use PyTorch's CrossEntropyLoss
                    loss_fct = nn.CrossEntropyLoss()
                    # Flatten the tokens
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # -----------------------------

                    # --- Backpropagation --- #
                    scaled_loss = loss / self.accelerator.gradient_accumulation_steps
                    self.accelerator.backward(scaled_loss)

                    # --- Optimizer Step --- #
                    if self.accelerator.sync_gradients:
                        # Clip gradients for the trainable parameters
                        if any(p.requires_grad for p in self.language_model.parameters()):
                            self.accelerator.clip_grad_norm_(self.language_model.parameters(), 1.0)
                        if any(p.requires_grad for p in self.projection_layer.parameters()):
                            self.accelerator.clip_grad_norm_(self.projection_layer.parameters(), 1.0)
                        # Add vision encoder clipping if it's being trained
                        if any(p.requires_grad for p in self.vision_encoder.parameters()):
                             self.accelerator.clip_grad_norm_(self.vision_encoder.parameters(), 1.0)

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1

                        avg_loss_step = self.accelerator.gather(loss).mean().item()
                        epoch_train_loss += avg_loss_step

                # --- Logging --- #
                current_avg_micro_batch_loss = self.accelerator.gather(loss).mean().item()
                if self.accelerator.is_main_process:
                     log_dict = {
                         "train/batch_loss": current_avg_micro_batch_loss, # Log micro-batch loss
                         "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                         "step": global_step
                     }
                     # Only log step-related metrics when gradients are synced and optimizer step occurs
                     if self.accelerator.sync_gradients:
                         log_dict["train/step_loss"] = avg_loss_step # Loss averaged over accumulation steps
                         self.accelerator.log(log_dict, step=global_step)
                     else: # Log only micro-batch loss if not an optimizer step
                         self.accelerator.log({"train/batch_loss": current_avg_micro_batch_loss}, step=global_step)

                     progress_bar.set_postfix({"micro_loss": current_avg_micro_batch_loss, "lr": self.lr_scheduler.get_last_lr()[0]}) 
                     

            # --- End of Epoch --- #
            num_optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
            avg_epoch_train_loss = epoch_train_loss / num_optimizer_steps_per_epoch if num_optimizer_steps_per_epoch > 0 else 0.0

            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Avg Train Loss: {avg_epoch_train_loss:.4f}")
                epoch_log_dict = {
                     "train/epoch_loss": avg_epoch_train_loss,
                     "epoch": epoch + 1
                 }
                self.accelerator.log(epoch_log_dict, step=global_step)

            if self.accelerator.is_main_process:
                 if (epoch + 1) % 2 == 0 or (epoch + 1) == self.num_epochs: # Save every 2 epochs and final
                     save_path = os.path.join(self.output_dir, f"checkpoint-epoch_{epoch+1}")
                     self.save_model(save_path)

        # --- Final Save --- #
        if self.accelerator.is_main_process:
            final_save_path = os.path.join(self.output_dir, "final_model")
            self.save_model(final_save_path)
            logger.info("Final Stage 2 model saved.")

        logger.info(f"Process {self.accelerator.process_index}: Stage 2 training complete!")

    def save_model(self, path):
        """Saves the trainable components (LLM, Projector) (main process only)"""
        if not self.accelerator.is_main_process:
            return

        os.makedirs(path, exist_ok=True)

        # Save components using accelerator.save_state
        # This saves optimizer, scheduler, and potentially other states managed by accelerator
        # It also saves the models handled by accelerator.prepare
        self.accelerator.save_state(path)
        logger.info(f"Full training state (models, optimizer, scheduler) saved to {path} using accelerator.save_state")

        # Additionally, save model weights separately for easier loading in inference
        # --- Save Language Model (if trained) --- #
        unwrapped_llm = self.accelerator.unwrap_model(self.language_model)
        if any(p.requires_grad for p in unwrapped_llm.parameters()):
             llm_save_path = os.path.join(path, "language_model")
             self.accelerator.save_model(unwrapped_llm, llm_save_path)
             # unwrapped_llm.save_pretrained(llm_save_path) # Use accelerator's method
             self.tokenizer.save_pretrained(llm_save_path) # Save tokenizer alongside model
             logger.info(f"Language model saved to {llm_save_path}")

        # --- Save Projection Layer (if trained) --- #
        unwrapped_proj = self.accelerator.unwrap_model(self.projection_layer)
        if any(p.requires_grad for p in unwrapped_proj.parameters()):
            proj_save_path = os.path.join(path, "projection_layer")
            self.accelerator.save_model(unwrapped_proj, proj_save_path)
            logger.info(f"Projection layer saved to {proj_save_path}")
            # Save projector config manually if needed (copy logic from Stage 1)
            try:
                 if hasattr(unwrapped_proj, 'model') and isinstance(unwrapped_proj.model, nn.Sequential):
                     seq_model = unwrapped_proj.model
                     if len(seq_model) > 0 and isinstance(seq_model[0], nn.Linear):
                         vision_dim = seq_model[0].in_features
                         last_linear_layer = None
                         for layer in reversed(seq_model):
                            if isinstance(layer, nn.Linear):
                                last_linear_layer = layer
                                break
                         llm_dim = last_linear_layer.out_features if last_linear_layer else 0
                     else: vision_dim, llm_dim = 0,0
                 else: vision_dim, llm_dim = 0,0
                 config = { "vision_dim": vision_dim, "llm_dim": llm_dim }
                 config_save_path = os.path.join(proj_save_path, "projector_config.json")
                 with open(config_save_path, 'w') as f: json.dump(config, f, indent=4)
                 logger.info(f"Manual projector config saved to {config_save_path}")
            except Exception as e:
                logger.error(f"Could not save manual projector config: {e}") 