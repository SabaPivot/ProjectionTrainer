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
from accelerate.utils import gather_object

logger = logging.getLogger(__name__)

# --- Custom Collate Function ---
def vqa_collate_fn(batch, tokenizer):
    """Collate function for VQADataset.

    Pads question_input_ids and answer_input_ids based on tokenizer.padding_side.
    Stacks pixel_values.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pad_token_id = tokenizer.pad_token_id
    padding_side = tokenizer.padding_side

    question_input_ids_list = [item["question_input_ids"] for item in batch]
    answer_input_ids_list = [item["answer_input_ids"] for item in batch]

    # --- Manual Padding based on tokenizer.padding_side ---
    def manual_pad(sequences, max_len):
        padded_sequences = []
        for seq_tensor in sequences:
            seq_len = seq_tensor.shape[0]
            pad_len = max_len - seq_len
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_token_id, dtype=seq_tensor.dtype, device=seq_tensor.device)
                if padding_side == 'left':
                    padded_seq = torch.cat([padding, seq_tensor], dim=0)
                else: # Default to right padding
                    padded_seq = torch.cat([seq_tensor, padding], dim=0)
            else:
                padded_seq = seq_tensor
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)

    # Determine max lengths
    max_q_len = max(seq.shape[0] for seq in question_input_ids_list)
    max_a_len = max(seq.shape[0] for seq in answer_input_ids_list)

    # Apply manual padding
    question_input_ids_padded = manual_pad(question_input_ids_list, max_q_len)
    answer_input_ids_padded = manual_pad(answer_input_ids_list, max_a_len)
    # --------------------------------------------------------

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
        val_dataset,
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
        enable_qlora: bool,
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
        self.enable_qlora = enable_qlora
        
        # Define validation_dir for all processes
        self.validation_dir = os.path.join(output_dir, "validation_examples")
        
        # Only main process creates the directory
        if self.accelerator.is_main_process:
            os.makedirs(self.validation_dir, exist_ok=True)
            logger.info(f"Created validation directory at {self.validation_dir}")

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        # --- Use functools.partial to pass the tokenizer to collate_fn ---
        collate_fn_partial = partial(vqa_collate_fn, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_partial
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_partial
        )

        # --- Setup Trainable Parameters and Optimizer ---
        trainable_params = self._setup_trainable_parameters(
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_projection_layer=freeze_projection_layer,
            freeze_llm=freeze_llm,
            enable_qlora=self.enable_qlora
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

        self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )

    def _setup_trainable_parameters(self, freeze_vision_encoder: bool, freeze_projection_layer: bool, freeze_llm: bool, enable_qlora: bool) -> list:
        """Applies freezing strategy and collects parameters for the optimizer."""
        
        trainable_params = []
        
        # Handle LLM parameters based on QLoRA flag
        if enable_qlora:
            logger.info("QLoRA enabled: Only LLM adapter parameters will be trained.")
            # PEFT model automatically handles freezing base parameters.
            # We collect all parameters flagged as requires_grad from the PeftModel.
            llm_params = list(filter(lambda p: p.requires_grad, self.language_model.parameters()))
            trainable_params.extend(llm_params)
            logger.info(f"Collected {len(llm_params)} trainable parameters from QLoRA LLM.")
        else:
            # Standard freezing logic for LLM if QLoRA is not enabled
            if freeze_llm:
                self.language_model.requires_grad_(False)
                logger.info("Freezing LLM parameters (QLoRA disabled).")
            else:
                self.language_model.requires_grad_(True)
                logger.info("LLM parameters are trainable (QLoRA disabled).")
                llm_params = list(filter(lambda p: p.requires_grad, self.language_model.parameters()))
                trainable_params.extend(llm_params)
                logger.info(f"Collected {len(llm_params)} trainable parameters from LLM.")

        # Handle Projector Layer freezing (independent of QLoRA)
        if freeze_projection_layer:
            self.projection_layer.requires_grad_(False)
            logger.info("Freezing projection_layer parameters.")
        else:
            self.projection_layer.requires_grad_(True)
            logger.info("projection_layer parameters are trainable.")
            proj_params = list(filter(lambda p: p.requires_grad, self.projection_layer.parameters()))
            trainable_params.extend(proj_params)
            logger.info(f"Collected {len(proj_params)} trainable parameters from projection_layer.")

        # Handle Vision Encoder freezing (independent of QLoRA)
        # Note: train_ve_first_epoch logic happens dynamically in train() loop
        if freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
            logger.info("Initially freezing vision_encoder parameters (may unfreeze in epoch 1).")
        else:
            self.vision_encoder.requires_grad_(True)
            logger.info("vision_encoder parameters are initially trainable.")
            # Only add VE params if not initially frozen, dynamic freezing handles epoch 1
            if not self.train_ve_first_epoch: # Add if not training dynamically in epoch 1
                ve_params = list(filter(lambda p: p.requires_grad, self.vision_encoder.parameters()))
                trainable_params.extend(ve_params)
                logger.info(f"Collected {len(ve_params)} trainable parameters from vision_encoder.")
            else:
                logger.info("Vision encoder params handled dynamically in train() loop due to --train_ve_first_epoch.")


        if not trainable_params:
            raise ValueError("No trainable parameters found. Check freezing configuration and QLoRA setup.")

        logger.info(f"Collected {len(trainable_params)} total trainable parameters for the optimizer.")
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
                        trainable_modules = []
                        if any(p.requires_grad for p in self.accelerator.unwrap_model(self.language_model).parameters()):
                             trainable_modules.append(self.language_model)
                        if any(p.requires_grad for p in self.accelerator.unwrap_model(self.projection_layer).parameters()):
                             trainable_modules.append(self.projection_layer)
                        if any(p.requires_grad for p in self.accelerator.unwrap_model(self.vision_encoder).parameters()):
                             trainable_modules.append(self.vision_encoder)
                             
                        # Clip gradients for all trainable modules prepared by accelerator
                        for module in trainable_modules:
                             if module is not None and hasattr(module, 'parameters'):
                                 self.accelerator.clip_grad_norm_(module.parameters(), 1.0)
                        
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
                     

            # Calculate and log epoch metrics 
            avg_epoch_loss = epoch_train_loss / len(self.train_loader)
            
            # Log training loss
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
                self.accelerator.log({
                    "train/loss": avg_epoch_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1
                }, step=global_step)
            
            # Run validation after each epoch
            self.evaluate(epoch, global_step)

            # --- Save Checkpoint --- #
            if self.accelerator.is_main_process:
                 # Always save at the end of each epoch
                 save_path = os.path.join(self.output_dir, f"checkpoint-epoch_{epoch+1}")
                 self.save_model(save_path)

        logger.info(f"Process {self.accelerator.process_index}: Stage 2 training complete!")

    def evaluate(self, epoch, global_step):
        """Runs evaluation on the validation set, calculates loss and saves examples to file.

        Args:
            epoch (int): Current epoch number.
            global_step (int): Current global training step.
        """
        logger.info(f"Starting evaluation for Epoch {epoch+1}...")
        
        # Store original padding side and set to left for evaluation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        # Ensure the pad token is set if it's None (important for left padding)
        if self.tokenizer.pad_token_id is None:
            logger.warning("Tokenizer pad_token_id is None. Setting to eos_token_id for evaluation.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.language_model.eval()
        self.projection_layer.eval()
        self.vision_encoder.eval()

        total_val_loss = 0.0

        # For collecting validation examples
        all_examples = []
        max_examples_to_log = min(20, len(self.val_loader.dataset)) # Limit examples for performance

        # Process entire validation set
        with torch.no_grad():
             for batch_idx, batch in enumerate(self.val_loader):
                pixel_values = batch["pixel_values"]
                question_input_ids = batch["question_input_ids"]
                answer_input_ids = batch["answer_input_ids"]

                # === Get Visual Features ===
                vision_dtype = next(self.vision_encoder.parameters()).dtype
                if hasattr(self.vision_encoder, 'module'):
                    vision_tower = self.vision_encoder.module.vision_model
                else:
                    vision_tower = self.vision_encoder.vision_model
                vision_outputs = vision_tower(
                    pixel_values=pixel_values.to(vision_dtype),
                    output_hidden_states=False,
                    return_dict=True
                )
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]

                # === Project Visual Features ===
                projected_embeds = self.projection_layer(patch_embeddings)

                # === Prepare LLM Inputs (Embeddings for Loss Calculation) ===
                llm_model_unwrapped = self.accelerator.unwrap_model(self.language_model)
                input_embed_layer = llm_model_unwrapped.get_input_embeddings()

                if 'gemma3' in llm_model_unwrapped.__class__.__name__.lower():
                    question_embeds = input_embed_layer.weight[question_input_ids]
                    answer_embeds = input_embed_layer.weight[answer_input_ids]
                    if hasattr(input_embed_layer, 'embed_scale'):
                        embed_scale = input_embed_layer.embed_scale.clone().detach()
                        question_embeds = question_embeds * embed_scale
                        answer_embeds = answer_embeds * embed_scale
                else:
                    question_embeds = input_embed_layer(question_input_ids)
                    answer_embeds = input_embed_layer(answer_input_ids)

                inputs_embeds_for_loss = torch.cat([projected_embeds, question_embeds, answer_embeds], dim=1)

                # === Prepare Labels and Attention Mask for Loss ===
                batch_size = projected_embeds.shape[0]
                num_visual_tokens = projected_embeds.shape[1]
                q_len = question_input_ids.shape[1]

                visual_attn_mask = torch.ones((batch_size, num_visual_tokens), dtype=torch.long, device=projected_embeds.device)
                # Correctly create question mask based on actual token IDs (including padding)
                question_attn_mask = (question_input_ids != self.tokenizer.pad_token_id).long()
                answer_attn_mask = (answer_input_ids != self.tokenizer.pad_token_id).long()
                attention_mask_for_loss = torch.cat([visual_attn_mask, question_attn_mask, answer_attn_mask], dim=1)

                # Labels: -100 for visual/question, token_id for answer (or -100 if padding)
                visual_labels = torch.full((batch_size, num_visual_tokens), fill_value=-100, dtype=torch.long, device=projected_embeds.device)
                question_labels = torch.full((batch_size, q_len), fill_value=-100, dtype=torch.long, device=projected_embeds.device)
                answer_labels = answer_input_ids.clone()
                answer_labels[answer_labels == self.tokenizer.pad_token_id] = -100
                labels_for_loss = torch.cat([visual_labels, question_labels, answer_labels], dim=1)

                # === Forward Pass for Loss ===
                outputs_for_loss = self.language_model(
                    inputs_embeds=inputs_embeds_for_loss,
                    attention_mask=attention_mask_for_loss,
                    return_dict=True
                )
                logits = outputs_for_loss.logits.to(torch.float32)
                """
                logits[..., :-1, :] removes the prediction for the last token position because there is no subsequent target label to compare it against.
                labels_for_loss[..., 1:] removes the first target label because the model doesn't make a prediction before seeing the first token.
                This ensures the prediction at time step t is compared against the actual label at time step t+1.
                """
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_for_loss[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                total_val_loss += self.accelerator.gather(loss).mean().item()

                # Generate predictions
                try:
                    # Prepare inputs for generation (only visual + question)
                    inputs_embeds_for_gen = torch.cat([projected_embeds, question_embeds], dim=1)
                    attention_mask_for_gen = torch.cat([visual_attn_mask, question_attn_mask], dim=1)

                    llm_model_unwrapped = self.accelerator.unwrap_model(self.language_model)
                    
                    # Set up generation config with explicit padding settings for Qwen3 compatibility
                    gen_kwargs = {
                        'inputs_embeds': inputs_embeds_for_gen,
                        'attention_mask': attention_mask_for_gen,
                        'max_new_tokens': 512,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'do_sample': True,
                        'num_beams': 3,
                        'top_p': 0.9,
                        'top_k': 50
                    }
                    
                    # Add flash attention specific config for Qwen3
                    if 'qwen' in llm_model_unwrapped.__class__.__name__.lower():
                        from transformers import GenerationConfig
                        gen_config = GenerationConfig(
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            padding_side='left'  # Crucial for Qwen3 with Flash Attention
                        )
                        gen_kwargs['generation_config'] = gen_config
                    
                    generated_outputs = llm_model_unwrapped.generate(**gen_kwargs)

                    # Decode outputs
                    decoded_preds = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
                    decoded_questions = self.tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
                    answer_refs = answer_input_ids.clone()
                    decoded_refs = self.tokenizer.batch_decode(answer_refs, skip_special_tokens=True)

                    # Store examples from this batch for gathering
                    batch_examples = []
                    for j in range(len(decoded_preds)):
                        batch_examples.append({
                            "epoch": epoch + 1,
                            "question": decoded_questions[j],
                            "ground_truth": decoded_refs[j],
                            "prediction": decoded_preds[j]
                        })

                    all_examples.extend(batch_examples)

                except Exception as e:
                    logger.error(f"Error during validation generation/logging: {e}", exc_info=True)

        # Wait for all processes to finish validation loop
        self.accelerator.wait_for_everyone()

        # --- Gather examples using accelerator.gather_object ---
        # Wrap the list in another list to potentially avoid incorrect flattening bug in gather_object
        gathered_nested_list = gather_object([all_examples])

        gathered_examples = []
        if self.accelerator.is_main_process:
            # Flatten the list of lists from each process
            for process_list in gathered_nested_list:
                 gathered_examples.extend(process_list)
            gathered_examples = gathered_examples[:max_examples_to_log]

        # Calculate Average Validation Loss
        avg_val_loss = total_val_loss / len(self.val_loader)
        log_metrics = {"val/loss": avg_val_loss, "epoch": epoch + 1}

        # Log metrics and save examples
        if self.accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            self.accelerator.log(log_metrics, step=global_step)

            if gathered_examples:
                try:
                    validation_file = os.path.join(self.validation_dir, f"epoch_{epoch+1}_examples.txt")
                    with open(validation_file, 'w', encoding='utf-8') as f:
                        f.write(f"Validation Examples for Epoch {epoch+1}\n")
                        f.write(f"{'='*80}\n\n")
                        for i, example in enumerate(gathered_examples):
                            f.write(f"Example {i+1}:\n")
                            f.write(f"Question: {example['question']}\n")
                            f.write(f"Ground Truth: {example['ground_truth']}\n")
                            f.write(f"Prediction: {example['prediction']}\n")
                            f.write(f"{'='*80}\n\n")

                    combined_file = os.path.join(self.validation_dir, "all_validation_examples.txt")
                    mode = 'a' if epoch > 0 else 'w'
                    with open(combined_file, mode, encoding='utf-8') as f:
                        f.write(f"\nValidation Examples for Epoch {epoch+1}\n")
                        f.write(f"{'='*80}\n\n")
                        for i, example in enumerate(gathered_examples):
                            f.write(f"Example {i+1}:\n")
                            f.write(f"Question: {example['question']}\n")
                            f.write(f"Ground Truth: {example['ground_truth']}\n")
                            f.write(f"Prediction: {example['prediction']}\n")
                            f.write(f"{'='*80}\n\n")

                    logger.info(f"Saved {len(gathered_examples)} validation examples to {os.path.abspath(validation_file)}")
                    logger.info(f"Updated combined validation examples file: {os.path.abspath(combined_file)}")
                except Exception as e:
                    logger.error(f"Failed to save validation examples to file: {e}", exc_info=True)

        # Set models back to training mode if they were trainable
        self.language_model.train(mode=any(p.requires_grad for p in self.accelerator.unwrap_model(self.language_model).parameters()))
        self.projection_layer.train(mode=any(p.requires_grad for p in self.accelerator.unwrap_model(self.projection_layer).parameters()))
        
        # Restore original tokenizer padding side
        self.tokenizer.padding_side = original_padding_side
        logger.info("Evaluation finished.")

    def save_model(self, path):
        """Saves the trainable components (Adapters if QLoRA, else full models if unfrozen) (main process only)"""
        if not self.accelerator.is_main_process:
            return

        os.makedirs(path, exist_ok=True)

        # Save accelerator state (optimizer, scheduler, etc.)
        self.accelerator.save_state(path)
        logger.info(f"Full training state (optimizer, scheduler) saved to {path} using accelerator.save_state")

        # --- Save Language Model / Adapters --- #
        unwrapped_llm = self.accelerator.unwrap_model(self.language_model)
        llm_save_path = os.path.join(path, "language_model") # Define save path
        if self.enable_qlora:
            logger.info("Saving QLoRA adapters using PEFT save_pretrained...")
            # Save only the adapters using PEFT's method
            # self.accelerator.save_model(unwrapped_llm, llm_save_path) # Using accelerator might not save adapter_config.json properly
            unwrapped_llm.save_pretrained(llm_save_path)
            logger.info(f"QLoRA adapters saved to {llm_save_path}")
        elif any(p.requires_grad for p in unwrapped_llm.parameters()): # Save full model only if not QLoRA and it was trained
             logger.info("Saving full fine-tuned language model...")
             # Use accelerator.save_model for saving the full model if needed
             self.accelerator.save_model(unwrapped_llm, llm_save_path)
             # unwrapped_llm.save_pretrained(llm_save_path) # Alternative HF way
             logger.info(f"Full language model saved to {llm_save_path}")
        else:
             logger.info("Language model was frozen and not saved.")
             
        # Always save tokenizer with the LLM/adapters
        if self.enable_qlora or any(p.requires_grad for p in unwrapped_llm.parameters()):
            self.tokenizer.save_pretrained(llm_save_path) 
            logger.info(f"Tokenizer saved to {llm_save_path}")

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