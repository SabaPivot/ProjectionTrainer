import os
import torch
import logging
import argparse
import json
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# --- Project Imports ---
# Assume scripts are run from the root directory or PYTHONPATH is set
import sys
# Add parent directory to sys.path to find sibling modules (projectors, accelerator_setup)
# This is a common pattern but might need adjustment based on execution context
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Stage2.dataset import XrayVQADataset
from Stage2.trainer import VQATrainerStage2
from projectors import MLPProjector # Import from parent directory
from accelerator_setup import setup_accelerator_and_logging # Import from parent directory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pretrained_projector(projector_path, accelerator):
    """Loads a pre-trained projector model and its config."""
    config_path = os.path.join(projector_path, "projector_config.json")
    model_weights_path = projector_path # Assumes accelerator.save_model saved directly in the dir

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Projector config file not found at {config_path}")
    # Check for model weights existence (might be pytorch_model.bin, etc.)
    # This part is tricky as accelerator.save_model might save under different names
    # Let's assume the path passed is the directory containing the weights
    if not os.path.isdir(model_weights_path):
         raise FileNotFoundError(f"Projector model directory not found at {model_weights_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded projector config: {config}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading projector config: {e}")
        raise

    # Instantiate the projector
    projector = MLPProjector(vision_dim=config['vision_dim'], llm_dim=config['llm_dim'])

    # Load the state dict
    # Use accelerator.load_state potentially? No, save_model is simpler.
    # We need to load the state dict manually here as it wasn't saved with full HF save_pretrained
    try:
        # Find the model weights file (common names)
        weight_files = [f for f in os.listdir(model_weights_path) if f.endswith(('.bin', '.safetensors')) and 'optimizer' not in f and 'scheduler' not in f]
        if not weight_files:
            raise FileNotFoundError(f"No model weight file (.bin or .safetensors) found in {model_weights_path}")
        # Prioritize safetensors if found
        model_file = next((f for f in weight_files if f.endswith('.safetensors')), weight_files[0])
        weights_load_path = os.path.join(model_weights_path, model_file)

        logger.info(f"Loading projector weights from: {weights_load_path}")
        state_dict = load_file(weights_load_path, device="cpu") # Load to CPU

        # Adjust keys if saved via accelerator.save_model (might have extra prefix)
        # This depends on how accelerator wraps the model during saving
        # Often, no prefix is added when saving unwrapped model like in Stage 1 script

        # Load state dict
        projector.load_state_dict(state_dict)
        logger.info(f"Successfully loaded pre-trained projector weights into {type(projector).__name__}.")

    except FileNotFoundError as e:
        logger.error(f"Error finding projector weights: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading projector state_dict from {weights_load_path}: {e}", exc_info=True)
        raise

    return projector

def main():
    # --- Disable Anomaly Detection (no longer needed) --- #
    # torch.autograd.set_detect_anomaly(True)
    # logger.info("Enabled PyTorch autograd anomaly detection.")
    # ---------------------------------------------------------------- #

    parser = argparse.ArgumentParser(description="Train Stage 2: VQA Fine-tuning")

    # --- Data Arguments ---
    parser.add_argument("--image_root", type=str, required=True, help="Root directory with training images")
    parser.add_argument("--train_json", type=str, required=True, help="JSON file with training image-question-answer triplets")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    parser.add_argument("--max_q_len", type=int, default=128, help="Max token length for questions")
    parser.add_argument("--max_a_len", type=int, default=512, help="Max token length for answers")

    # --- Model Arguments ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Pre-trained vision encoder name or path.")
    parser.add_argument("--llm_name", type=str, default="google/gemma-3-1b-it", help="Pre-trained language model name or path.")
    parser.add_argument("--stage1_projector_path", type=str, required=True, help="Path to the *directory* containing the trained Stage 1 projector weights and config (e.g., ./trained_projection_stage1/final_model)")

    # --- Training Arguments ---
    parser.add_argument("--output_dir", type=str, default="./Stage2/trained_vqa_stage2", help="Output directory for Stage 2 model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size *per GPU*.") # Reduced default
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs for Stage 2")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total training steps for linear warmup")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients over.") # Increased default
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=True, help="Freeze the vision encoder (default: True)")
    parser.add_argument("--unfreeze_projection_layer", action="store_true", help="Unfreeze and fine-tune the projection layer (default: Frozen)")
    parser.add_argument("--unfreeze_llm", action="store_true", default=True, help="Unfreeze and fine-tune the LLM (default: True)") # Fine-tuning LLM is typical

    # --- Logging Arguments ---
    parser.add_argument("--wandb_project", type=str, default="xray_vqa_training_stage2", help="WandB project name for Stage 2")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--wandb_log_freq", type=int, default=100, help="Log gradients/params to WandB every N steps (if watched)")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # Handle freezing logic based on args
    # Note: Our VQATrainer defaults are set, args allow overriding
    freeze_proj = not args.unfreeze_projection_layer
    freeze_llm = not args.unfreeze_llm

    # Initialize Accelerator, logging, and trackers
    accelerator = setup_accelerator_and_logging(args)
    device = accelerator.device

    # Load Models
    logger.info(f"Process {accelerator.process_index}: Loading base models...")
    # --- Match model loading dtype to accelerator's mixed_precision setting --- #
    if accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    else: # no mixed precision
        model_dtype = torch.float32 # Or keep default None if from_pretrained handles it
    logger.info(f"Using model loading dtype: {model_dtype} based on accelerator precision: {accelerator.mixed_precision}")

    # --- Load Vision Encoder --- (Load fresh, freezing handled by trainer)
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    vision_encoder = AutoModel.from_pretrained(
        args.vision_model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    logger.info(f"Loaded vision encoder: {args.vision_model_name}")

    # --- Load Language Model --- (Load fresh, freezing handled by trainer)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    # Enable gradient checkpointing after model loading
    llm_model.gradient_checkpointing_enable()
    logger.info(f"Loaded language model: {args.llm_name} with attn_implementation='eager' and gradient_checkpointing enabled.")

    # Handle padding token
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model.config.pad_token_id = llm_tokenizer.eos_token_id
        if accelerator.is_main_process: logger.info("Set tokenizer pad_token to eos_token")

    # --- Load Pre-trained Projector --- #
    logger.info(f"Loading pre-trained projector from: {args.stage1_projector_path}")
    try:
        projection_layer = load_pretrained_projector(args.stage1_projector_path, accelerator)
        # Move projector to appropriate dtype and device if needed (before trainer prepare)
        projection_layer = projection_layer.to(dtype=model_dtype)
    except Exception as e:
        logger.error(f"Failed to load Stage 1 projector: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.trackers:
             accelerator.end_training()
        return # Exit if projector fails to load

    # --- Create Dataset --- #
    if accelerator.is_main_process: logger.info("Creating training dataset for Stage 2 VQA...")
    try:
        train_dataset = XrayVQADataset(
            image_root=args.image_root,
            json_file=args.train_json,
            processor=processor,
            tokenizer=llm_tokenizer,
            img_size=args.img_size,
            max_q_len=args.max_q_len,
            max_a_len=args.max_a_len,
        )
        logger.info(f"Loaded VQA dataset with {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.trackers:
             accelerator.end_training()
        return

    # --- Create Trainer --- #
    trainer = VQATrainerStage2(
        accelerator=accelerator,
        vision_encoder=vision_encoder,
        language_model=llm_model,
        projection_layer=projection_layer,
        tokenizer=llm_tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        freeze_vision_encoder=args.freeze_vision_encoder,
        freeze_projection_layer=freeze_proj, # Use derived value
        freeze_llm=freeze_llm,             # Use derived value
        wandb_project=args.wandb_project
    )

    # --- WandB Watch (Optional) --- #
    # Watch trainable parameters
    if accelerator.is_main_process and not args.disable_wandb and "wandb" in accelerator.trackers:
        try:
            tracker = accelerator.get_tracker("wandb", unwrap=False)
            models_to_watch = []
            if trainer.language_model.requires_grad:
                models_to_watch.append(trainer.language_model)
            if trainer.projection_layer.requires_grad:
                models_to_watch.append(trainer.projection_layer)
            # Add vision encoder if trainable (unlikely but possible)
            if trainer.vision_encoder.requires_grad:
                 models_to_watch.append(trainer.vision_encoder)

            if models_to_watch and tracker:
                logger.info(f"WandB watching {len(models_to_watch)} trainable model component(s)...")
                for model_to_watch in models_to_watch:
                     tracker.watch(model_to_watch, log="all", log_freq=args.wandb_log_freq)
            elif tracker:
                 logger.info("No components set as trainable, WandB watching skipped.")
            else:
                logger.warning("WandB tracker specified but not found via accelerator.get_tracker, skipping watch.")
        except Exception as e:
            logger.warning(f"Could not watch model(s) with WandB: {e}")

    # --- Train --- #
    logger.info(f"Process {accelerator.process_index}: Starting Stage 2 training run...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception(f"Process {accelerator.process_index}: An error occurred during Stage 2 training:")
        # Re-raise the exception after logging
        raise
    finally:
        # Ensure WandB run is ended cleanly
        if accelerator.is_main_process and accelerator.trackers:
            accelerator.end_training()
            logger.info("Ended WandB tracking run.")

    logger.info(f"Process {accelerator.process_index}: Stage 2 training script finished!")

if __name__ == "__main__":
    main() 