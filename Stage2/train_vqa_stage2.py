import os
import torch
import logging
import argparse
import json
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from safetensors.torch import load_file
import sys
# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Stage2.dataset import XrayVQADataset
from Stage2.trainer import VQATrainerStage2
from Stage1.projectors import MLPProjector
from Stage1.accelerator_setup import setup_accelerator_and_logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pretrained_projector(projector_path):
    """Loads a pre-trained projector model and its config."""
    config_path = os.path.join(projector_path, "projector_config.json")
    model_weights_path = projector_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Projector config file not found at {config_path}")
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
    try:
        # Find the model weights file (common names)
        weight_files = [f for f in os.listdir(model_weights_path) if f.endswith(('.bin', '.safetensors')) and 'optimizer' not in f and 'scheduler' not in f]
        if not weight_files:
            raise FileNotFoundError(f"No model weight file (.bin or .safetensors) found in {model_weights_path}")
        # Prioritize safetensors if found
        model_file = next((f for f in weight_files if f.endswith('.safetensors')), weight_files[0])
        weights_load_path = os.path.join(model_weights_path, model_file)

        logger.info(f"Loading projector weights from: {weights_load_path}")
        
        # Check file extension and use appropriate loading method
        if weights_load_path.endswith('.safetensors'):
            # Use safetensors for .safetensors files
            state_dict = load_file(weights_load_path, device="cpu") # Load to CPU
        else:
            # Use torch.load for .bin files
            state_dict = torch.load(weights_load_path, map_location="cpu")

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
    parser = argparse.ArgumentParser(description="Train Stage 2: VQA Fine-tuning")

    # --- Data Arguments ---
    parser.add_argument("--image_root", type=str, required=True, help="Root directory with training images")
    parser.add_argument("--image_root_2", type=str, default=None, help="Secondary root directory for images with a different path format (e.g., 'p10012261/s50349409')")
    parser.add_argument("--train_json", type=str, required=True, help="JSON file with training image-question-answer triplets")
    parser.add_argument("--val_json", type=str, required=True, help="JSON file with validation image-question-answer triplets")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    parser.add_argument("--max_q_len", type=int, default=128, help="Max token length for questions")
    parser.add_argument("--max_a_len", type=int, default=512, help="Max token length for answers")

    # --- Model Arguments ---
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Pre-trained vision encoder name or path.")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B", help="Pre-trained language model name or path.")
    parser.add_argument("--stage1_projector_path", type=str, required=True, help="Path to the *directory* containing the trained Stage 1 projector weights and config")

    # --- Training Arguments ---
    parser.add_argument("--output_dir", type=str, default="./Stage2/trained_vqa_stage2", help="Output directory for Stage 2 model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size *per GPU*.") # Default 1 for QLoRA on larger models
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for fine-tuning (QLoRA often uses higher)") # Updated default
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs for Stage 2")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total training steps for linear warmup")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients over.")
    # --- Finetuning Flags ---
    parser.add_argument("--enable_qlora", action="store_true", help="Enable QLoRA fine-tuning for the LLM.")
    parser.add_argument("--unfreeze_projection_layer", action="store_true", help="Unfreeze and fine-tune the projection layer (default: Frozen)")
    parser.add_argument("--unfreeze_llm", action="store_true", help="Unfreeze and fine-tune the FULL LLM (ignored if --enable_qlora is set).") # Clarified help text
    parser.add_argument("--train_ve_first_epoch", action="store_true", help="Train the Vision Encoder only during the first epoch (requires VE params in optimizer).")
    parser.add_argument("--resume_qlora_adapter_path", type=str, default=None, help="Path to a pre-trained QLoRA adapter directory to resume training from (requires --enable_qlora).")

    # --- Logging Arguments ---
    parser.add_argument("--wandb_project", type=str, default="xray_vqa_training_stage2", help="WandB project name for Stage 2")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # --- Determine Freezing Strategy --- #
    freeze_proj = not args.unfreeze_projection_layer
    # LLM freezing depends on QLoRA flag
    if args.enable_qlora:
        freeze_llm = True # QLoRA freezes base model, only adapters are trained
        logger.info("QLoRA enabled. Base LLM parameters will be frozen. --unfreeze_llm flag is ignored.")
    else:
        freeze_llm = not args.unfreeze_llm # Use the standard flag if QLoRA is off

    initial_freeze_ve = not args.train_ve_first_epoch
    if not initial_freeze_ve:
        logger.info("Note: Vision Encoder parameters will be included in optimizer for potential first-epoch training (--train_ve_first_epoch=True).")
    else:
        logger.info("Vision Encoder parameters will NOT be included in optimizer (--train_ve_first_epoch=False).")

    # Initialize Accelerator, logging, and trackers
    accelerator = setup_accelerator_and_logging(args)

    # Load Models
    logger.info(f"Process {accelerator.process_index}: Loading base models...")
    if accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16 # QLoRA compute dtype
    else:
        model_dtype = torch.float32
    logger.info(f"Using model loading dtype: {model_dtype} based on accelerator precision: {accelerator.mixed_precision}")

    # --- Load Vision Encoder --- #
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    vision_encoder = AutoModel.from_pretrained(
        args.vision_model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    logger.info(f"Loaded vision encoder: {args.vision_model_name}")

    # --- Load Language Model (potentially with QLoRA) --- #
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    # Ensure padding side is left for Qwen3 + Flash Attention compatibility
    if 'qwen' in args.llm_name.lower():
        llm_tokenizer.padding_side = 'left'
        if accelerator.is_main_process: 
            logger.info(f"Explicitly set tokenizer padding_side to '{llm_tokenizer.padding_side}' for Qwen model.")

    logger.info(f"Loading language model: {args.llm_name}")

    quantization_config = None
    if args.enable_qlora:
        logger.info("Setting up QLoRA configuration...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype, # Use bf16 or fp16 based on accelerator
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"BitsAndBytesConfig: {quantization_config}")

    # Load the base LLM, applying quantization if enabled
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=quantization_config, # Pass config here
        torch_dtype=model_dtype, # Load in compute dtype if not quantizing
        low_cpu_mem_usage=True,
        # device_map="auto" # Let Accelerator handle device mapping
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and args.enable_qlora else None # Use FA2 if QLoRA + CUDA
    )
    logger.info(f"Loaded language model '{args.llm_name}' using AutoModelForCausalLM.")
    if args.enable_qlora:
         logger.info("LLM loaded with 4-bit quantization.")
         # Prepare model for k-bit training *before* applying PEFT config
         llm_model = prepare_model_for_kbit_training(llm_model)
         logger.info("Model prepared for k-bit training.")

    # Handle padding token (after model load, before PEFT potentially)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        # Don't set model.config.pad_token_id if using PEFT, PEFT handles it?
        # Check PEFT docs, but usually tokenizer is enough.
        # llm_model.config.pad_token_id = llm_tokenizer.eos_token_id # Maybe not needed
        if accelerator.is_main_process: logger.info("Set tokenizer pad_token to eos_token")

    # --- Apply PEFT / QLoRA or Load Adapters --- #
    if args.enable_qlora:
        # Prepare base model for k-bit training first
        llm_model = prepare_model_for_kbit_training(llm_model)
        logger.info("Base model prepared for k-bit training.")

        if args.resume_qlora_adapter_path:
            logger.info(f"Attempting to load QLoRA adapter from: {args.resume_qlora_adapter_path}")
            if not os.path.isdir(args.resume_qlora_adapter_path):
                logger.error(f"Provided adapter path is not a directory: {args.resume_qlora_adapter_path}")
                if accelerator.is_main_process and accelerator.trackers:
                    accelerator.end_training()
                return

            try:
                # Load the adapter onto the base model. is_trainable=True ensures adapters are ready for training.
                llm_model = PeftModel.from_pretrained(llm_model, args.resume_qlora_adapter_path, is_trainable=True)
                logger.info(f"Successfully loaded QLoRA adapter from {args.resume_qlora_adapter_path}.")
                if accelerator.is_main_process:
                    llm_model.print_trainable_parameters()
            except Exception as e:
                logger.error(f"Failed to load QLoRA adapter from {args.resume_qlora_adapter_path}: {e}", exc_info=True)
                # Exit if adapter loading fails
                if accelerator.is_main_process and accelerator.trackers:
                     accelerator.end_training()
                return
        else:
            logger.info("Applying NEW LoRA configuration for QLoRA (no adapter path provided)...")
            # Define LoRA configuration
            lora_config = LoraConfig(
                r=16, # Rank of the update matrices
                lora_alpha=32, # Alpha scaling factor (often 2*r)
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"], # Common targets for Qwen/Llama-like
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            logger.info(f"New LoraConfig: {lora_config}")

            # Get NEW PEFT model
            llm_model = get_peft_model(llm_model, lora_config)
            logger.info("Applied NEW PEFT model wrapper for QLoRA.")
            if accelerator.is_main_process:
                 llm_model.print_trainable_parameters()

    # Enable gradient checkpointing *after* potential PEFT wrapping/loading
    # Note: If using QLoRA, gradient checkpointing is often enabled by prepare_model_for_kbit_training
    # or PeftModel loading
    if not args.enable_qlora: # Only enable explicitly if not using QLoRA
         llm_model.gradient_checkpointing_enable()
         logger.info("Gradient checkpointing enabled for full-precision language model.")
    elif hasattr(llm_model, 'is_gradient_checkpointing'): # Check if already enabled by PEFT utils
         if not llm_model.is_gradient_checkpointing:
             # Recommended for QLoRA: use_reentrant=False
             llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
             logger.info("Gradient checkpointing explicitly enabled for QLoRA model.")
         else:
             logger.info("Gradient checkpointing already enabled (likely by PEFT utils). ")
    else:
        # Attempt to enable if attribute doesn't exist (might be older PEFT/Transformers)
        try:
            llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info("Attempted to enable gradient checkpointing for QLoRA model (attribute missing).")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing for QLoRA model: {e}")

    # --- Load Pre-trained Projector --- #
    logger.info(f"Loading pre-trained projector from: {args.stage1_projector_path}")
    try:
        projection_layer = load_pretrained_projector(args.stage1_projector_path)
        projection_layer = projection_layer.to(dtype=model_dtype)
    except Exception as e:
        logger.error(f"Failed to load Stage 1 projector: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.trackers:
             accelerator.end_training()
        return

    # --- Create Dataset --- #
    if accelerator.is_main_process: logger.info("Creating training and validation datasets for Stage 2 VQA...")
    try:
        train_dataset = XrayVQADataset(
            image_root=args.image_root,
            json_file=args.train_json,
            processor=processor,
            tokenizer=llm_tokenizer,
            img_size=args.img_size,
            max_q_len=args.max_q_len,
            max_a_len=args.max_a_len,
            image_root_2=args.image_root_2,
        )
        logger.info(f"Loaded Train dataset with {len(train_dataset)} samples.")
        val_dataset = XrayVQADataset(
            image_root=args.image_root,
            json_file=args.val_json,
            processor=processor,
            tokenizer=llm_tokenizer,
            img_size=args.img_size,
            max_q_len=args.max_q_len,
            max_a_len=args.max_a_len,
            image_root_2=args.image_root_2,
        )
        logger.info(f"Loaded Validation dataset with {len(val_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.trackers:
             accelerator.end_training()
        return

    # --- Create Trainer --- #
    trainer = VQATrainerStage2(
        accelerator=accelerator,
        vision_encoder=vision_encoder,
        language_model=llm_model, # Pass potentially PEFT-wrapped model
        projection_layer=projection_layer,
        tokenizer=llm_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        # Pass freezing flags derived *after* considering QLoRA
        freeze_vision_encoder=initial_freeze_ve,
        freeze_projection_layer=freeze_proj,
        freeze_llm=freeze_llm, # This will be True if QLoRA is enabled
        enable_qlora=args.enable_qlora, # Pass the QLoRA flag
        train_ve_first_epoch=args.train_ve_first_epoch,
        wandb_project=args.wandb_project
    )

    # --- Train --- #
    logger.info(f"Process {accelerator.process_index}: Starting Stage 2 training run...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception(f"Process {accelerator.process_index}: An error occurred during Stage 2 training:")
        raise
    finally:
        if accelerator.is_main_process and accelerator.trackers:
            accelerator.end_training()
            logger.info("Ended WandB tracking run.")

    logger.info(f"Process {accelerator.process_index}: Stage 2 training script finished!")

if __name__ == "__main__":
    main() 