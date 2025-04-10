import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import argparse
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import Gemma3ForCausalLM
import json
from projectors import MLPProjector
from projector_trainer import ProjectionTrainerStage1
from accelerator_setup import setup_accelerator_and_logging

# Set up logging (Initial basicConfig, might be overridden by setup function)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- XrayTextPairDataset ---
# This dataset needs to provide 'pixel_values' for the image
# and 'labels' for the target caption/report (derived from 'normal_caption' in the original script).
# The 'input_ids' and 'attention_mask' corresponding to the 'problem' text are ignored in this stage.
# This aligns with the CheXagent paper's description.
class XrayTextPairDataset(Dataset):
    """Dataset for X-ray images and their text descriptions (captions/reports) from JSON"""
    def __init__(self, image_root, json_file, processor, tokenizer, img_size, max_length=512):
        self.image_root = image_root
        self.img_size = img_size
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: JSON file not found at {json_file}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Error decoding JSON from {json_file}")
            raise
        except Exception as e:
            print(f"ERROR: An error occurred loading JSON: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            image_filename = sample["image"]
            normal_caption = sample["normal_caption"] # Target text
            image_path = os.path.join(self.image_root, image_filename)

            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            image_inputs = self.processor(images=image, return_tensors="pt")

            # Tokenize the target caption/report
            text_inputs = self.tokenizer(
                normal_caption, max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            token_ids = text_inputs.input_ids.squeeze(0)

            # Copy token_ids to labels before modifying for padding
            labels = token_ids.clone()

            # Replace padding token id with -100 for loss calculation
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100

            # Return both original token IDs and the labels for loss
            return {
                "pixel_values": image_inputs.pixel_values.squeeze(0),
                "token_ids": token_ids, # Original IDs for embedding lookup
                "labels": labels      # IDs with -100 for loss calculation
            }
        except FileNotFoundError:
            # Handle missing images gracefully
            print(f"WARNING: Image file not found for sample {idx}: {image_path}. Skipping.")
            # Return the next valid item recursively
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            # Handle other potential errors during data loading/processing
            print(f"ERROR: Error processing sample {idx} ({image_path}): {e}")
            # Return the next valid item recursively
            return self.__getitem__((idx + 1) % len(self))


def main():
    # Updated description for Stage 1
    parser = argparse.ArgumentParser(description="Train Stage 1: Vision-Language Projector Alignment (CheXagent-style)")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory with training images")
    parser.add_argument("--train_json", type=str, required=True, help="JSON file with training image-caption/report data")
    # Updated default output dir
    parser.add_argument("--output_dir", type=str, default="./trained_projection_stage1", help="Output directory for Stage 1 projector")
    parser.add_argument("--vision_model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli", help="Pre-trained vision encoder name or path.")
    parser.add_argument("--llm_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Pre-trained language model name or path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size *per GPU*.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for projector")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs for Stage 1") # Adjust default as needed
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Ratio of total training steps for linear warmup (e.g., 0.05)") # Added
    parser.add_argument("--wandb_project", type=str, default="xray_projection_training_stage1", help="WandB project name for Stage 1")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--wandb_log_freq", type=int, default=100, help="Log gradients/params to WandB every N steps (if watched)")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable Weights & Biases logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")

    args = parser.parse_args()

    # Initialize Accelerator, logging, and trackers using the setup function
    accelerator = setup_accelerator_and_logging(args)
    device = accelerator.device # Get device after accelerator is initialized

    # Load Models
    logger.info(f"Process {accelerator.process_index}: Loading base models (will be frozen)...")
    # Use bfloat16 if available and supported, otherwise float16
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using model dtype: {model_dtype}")

    # --- Load Vision Encoder (Frozen) ---
    # Use the specific SigLIP model mentioned or your equivalent
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    vision_encoder = AutoModel.from_pretrained(
        args.vision_model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    logger.info(f"Loaded vision encoder: {args.vision_model_name}")

    # --- Load Language Model (Frozen) ---
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    llm_model = Gemma3ForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    # Enable gradient checkpointing to save memory on the frozen LLM
    llm_model.gradient_checkpointing_enable()
    logger.info(f"Loaded language model: {args.llm_name} and enabled gradient checkpointing.")

    # Handle padding token for tokenizer and model
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model.config.pad_token_id = llm_tokenizer.eos_token_id
        if accelerator.is_main_process: logger.info("Set tokenizer pad_token to eos_token")


    # --- Initialize Projector (Trainable) ---
    logger.info("Initializing projector (Trainable Component)...")
    
    vision_dim = vision_encoder.config.vision_config.hidden_size
    llm_dim = llm_model.config.hidden_size

    # Make sure the projector dimensions match the frozen models
    # --- Use the imported MLPProjector ---
    projection = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    logger.info(f"Initialized Projector ({type(projection).__name__}) with vision_dim={vision_dim}, llm_dim={llm_dim}. Output tokens = input patch tokens.")


    # --- Create Dataset ---
    if accelerator.is_main_process: logger.info("Creating training dataset for Stage 1...")
    try:
        train_dataset = XrayTextPairDataset(
            args.image_root,
            args.train_json,
            processor,
            llm_tokenizer,
            img_size=args.img_size,
            # Add max_length if needed by tokenizer in dataset
        )
        logger.info(f"Loaded dataset with {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.log_with is not None:
             accelerator.end_training()
        return

    # --- Create Trainer --- # Use the imported class
    trainer = ProjectionTrainerStage1(
        accelerator=accelerator,
        vision_encoder=vision_encoder,
        language_model=llm_model,
        projection_layer=projection,
        processor=processor,
        tokenizer=llm_tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        wandb_project=args.wandb_project
    )

    # --- WandB Watch ---
    if accelerator.is_main_process and not args.disable_wandb and accelerator.log_with is not None and "wandb" in accelerator.log_with:
        try:
            tracker = accelerator.get_tracker("wandb", unwrap=False)
            if tracker is not None:
                tracker.watch(
                    trainer.projection_layer, log="all", log_freq=args.wandb_log_freq
                )
                logger.info("WandB watching trainable projection layer parameters.")
            else:
                logger.warning("WandB tracker specified but not found via accelerator.get_tracker, skipping watch.")
        except Exception as e:
            logger.warning(f"Could not watch model with WandB: {e}")

    # --- Train ---
    logger.info(f"Process {accelerator.process_index}: Starting Stage 1 training run...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception(f"Process {accelerator.process_index}: An error occurred during Stage 1 training:")
        # Re-raise the exception after logging
        raise
    finally:
        # Ensure WandB run is ended cleanly
        if accelerator.is_main_process and accelerator.trackers:
            accelerator.end_training()
            logger.info("Ended WandB tracking run.")

    logger.info(f"Process {accelerator.process_index}: Stage 1 training script finished!")

if __name__ == "__main__":
    main()