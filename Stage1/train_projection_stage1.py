import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import argparse
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Gemma3ForCausalLM
import json
from projectors import MLPProjector
from projector_trainer import ProjectionTrainerStage1
from accelerator_setup import setup_accelerator_and_logging
import re
from sklearn.model_selection import train_test_split

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
    def __init__(self, image_root, json_file, processor, tokenizer, img_size, max_length=512, image_root_2=None):
        self.image_root = image_root
        self.image_root_2 = image_root_2
        self.img_size = img_size
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [] # Initialize samples list

        # Only load from file if json_file is provided
        if json_file:
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
        # If json_file is None, samples list remains empty, expecting it to be populated externally.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = None # Initialize image_path for logging in case of early error
        try:
            sample = self.samples[idx]
            image_filename_or_dir = sample["image"] # This might be a filename or a directory path part
            normal_caption = sample["normal_caption"] # Target text

            # Default assume primary root with direct filename
            image_path = os.path.join(self.image_root, image_filename_or_dir)
            is_mimic_path = False

            # Check if file exists in primary root
            if not os.path.exists(image_path):
                if self.image_root_2:
                    # Construct potential MIMIC directory path
                    mimic_dir_path = os.path.join(self.image_root_2, image_filename_or_dir)
                    if os.path.isdir(mimic_dir_path):
                        is_mimic_path = True
                        # Find the first .jpg file in the directory
                        try:
                            jpg_files = [f for f in os.listdir(mimic_dir_path) if f.lower().endswith('.jpg')]
                            if jpg_files:
                                image_path = os.path.join(mimic_dir_path, jpg_files[0]) # Use the first JPG found
                                logger.debug(f"Found MIMIC image: {image_path}")
                            else:
                                raise FileNotFoundError(f"No .jpg file found in MIMIC directory: {mimic_dir_path}")
                        except Exception as e:
                            raise FileNotFoundError(f"Error listing or finding JPG in {mimic_dir_path}: {e}")
                    else:
                        # If not a directory in root 2, maybe it's a direct file path there?
                        image_path = mimic_dir_path # Treat as potential file path in root 2
                else:
                     # If not found in root 1 and no root 2, raise error
                     raise FileNotFoundError(f"Image not found in primary root and no secondary root provided: {image_path}")
            
            # Final check if a valid image_path was found/constructed
            if not os.path.exists(image_path) or os.path.isdir(image_path):
                 # Log details if path resolution failed
                 if is_mimic_path:
                     details = f"Failed to resolve MIMIC path. Started with: {image_filename_or_dir}. Looked in: {os.path.join(self.image_root_2, image_filename_or_dir)}. Final invalid path: {image_path}"
                 else:
                     details = f"Failed to find image file. Started with: {image_filename_or_dir}. Looked in: {self.image_root}. Final invalid path: {image_path}"
                 raise FileNotFoundError(f"Image path is invalid or a directory. {details}")

            # --- Load and Process Image --- #
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            image_inputs = self.processor(images=image, return_tensors="pt")

            # --- Tokenize Text --- #
            text_inputs = self.tokenizer(
                normal_caption, max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            token_ids = text_inputs.input_ids.squeeze(0)
            labels = token_ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": image_inputs.pixel_values.squeeze(0),
                "token_ids": token_ids,
                "labels": labels
            }
        except FileNotFoundError as e:
            # Log and re-raise to make the error explicit
            logger.error(f"ERROR (FileNotFound) processing sample {idx}: {e}")
            raise e # Re-raise the exception
        except Exception as e:
            # Log and re-raise other errors
            logger.error(f"ERROR processing sample {idx} (path was {image_path}): {e}", exc_info=True)
            raise e # Re-raise the exception

# Helper function to extract the last word
def get_last_word(text):
    if not text or not isinstance(text, str):
        return ""
    # Find all word characters sequences
    words = re.findall(r'\b\w+\b', text.lower())
    return words[-1] if words else ""

def main():
    # Updated description for Stage 1
    parser = argparse.ArgumentParser(description="Train Stage 1: Vision-Language Projector Alignment (CheXagent-style)")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory with training images")
    parser.add_argument("--image_root_2", type=str, default=None, help="Secondary root directory for images (e.g., MIMIC-CXR)")
    parser.add_argument("--train_json", type=str, required=True, help="JSON file with training image-caption/report data")
    parser.add_argument("--val_json", type=str, default=None, help="JSON file with validation image-caption/report data (optional, overrides train_val_split)")
    parser.add_argument("--train_val_split", type=float, default=0.0, help="Fraction of train_json to use for validation (if val_json is not provided). Default 0 means no validation split.")
    parser.add_argument("--output_dir", type=str, default="./trained_projection_stage1", help="Output directory for Stage 1 projector")
    parser.add_argument("--vision_model_name", type=str, default="/mnt/samuel/Siglip/soombit/checkpoint/epoch_16", help="Pre-trained vision encoder name or path.")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B", help="Pre-trained language model name or path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size *per GPU*.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for projector")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs for Stage 1")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Ratio of total training steps for linear warmup (e.g., 0.05)")
    parser.add_argument("--wandb_project", type=str, default="xray_projection_training_stage1", help="WandB project name for Stage 1")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--wandb_log_freq", type=int, default=100, help="Log gradients/params to WandB every N steps (if watched)")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable Weights & Biases logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    parser.add_argument("--save_every_n_epochs", type=int, default=2, help="Save projector checkpoint every N epochs. Set to 0 to disable epoch checkpoints (only save at end).")
    parser.add_argument("--enable_qlora", action="store_true", help="Load LLM in 4-bit using QLoRA config for memory saving (LLM remains frozen).")

    args = parser.parse_args()

    # Initialize Accelerator, logging, and trackers using the setup function
    accelerator = setup_accelerator_and_logging(args)
    device = accelerator.device

    # Load Models
    logger.info(f"Process {accelerator.process_index}: Loading base models (will be frozen)...")
    if accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16 # Use bf16 for QLoRA compute if enabled
    else:
        model_dtype = torch.float32
    logger.info(f"Using model loading dtype: {model_dtype} based on accelerator precision: {accelerator.mixed_precision}")

    # --- Load Vision Encoder (Frozen) ---
    processor = AutoProcessor.from_pretrained(args.vision_model_name)
    vision_encoder = AutoModel.from_pretrained(
        args.vision_model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    logger.info(f"Loaded vision encoder: {args.vision_model_name}")

    # --- Load Language Model (Frozen) ---
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    # Set padding side to left for Flash Attention compatibility during generation
    llm_tokenizer.padding_side = 'left'
    logger.info(f"Set tokenizer padding_side to '{llm_tokenizer.padding_side}'")

    quantization_config = None
    if args.enable_qlora:
        logger.info("Setting up QLoRA configuration for 4-bit LLM loading...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype, # bf16 or fp16
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"BitsAndBytesConfig: {quantization_config}")

    logger.info(f"Loading language model: {args.llm_name}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=quantization_config, # Apply if QLoRA enabled
        torch_dtype=model_dtype if quantization_config is None else None, # Load in compute dtype only if not quantizing
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() and args.enable_qlora else None # REMOVED: Let transformers handle default attn for stability in Stage 1
    )
    logger.info(f"Loaded language model '{args.llm_name}'. Quantized: {args.enable_qlora}")

    # LLM remains frozen in Stage 1, regardless of QLoRA loading
    # llm_model.requires_grad_(False) # This will be handled by the trainer
    # Enable gradient checkpointing ONLY if NOT using QLoRA (QLoRA handles it internally)
    if not args.enable_qlora:
        llm_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for full-precision LLM.")
    else:
        # Ensure gradient checkpointing is used with QLoRA, prepare_model... does this
        # but we aren't calling that here as we don't apply PEFT config. Check if needed manually.
        if hasattr(llm_model, "is_gradient_checkpointing") and not llm_model.is_gradient_checkpointing:
             llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # Recommended for QLoRA
             logger.info("Explicitly enabled gradient checkpointing for QLoRA-loaded LLM.")
        elif not hasattr(llm_model, "is_gradient_checkpointing"):
             # If the attribute doesn't exist, try enabling anyway
             try:
                 llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # Recommended for QLoRA
                 logger.info("Enabled gradient checkpointing for QLoRA-loaded LLM (attribute check failed).")
             except Exception as e:
                 logger.warning(f"Could not enable gradient checkpointing for QLoRA model: {e}")
        else:
             logger.info("Gradient checkpointing likely already enabled for QLoRA-loaded LLM.")

    # Handle padding token for tokenizer and model
    if llm_tokenizer.pad_token is None:
        # Important: Set pad_token AFTER setting padding_side
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")
        # No need to set model config pad_token_id when padding_side is left and pad=eos
        # if llm_model.config.pad_token_id is None:
        #     llm_model.config.pad_token_id = llm_tokenizer.eos_token_id

    # --- Initialize Projector (Trainable) ---
    logger.info("Initializing projector (Trainable Component)...")
    
    vision_dim = vision_encoder.config.vision_config.hidden_size
    llm_dim = llm_model.config.hidden_size

    # Make sure the projector dimensions match the frozen models
    # --- Use the imported MLPProjector ---
    projection = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    logger.info(f"Initialized Projector ({type(projection).__name__}) with vision_dim={vision_dim}, llm_dim={llm_dim}. Output tokens = input patch tokens.")
    
    # --- Add Debug Log --- #
    # Assuming default expansion_factor=10 from MLPProjector definition
    expansion_factor = 10 # TODO: Hardcoded based on MLPProjector default, consider making dynamic if default changes
    intermediate_dim = vision_dim * expansion_factor
    logger.info(f"Projector Dimensions for LLM '{args.llm_name}': Vision({vision_dim}) -> Expanded({intermediate_dim}) -> LLM({llm_dim})")
    # ---------------------

    # --- Create Dataset ---
    if accelerator.is_main_process: logger.info("Creating datasets for Stage 1...")
    train_samples = []
    val_samples = []

    try:
        # Load main training data
        with open(args.train_json, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        logger.info(f"Loaded {len(all_samples)} total samples from {args.train_json}")

        if args.val_json:
            # Use a separate validation file if provided
            with open(args.val_json, 'r', encoding='utf-8') as f:
                val_samples = json.load(f)
            train_samples = all_samples # Use all loaded samples for training
            logger.info(f"Using separate validation file: {args.val_json} ({len(val_samples)} samples)")
        elif args.train_val_split > 0 and args.train_val_split < 1:
            # Split the loaded data if split ratio is valid
            if not all_samples:
                 raise ValueError("Cannot perform train/val split on empty dataset.")
            try:
                train_samples, val_samples = train_test_split(
                    all_samples,
                    test_size=args.train_val_split,
                    random_state=42 # for reproducibility
                )
                logger.info(f"Splitting data: {len(train_samples)} train, {len(val_samples)} validation samples.")
            except Exception as e:
                logger.error(f"Error during train_test_split: {e}. Check scikit-learn installation and data format.", exc_info=True)
                raise
        else:
            # No validation split requested or needed
            train_samples = all_samples
            logger.info("No validation split performed.")

        # Create training dataset
        if not train_samples:
             raise ValueError("Training dataset is empty after processing splits/loading.")
        train_dataset = XrayTextPairDataset(
            args.image_root,
            None, # samples provided directly
            processor,
            llm_tokenizer,
            img_size=args.img_size,
            image_root_2=args.image_root_2
        )
        train_dataset.samples = train_samples # Directly assign split samples
        logger.info(f"Created training dataset with {len(train_dataset)} samples.")

        # Create validation dataset (if val_samples exist)
        val_dataset = None
        if val_samples:
            val_dataset = XrayTextPairDataset(
                args.image_root,
                None, # samples provided directly
                processor,
                llm_tokenizer,
                img_size=args.img_size,
                image_root_2=args.image_root_2
            )
            val_dataset.samples = val_samples # Directly assign split samples
            logger.info(f"Created validation dataset with {len(val_dataset)} samples.")

    except FileNotFoundError as e:
        logger.error(f"Failed to find dataset file: {e}", exc_info=True)
        if accelerator.is_main_process and accelerator.log_with is not None:
             accelerator.end_training()
        return
    except Exception as e:
        logger.error(f"Failed to create dataset(s): {e}", exc_info=True)
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
        val_dataset=val_dataset,  # Add validation dataset
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        wandb_project=args.wandb_project,
        save_every_n_epochs=args.save_every_n_epochs
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
    training_successful = False
    try:
        trainer.train()
        training_successful = True # Mark training as successful if train() completes
    except Exception as e:
        logger.exception(f"Process {accelerator.process_index}: An error occurred during Stage 1 training:")
        # Re-raise the exception after logging? Or just log and proceed to cleanup? Decide based on desired behavior.
        # raise # Option 1: Stop execution here
        # Option 2: Log and attempt cleanup/evaluation if needed
    finally:
        # Ensure WandB run is ended cleanly, only if training didn't already end it potentially
        # We might want to keep it active for post-training eval logging
        # if accelerator.is_main_process and accelerator.trackers:
        #     accelerator.end_training()
        #     logger.info("Ended WandB tracking run.")
        pass # Let evaluation handle final WandB end

    # --- Final Cleanup ---
    # Ensure WandB run is ended cleanly if not already done
    if accelerator.is_main_process and accelerator.trackers:
        # Check if tracker is initialized before ending
        try:
            # Attempt to get a tracker to see if it's active
             tracker_list = accelerator.trackers
             if tracker_list: # Ensure trackers exist
                 accelerator.end_training()
                 logger.info("Ended WandB tracking run.")
        except Exception as e:
            logger.warning(f"Issue ending WandB tracking run: {e}")


    logger.info(f"Process {accelerator.process_index}: Stage 1 script finished!")

if __name__ == "__main__":
    main()