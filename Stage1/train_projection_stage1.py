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
    def __init__(self, image_root, json_file, processor, tokenizer, img_size, max_length=512):
        self.image_root = image_root
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
    parser.add_argument("--train_json", type=str, required=True, help="JSON file with training image-caption/report data")
    # --- Added arguments for validation ---
    parser.add_argument("--val_json", type=str, default=None, help="JSON file with validation image-caption/report data (optional, overrides train_val_split)")
    parser.add_argument("--train_val_split", type=float, default=0.0, help="Fraction of train_json to use for validation (if val_json is not provided). Default 0 means no validation split.")
    # parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Evaluate on validation set every N epochs (Currently only evaluates after training).") # TODO: Implement evaluation during training if ProjectorTrainer allows
    # ---------------------------------------
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
    parser.add_argument("--save_every_n_epochs", type=int, default=2, help="Save projector checkpoint every N epochs. Set to 0 to disable epoch checkpoints (only save at end).") # Added

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
    if "gemma" in args.llm_name:
        llm_model = Gemma3ForCausalLM.from_pretrained(
            args.llm_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
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

    # --- Post-Training Evaluation (if validation set exists and training succeeded) ---
    if training_successful and val_dataset and accelerator.is_main_process: # Only run eval on main process
        logger.info("Starting post-training evaluation on validation set...")
        
        # Ensure models are on the correct device and in eval mode
        vision_encoder.to(device).eval()
        llm_model.to(device).eval()
        projection.to(device).eval()

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, # Use same batch size or a dedicated eval batch size
            shuffle=False # No need to shuffle for evaluation
        )

        # Prepare dataloader for accelerator
        val_dataloader = accelerator.prepare(val_dataloader)

        total_samples = 0
        correct_last_word = 0
        
        # Get models needed for generation
        unwrapped_llm = accelerator.unwrap_model(llm_model)
        unwrapped_vision = accelerator.unwrap_model(vision_encoder)
        unwrapped_proj = accelerator.unwrap_model(projection)


        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(model_dtype) # Ensure correct dtype
                labels = batch["labels"] # Keep on CPU for decoding reference text? Or move to device? Needs tokenizer decode later.
                # label_token_ids = batch["token_ids"] # Original token IDs for reference decoding

                # Get image embeddings
                vision_outputs = unwrapped_vision(pixel_values=pixel_values, output_hidden_states=False)
                image_embeds = vision_outputs.last_hidden_state # Assuming this is the right output

                # Project embeddings
                projected_embeds = unwrapped_proj(image_embeds)

                # Prepare inputs for LLM generation (need input_ids for prefix/prompt if applicable, or just embeddings)
                # Here we assume direct generation from projected embeds. May need adjustment.
                # This simplified approach might need dummy input_ids for some architectures.
                # Let's create minimal input_ids (e.g., BOS token) and use inputs_embeds.
                batch_size = projected_embeds.shape[0]
                dummy_input_ids = torch.full((batch_size, 1), llm_tokenizer.bos_token_id, dtype=torch.long, device=device)
                
                # Get LLM's embedding layer to potentially concatenate BOS embedding
                inputs_embeds = projected_embeds # Use projected embeds directly as input

                # Generate text
                # Adjust max_new_tokens as needed
                generated_ids = unwrapped_llm.generate(
                    inputs_embeds=inputs_embeds,
                    # input_ids=dummy_input_ids, # Use this if inputs_embeds must be combined with token_ids
                    max_new_tokens=64, # Limit generation length
                    num_beams=1, # Use greedy decoding for speed, or adjust for beam search
                    do_sample=False,
                    pad_token_id=llm_tokenizer.pad_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id
                )

                # Decode generated text and reference text
                generated_texts = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Decode reference captions from token_ids (ensure labels are usable)
                # Need original token_ids, not labels with -100
                # We need to modify the dataset or dataloader to yield original token_ids if not already present
                # Assuming dataset __getitem__ returns "token_ids" which are original, non-masked ids.
                label_token_ids_cpu = batch["token_ids"].cpu().numpy() # Move to CPU for decoding
                reference_texts = llm_tokenizer.batch_decode(label_token_ids_cpu, skip_special_tokens=True)


                # Compare last words
                for ref_text, gen_text in zip(reference_texts, generated_texts):
                    ref_last = get_last_word(ref_text)
                    gen_last = get_last_word(gen_text)
                    if ref_last and gen_last and ref_last == gen_last:
                        correct_last_word += 1
                    total_samples += 1 # Count per-sample comparisons

        if total_samples > 0:
            last_word_accuracy = (correct_last_word / total_samples) * 100
            logger.info(f"Validation Complete: Last Word Accuracy = {last_word_accuracy:.2f}% ({correct_last_word}/{total_samples})")
            # Log to WandB if enabled
            if accelerator.trackers:
                 try:
                     accelerator.log({"validation/last_word_accuracy": last_word_accuracy})
                     logger.info("Logged validation accuracy to tracker.")
                 except Exception as e:
                     logger.warning(f"Failed to log validation accuracy to tracker: {e}")
        else:
            logger.warning("No samples processed during validation.")

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