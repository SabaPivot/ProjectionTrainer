# Stage 0: Vision Encoder SigLIP Fine-tuning

This stage focuses on fine-tuning a pre-trained SigLIP vision encoder for the specific domain of Chest X-ray (CXR) images.

## Goal

The objective is to adapt the general-purpose vision encoder to produce more effective and semantically meaningful embeddings (numerical representations) for CXR images. This is achieved by aligning the image embeddings with the embeddings of corresponding text captions using a contrastive learning approach.

## Methodology

- **Contrastive Learning:** The training uses the SigLIP loss function. This loss encourages the model to pull the embeddings of paired images and captions closer together in the embedding space while pushing apart the embeddings of unpaired images and captions.
- **Data:** It requires a dataset of CXR images paired with relevant text descriptions (e.g., captions or sections from radiology reports).
- **Model Freezing:** Typically, the pre-trained SigLIP *text encoder* and the `logit_scale` parameter are kept frozen during this stage. The optimization focuses solely on updating the weights of the *vision encoder*.

## Key Components

1.  **`run_train_vision_encoder_stage0.sh` (Launcher Script):**
    *   Sets all configuration parameters (model name, dataset paths, hyperparameters like learning rate and batch size, freezing options, logging details).
    *   Uses `accelerate launch` to initiate the training process, handling multi-GPU distribution and mixed-precision settings.
    *   Passes the configuration parameters as command-line arguments to the Python training script.

2.  **`train_vision_encoder_stage0.py` (Main Training Script):**
    *   **`parse_args()`:** Defines and parses command-line arguments received from the run script.
    *   **`accelerator_setup.py` (`setup_accelerator_and_logging()`):** Imported function that initializes the Hugging Face `Accelerator`, configures distributed training, sets up logging, and initializes WandB tracking based on arguments.
    *   **`ImageTextContrastiveDataset`:** Custom PyTorch `Dataset` class.
        *   Loads image paths and text captions from the provided JSON file.
        *   `__getitem__`: Loads a single image, preprocesses it using the `SiglipImageProcessor` (resizing, normalization), tokenizes the corresponding caption using the `SiglipTokenizer` (padding, truncation, adding attention mask), and returns the necessary tensors (`pixel_values`, `input_ids`, `attention_mask`).
        *   `collate_fn`: Batches individual samples returned by `__getitem__`.
    *   **`siglip_loss()`:** Implements the SigLIP loss calculation using image/text features and the logit scale.
    *   **`VisionEncoderTrainerStage0`:** Class encapsulating the training logic.
        *   `__init__`: Initializes the tokenizer, processor, dataset, dataloader, loads the base SigLIP model (`AutoModel`), freezes specified components (text encoder, logit scale), sets up the optimizer (`AdamW`) for trainable parameters (vision encoder), configures the learning rate scheduler, and prepares all these components using `accelerator.prepare()`.
        *   `train()`: Contains the main training loop iterating over epochs and batches. Performs the forward pass, calculates the SigLIP loss, executes the backward pass (`accelerator.backward()`), updates the optimizer and scheduler, and logs metrics using `accelerator.log()`.
        *   `save_model()`: Saves the fine-tuned `vision_model` state dictionary and the processor/tokenizer configuration files to the output directory, ensuring saving only occurs on the main process.
    *   **`main()`:** Orchestrates the overall process: parses args, sets up accelerator/logging, loads processor/tokenizer, creates the dataset and trainer instances, and starts the training.

## Workflow

1.  Execute `run_train_vision_encoder_stage0.sh`.
2.  The script configures parameters and calls `accelerate launch`.
3.  `accelerate launch` starts multiple Python processes (one per GPU).
4.  Each `train_vision_encoder_stage0.py` process:
    *   Parses arguments.
    *   Calls `setup_accelerator_and_logging()` to initialize `Accelerator` and logging/WandB for that process.
    *   Loads processor and tokenizer.
    *   Initializes the dataset and trainer.
    *   The `trainer.train()` method begins:
        *   The prepared `DataLoader` distributes batches across GPUs.
        *   Model forward pass, SigLIP loss calculation, backward pass, and optimizer steps are performed, synchronized across GPUs by `Accelerator`.
        *   Metrics are logged periodically (e.g., to WandB) by the main process.
        *   Checkpoints (vision encoder weights, processor/tokenizer configs) are saved periodically by the main process.
5.  After training completes, the final model checkpoint is saved.

## Configuration

Key parameters in `run_train_vision_encoder_stage0.sh` that may need adjustment:

-   `MODEL_NAME`: The base pre-trained SigLIP model.
-   `TRAIN_JSON`, `IMAGE_ROOT`: Paths to your specific image-text dataset.
-   `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`: Core training hyperparameters.
-   `FREEZE_FLAGS`: Add `--no_freeze_text_encoder` or `--no_freeze_logit_scale` to train these components (default is frozen).
-   `OUTPUT_DIR`, `WANDB_PROJECT`, `RUN_NAME`: Output and logging destinations.
-   `NUM_GPUS`: Should match your available hardware.

## How to Run

```bash
# Make the script executable (if needed)
chmod +x Stage0/run_train_vision_encoder_stage0.sh

# Run the training script
./Stage0/run_train_vision_encoder_stage0.sh
```

## Output

The script saves checkpoints in subdirectories within the specified `OUTPUT_DIR`. Each checkpoint directory (e.g., `epoch_1`, `final_model`) contains:

-   `vision_encoder.bin`: The state dictionary of the fine-tuned vision encoder.
-   Processor and Tokenizer configuration files (e.g., `preprocessor_config.json`, `tokenizer_config.json`, `vocab.txt`). 