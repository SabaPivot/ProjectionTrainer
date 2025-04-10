# Siglip-Based Chest X-Ray Vision-Language Model

## Overview

This project implements a multi-modal Vision-Language Model (VLM) specifically designed for Chest X-Ray (CXR) analysis. Inspired by the CheXagent methodology, it leverages a pre-trained SigLIP vision encoder and a Gemma3 language model, connecting them via a trained MLP projector. The model takes a CXR image and a text instruction (e.g., a question) as input and generates a free-form text response.

The training process involves two distinct stages: first aligning the vision and language modalities by training the projector, and then fine-tuning the language model (and optionally other components) on instruction-following VQA tasks.

## Model Architecture

The model comprises three main components:

1.  **Image Encoder:** Encodes CXR images into patch-based visual features.
    *   Uses a pre-trained **SigLIP** model (e.g., `StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli`).
    *   The final hidden states corresponding to image patches (excluding the `[CLS]` token) are used.
2.  **Vision-Language Projector:** An MLP (defined in `projectors.py`) that projects the visual patch features into the language model's embedding space.
3.  **Language Decoder:** Processes the projected visual features and input text instructions to generate responses.
    *   Uses a pre-trained **Gemma3** model (e.g., `google/gemma-3-1b-it`).

## Training Process

A two-stage training approach is employed:

### Stage 1: Vision-Language Projector Training (`Stage1/`)

*   **Goal:** Align the vision encoder's output space with the language model's input space by training the MLP projector.
*   **Method:**
    *   The SigLIP vision encoder and Gemma3 language model weights are **frozen**.
    *   Only the **MLP Projector** weights are trained.
    *   Uses **image-text pairs** (e.g., CXR images and captions/reports).
    *   The projector learns to map visual patch embeddings (CLS token excluded) such that the frozen LLM can predict the associated text when conditioned on these projected embeddings.
    *   The loss is the standard causal language modeling loss, calculated only on the text tokens.
*   **Key Scripts:**
    *   `Stage1/train_projection_stage1.py`: Main training script.
    *   `Stage1/projector_trainer.py`: Contains the `ProjectionTrainerStage1` class.
    *   `Stage1/dataset.py`: Dataset definition for image-text pairs.

### Stage 2: VQA Instruction Fine-tuning (`Stage2/`)

*   **Goal:** Fine-tune the model to understand and follow instructions related to CXR images.
*   **Method:**
    *   The **Language Model** is typically **fine-tuned**.
    *   The **MLP Projector** can be **fine-tuned or kept frozen** (controlled by `--unfreeze_projection_layer`).
    *   The **Vision Encoder** is typically **frozen**, but can optionally be **fine-tuned only during the first epoch** (controlled by `--train_ve_first_epoch`, mimicking the CheXagent strategy).
    *   Uses **(image, question, answer) triplets**.
    *   The model takes projected visual embeddings and question embeddings as input context.
    *   The loss is the causal language modeling loss, calculated only on the **answer tokens**.
    *   **Dynamic Padding:** Both question and answer sequences are dynamically padded per batch for efficiency.
*   **Key Scripts:**
    *   `Stage2/train_vqa_stage2.py`: Main training script.
    *   `Stage2/trainer.py`: Contains the `VQATrainerStage2` class.
    *   `Stage2/dataset.py`: Dataset definition for VQA triplets.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd Siglip # Or your repository's root directory name
    ```
2.  **Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Dependencies:** Create a `requirements.txt` file with necessary packages:
    Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Accelerator Configuration:** Configure 🤗 Accelerate for your hardware setup (CPU, single/multi-GPU):
    ```bash
    accelerate config
    ```

## Data Preparation

*   **Stage 1:** Prepare a directory containing CXR images and a JSON file. The JSON file should be a list of dictionaries, each mapping an `"image"` key (filename relative to image root) to a `"caption"` key (corresponding text).
*   **Stage 2:** Prepare a directory containing CXR images and a JSON file. The JSON file should be a list of dictionaries, each mapping an `"image"` key (filename) to a `"problem"` key (question/instruction) and a `"normal_caption"` key (the desired answer/response).
*   Place images in directories accessible via `--image_root`.
*   Place JSON files in accessible locations.

## Usage

Run training scripts using `accelerate launch`.

### Stage 1: Train Projector

```bash
accelerate launch Stage1/train_projection_stage1.py \
    --image_root /path/to/stage1/images \
    --train_json /path/to/stage1_data.json \
    --output_dir ./trained_projection_stage1 \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --llm_name "google/gemma-3-1b-it" \
    --batch_size <your_batch_size> \
    --learning_rate 1e-4 \
    --num_epochs <stage1_epochs> \
    --wandb_project "stage1_projector_training"
    # Adjust other args as needed (--weight_decay, --gradient_accumulation_steps, etc.)
```

### Stage 2: Train VQA Model

*(Requires trained projector from Stage 1)*

```bash
accelerate launch Stage2/train_vqa_stage2.py \
    --image_root /path/to/stage2/images \
    --train_json /path/to/stage2_data.json \
    --stage1_projector_path ./trained_projection_stage1/final_model \
    --output_dir ./trained_vqa_stage2 \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --llm_name "google/gemma-3-1b-it" \
    --batch_size <your_batch_size> \
    --learning_rate 2e-5 \
    --num_epochs <stage2_epochs> \
    --gradient_accumulation_steps <your_accumulation_steps> \
    --train_ve_first_epoch \ # OPTIONAL: Add this flag to train VE on epoch 1
    # --unfreeze_projection_layer # OPTIONAL: Add this flag to fine-tune projector
    # --unfreeze_llm is True by default
    --wandb_project "stage2_vqa_training"
    # Adjust other args as needed (--max_q_len, --max_a_len, --weight_decay, etc.)
```

### Inference

*(Requires trained models from Stage 2)*

```bash
python Stage2/inference_vqa_stage2.py \
    --llm_path ./trained_vqa_stage2/final_model/language_model \
    --projector_path ./trained_vqa_stage2/final_model/projection_layer \
    --image_path /path/to/your/test_image.jpg \
    --question "Describe any abnormalities in the lungs." \
    --vision_model_name "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli" \
    --device cuda \
    --max_new_tokens 256 \
    --num_beams 3 # Example generation param
    # Adjust other generation params (--temperature, --top_p, etc.)
```

## Key Scripts

*   **`projectors.py`**: Defines the MLP projector architecture.
*   **`accelerator_setup.py`**: Utility for setting up 🤗 Accelerate and Weights & Biases.
*   **`Stage1/`**: Contains code for projector training (dataset, trainer, main script).
*   **`Stage2/`**: Contains code for VQA fine-tuning (dataset, trainer, main script) and inference.

## Acknowledgements

*   This project draws inspiration from the methodologies presented in the CheXagent paper.
*   Utilizes pre-trained models (SigLIP, Gemma3) and libraries (Transformers, Accelerate, PyTorch) from Hugging Face and the broader open-source community.