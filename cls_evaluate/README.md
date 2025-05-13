# soombit

**Diagnostic Accuracy and Clinical Value of a Domain-specific Multimodal Generative AI Model for Chest Radiograph Report Generation**

This repository contains a reimplementation of the model and experimental pipeline described in:

> Hong, E. K., Ham, J., Roh, B., Gu, J., Park, B., Kang, S., ... & Kim, T. H. (2025). Diagnostic accuracy and clinical value of a domain-specific multimodal generative AI model for chest radiograph report generation. *Radiology*, 314(3), e241476.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Training Script](#training-script)
  - [Evaluation Script](#evaluation-script)
  - [Testing Script](#testing-script)
- [Code Highlights](#code-highlights)
  - [Data Loading & Preparation](#data-loading--preparation)
  - [Model Definitions](#model-definitions)
  - [Loss Functions](#loss-functions)
  - [Checkpointing & Best Model Selection](#checkpointing--best-model-selection)
  - [Dynamic Test JSON Selection](#dynamic-test-json-selection)
- [Citation](#citation)

## Project Structure

```
soombit/
├── data/                      # JSON metadata and image paths for train/val/test splits
├── checkpoints/               # Saved model checkpoints and `best_model.pth` files
├── models.py                  # Definitions for dataset, collate_fn, classifier, and evaluate()
├── train_utils.py             # Data loading, preprocessing, splitting logic
├── train.py                   # Training entrypoint; parses args and calls `main(args)`
├── train_twoway_loss.py       # Custom TwoWayMultiLabelLoss implementation
├── evaluate_experiment.py     # Script to evaluate both checkpoint and best-model files
├── test.py                    # Script for inference/testing on new data
├── run_experiments.sh         # Bash wrapper to kick off multiple configurations
└── README.md                  # This file
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repo_url> soombit
   cd soombit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

Use the provided shell script to run a suite of configurations:
```bash
bash run_experiments.sh
```

This wrapper will:
- Dynamically select the appropriate `TEST_JSON` based on the number of classes in the experiment.
- Pass flags such as `--class_names`, `--handle_abnormal`, and `--use_two_way_loss` to `soombit/train.py`.

### Training Script

```bash
python -m soombit.train \
  --data_json data/train_full.json \
  --class_names "Atelectasis,Cardiomegaly,Effusion" \
  --handle_abnormal positive_only \
  --lr 1e-4 \
  --epochs 50 \
  --batch_size 32
```

- **`--class_names`**: Comma-separated list of target classes. Samples not in this set are skipped.
- **`--handle_abnormal`**: Strategy for abnormal labels (`positive_only`, `binary`, etc.).
- **`--use_two_way_loss`**: Enable the `TwoWayMultiLabelLoss` variant.

### Evaluation Script

After training finishes, evaluate all saved checkpoints and the designated best model:
```bash
python -m soombit.evaluate_experiment \
  --exp_dir checkpoints/exp_xyz \
  --data_json data/test_*.json
```

This script will report metrics (AUROC, accuracy) for each checkpoint and highlight the best-performing model.

### Testing Script

For ad-hoc inference on new radiographs:
```bash
python -m soombit.test \
  --model_path checkpoints/exp_xyz/best_model.pth \
  --image_dir /path/to/images \
  --output_report report.json
```

## Code Highlights

### Data Loading & Preparation

Located in `soombit/train_utils.py`, the function `load_and_prepare_data()`:
1. Reads all samples from the provided JSON (which may contain 4+ classes).
2. Filters each sample based on `args.class_names` and `args.handle_abnormal`.
3. Splits the filtered set into training and validation subsets.

### Model Definitions

Extracted to `soombit/models.py`:
- `XrayClassificationDataset`
- `classification_collate_fn`
- `AbnormalityClassifier`
- `evaluate()`

### Loss Functions

In `soombit/train_twoway_loss.py`:
- **`TwoWayMultiLabelLoss`** implements separate positive and negative paths, unlike PyTorch’s built-in `pos_weight` which only scales the positive class term.

### Checkpointing & Best Model Selection

- **Periodic Checkpoints:** Saved every `N` steps/epochs.
- **Best Model:** Updated whenever validation AUROC improves.
- **Naming Convention:** `checkpoint_epoch_{EPOCH}.pth` and `best_model.pth` in each experiment folder.

### Dynamic Test JSON Selection

The `run_experiments.sh` script infers which test JSON to use based on the number of classes specified:
```bash
num_classes=$(echo "$class_names" | tr -cd , | wc -c)
if [ "$num_classes" -eq 2 ]; then
  TEST_JSON=data/test_binary.json
elif [ "$num_classes" -eq 3 ]; then
  TEST_JSON=data/test_3class.json
else
  TEST_JSON=data/test_full.json
fi
``` 

## Citation

If you use this code, please cite:


```bibtex
@article{hong2025diagnostic,
  title={Diagnostic accuracy and clinical value of a domain-specific multimodal generative AI model for chest radiograph report generation},
  author={Hong, E. K. and Ham, J. and Roh, B. and Gu, J. and Park, B. and Kang, S. and ... and Kim, T. H.},
  journal={Radiology},
  volume={314},
  number={3},
  pages={e241476},
  year={2025}
}
```

---
Feel free to open issues or pull requests for improvements!