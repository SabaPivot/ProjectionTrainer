# soombit/models.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# from transformers import AutoModel # Use AutoModel for flexibility - Reverting back
from transformers import SiglipVisionModel # Specific import for Siglip vision part
from PIL import Image
import os
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

class XrayClassificationDataset(Dataset):
    """Dataset for X-ray image classification using SigLIP."""
    def __init__(self, samples, image_root, class_names, processor, img_size, image_root_2=None,
                 handle_abnormal=False, abnormal_source_classes=None):
        self.image_root = image_root
        self.image_root_2 = image_root_2
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.processor = processor # Expect processor to be passed in
        self.img_size = img_size
        self.samples = samples
        self.handle_abnormal = handle_abnormal
        self.abnormal_source_classes = abnormal_source_classes or []

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.abnormal_target_idx = self.class_to_idx.get("Abnormal", -1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_filename = sample.get("image")
        original_label_str = sample.get("normal_caption", "").strip()

        target_index = -1
        target_label_str = original_label_str

        if self.handle_abnormal and original_label_str in self.abnormal_source_classes:
            target_label_str = "Abnormal"
            target_index = self.abnormal_target_idx
        elif target_label_str in self.class_to_idx:
            target_index = self.class_to_idx[target_label_str]

        if target_index == -1 and original_label_str:
             logger.warning(f"Sample {idx}: Original label \"{original_label_str}\" (mapped to \"{target_label_str}\") "
                            f"not found in effective CLASS_NAMES: {self.class_names}. Assigning index -1.")

        if not image_filename:
            logger.warning(f"Sample {idx} missing image filename. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if image_filename.startswith("p") and "/" in image_filename and self.image_root_2:
            image_path = os.path.join(self.image_root_2, image_filename)
            if os.path.isdir(image_path):
                try:
                    jpg_files = [f for f in os.listdir(image_path) if f.lower().endswith('.jpg')]
                    if not jpg_files:
                        logger.warning(f"No .jpg files found in directory: {image_path}. Skipping sample {idx}.")
                        return self.__getitem__((idx + 1) % len(self))
                    image_path = os.path.join(image_path, jpg_files[0])
                except Exception as e:
                     logger.error(f"Error listing files in {image_path} for sample {idx}: {e}", exc_info=True)
                     return self.__getitem__((idx + 1) % len(self))
        else:
            image_path = os.path.join(self.image_root, image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.pixel_values.squeeze(0)

            if target_index == -1:
                logger.warning(f"Sample {idx} has invalid target index (-1) for label '{original_label_str}'. Skipping.")
                return self.__getitem__((idx + 1) % len(self))

            return {
                "pixel_values": pixel_values,
                "target_index": torch.tensor(target_index, dtype=torch.long)
            }

        except FileNotFoundError:
            logger.warning(f"Image file not found for sample {idx}: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({image_path}): {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))

def classification_collate_fn(batch):
    """Stacks pixel_values and target_indices."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    target_indices = torch.stack([item["target_index"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "target_indices": target_indices
    }

class AbnormalityClassifier(nn.Module):
    def __init__(self, vision_model_name, class_names, num_heads=16, dropout_rate=0.1):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Revert to using SiglipVisionModel
        self.vision_model = SiglipVisionModel.from_pretrained(vision_model_name)
        # Get the embedding dimension from the vision model's config
        self.embed_dim = self.vision_model.config.hidden_size

        self.abnormality_queries = nn.Parameter(torch.randn(1, self.num_classes, self.embed_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classification_head = nn.Linear(self.embed_dim, 1)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # SiglipVisionModel output should be compatible again
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        image_features = vision_outputs.last_hidden_state

        batch_queries = self.abnormality_queries.repeat(batch_size, 1, 1)
        attn_output, _ = self.mha(batch_queries, image_features, image_features)

        attn_output_dropout = self.dropout(attn_output)
        logits = self.classification_head(attn_output_dropout)
        logits = logits.squeeze(-1)

        return logits 