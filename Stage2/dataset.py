import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)

class XrayVQADataset(Dataset):
    """Dataset for X-ray images, questions, and their corresponding answers from JSON"""
    def __init__(self, image_root, json_file, processor, tokenizer, img_size, max_q_len=128, max_a_len=512):
        """
        Args:
            image_root (str): Path to the directory containing images.
            json_file (str): Path to the JSON file containing the data triplets.
            processor: Vision processor for images.
            tokenizer: Language model tokenizer.
            img_size (int): Target size to resize images to (e.g., 384).
            max_q_len (int): Maximum token length for the question ('problem').
            max_a_len (int): Maximum token length for the answer ('normal_caption').
        """
        self.image_root = image_root
        self.img_size = img_size
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

        # Ensure tokenizer has a pad token; set to EOS if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token ({self.tokenizer.eos_token})")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
            logger.info(f"Loaded {len(self.samples)} samples from {json_file}")
        except FileNotFoundError:
            logger.error(f"JSON file not found at {json_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_file}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading JSON: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing processed image, tokenized question,
        and tokenized answer suitable for Stage 2 training.
        """
        try:
            sample = self.samples[idx]
            image_filename = sample.get("image")
            question_text = sample.get("problem")
            answer_text = sample.get("normal_caption")

            if not all([image_filename, question_text, answer_text]):
                logger.warning(f"Sample {idx} is missing required fields (image, problem, or normal_caption). Skipping.")
                return self.__getitem__((idx + 1) % len(self)) # Recursively get next valid item

            image_path = os.path.join(self.image_root, image_filename)

            # --- Image Processing ---
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size)) # Align with the vision encoder
            # Note: processor likely adds batch dim, remove it here, add back in dataloader collate_fn if needed
            image_inputs = self.processor(images=image, return_tensors="pt") # Pytorch tensor
            pixel_values = image_inputs.pixel_values.squeeze(0) # Remove batch dim

            # --- Text Tokenization ---
            # (padding handled in collate_fn)

            # Tokenize Question (problem)
            question_tokens = self.tokenizer(
                question_text,
                max_length=self.max_q_len,
                truncation=True,
                add_special_tokens=False # Vision encoder will add special tokens (<s>, </s> etc.)
            ).input_ids

            # Tokenize Answer (normal_caption)
            answer_tokens = self.tokenizer(
                answer_text,
                max_length=self.max_a_len,
                truncation=True,
            ).input_ids

            # The trainer will concatenate: [VISUAL_TOKENS] + [QUESTION_TOKENS] + [ANSWER_TOKENS]
            # The labels will be: [-100] * len(VISUAL+QUESTION) + [ANSWER_TOKENS] (with padding as -100)

            return {
                "pixel_values": pixel_values,
                "question_input_ids": torch.tensor(question_tokens, dtype=torch.long),
                "answer_input_ids": torch.tensor(answer_tokens, dtype=torch.long),
            }

        except FileNotFoundError:
            logger.warning(f"Image file not found for sample {idx}: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({image_path}): {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))
