"""
Stage 0: Fine-tuning Vision Encoder using Image-Text Contrastive Loss (SigLIP-style).

This script adapts a pre-trained vision encoder (like XraySigLIP)
by fine-tuning it on paired image-text data (using 'normal_caption' as text).
The goal is to make the vision encoder produce embeddings that are better aligned
with the semantics of the text descriptions, potentially leading to better
separation of concepts in downstream tasks or visualizations like t-SNE.

It uses a contrastive loss similar to CLIP/SigLIP.
The text encoder is kept frozen during this stage.
"""
import os
# Set tokenizer parallelism to False to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F # For cosine similarity & loss
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
from tqdm import tqdm
import logging
import math
from transformers import get_cosine_schedule_with_warmup, AutoProcessor, AutoModel, AutoTokenizer
import json
from PIL import Image
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adjust Python Path to find augmentation.py
import sys
# Add the parent directory of Stage0 to the Python path
# This allows importing augmentation.py which is in the workspace root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Online Augmentation Imports
import cv2 # For color conversion if using augmentation
# import numpy as np # Already imported above, for array conversion if using augmentation
try:
    from augmentation import apply_augmentation_pipeline, AUGMENTATION_PIPELINE # Import pipeline and its definition
except ImportError as e:
    logger.warning(f"Could not import augmentation module: {e}. Augmentation will be disabled.")
    apply_augmentation_pipeline = None
    AUGMENTATION_PIPELINE = None

# Setup Distributed Training
def init_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        logger.info(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        try:
            dist.init_process_group(backend='nccl')
            logger.info(f"Process group initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing process group: {e}")
            raise
            
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        logger.info(f"Process {rank}/{world_size} using device: {device}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        return True, rank, world_size, device
    else:
        logger.warning("Not running in distributed mode")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

def setup_wandb(args, rank):
    """Setup W&B logging."""
    if args.log_with.lower() == "wandb" and rank == 0 and not args.disable_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            logger.info(f"Initialized wandb with project: {args.wandb_project}")
            return wandb
        except ImportError:
            logger.warning("wandb not installed. Running without wandb logging.")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    return None

def cleanup():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Process group destroyed")

# Dataset Class
class ImageTextContrastiveDataset(Dataset):
    """Dataset for Image-Text Contrastive Learning."""
    def __init__(self, json_path, image_root, processor, tokenizer, max_text_len=77, 
                 use_online_augmentation=False, augmentation_pipeline_config=None, image_root_2=None):
        self.image_root = image_root
        self.image_root_2 = image_root_2
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.use_online_augmentation = use_online_augmentation
        self.augmentation_pipeline_config = augmentation_pipeline_config
        if self.use_online_augmentation and self.augmentation_pipeline_config is None:
            logger.warning("use_online_augmentation is True, but no augmentation_pipeline_config was provided. Augmentations will not be applied.")
            self.use_online_augmentation = False
        elif self.use_online_augmentation:
            logger.info("Online augmentation enabled for this dataset.")

        logger.info(f"Loading data from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            initial_count = len(df)
            logger.info(f"Initial dataset size: {initial_count}")
            self.data = df.copy()
            self.data['normal_caption'] = self.data['normal_caption'].astype(str)
            initial_len = len(self.data)
            self.data = self.data[self.data['normal_caption'].str.strip().str.len() > 0]
            filtered_len = len(self.data)
            if initial_len > filtered_len:
                logger.warning(f"Filtered out {initial_len - filtered_len} rows with empty/whitespace captions.")
            if len(self.data) == 0:
                logger.warning("Dataset is empty after filtering/processing!")
            if 'normal_caption' not in self.data.columns:
                raise ValueError("Dataset must contain a 'normal_caption' column for text data.")
            # Ensure text data is string (already done, but as a safeguard if logic changes)
            if not pd.api.types.is_string_dtype(self.data['normal_caption']):
                self.data['normal_caption'] = self.data['normal_caption'].astype(str)
            self.class_names = sorted(self.data['normal_caption'].unique().tolist())
            logger.info(f"Dataset has {len(self.class_names)} unique class names")
        except Exception as e:
            logger.error(f"Error loading dataset from {json_path}: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path_original = item['image']
        image_path = os.path.join(self.image_root, image_path_original)
        caption = item['normal_caption']
        default_return = {
            "pixel_values": torch.zeros((3, 224, 224), dtype=torch.float32),
            "input_ids": torch.zeros((self.max_text_len,), dtype=torch.long),
            "attention_mask": torch.zeros((self.max_text_len,), dtype=torch.long),
            "class_idx": torch.tensor(0, dtype=torch.long),
            "valid": torch.tensor(0, dtype=torch.bool)
        }
        try:
            found_image = False
            image = None
            if os.path.exists(image_path):
                if os.path.isdir(image_path):
                    jpg_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                    if jpg_files:
                        image_path = os.path.join(image_path, jpg_files[0])
                        image = Image.open(image_path).convert('RGB')
                        found_image = True
                else:
                    image = Image.open(image_path).convert('RGB')
                    found_image = True
            if not found_image and self.image_root_2:
                image_path = os.path.join(self.image_root_2, image_path_original)
                if os.path.exists(image_path):
                    if os.path.isdir(image_path):
                        jpg_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                        if jpg_files:
                            image_path = os.path.join(image_path, jpg_files[0])
                            image = Image.open(image_path).convert('RGB')
                            found_image = True
                    else:
                        image = Image.open(image_path).convert('RGB')
                        found_image = True
            if not found_image:
                logger.warning(f"Image not found in either root: {image_path_original} (tried {os.path.join(self.image_root, image_path_original)} and {os.path.join(self.image_root_2, image_path_original) if self.image_root_2 else 'N/A'})")
                return default_return
            if self.use_online_augmentation and self.augmentation_pipeline_config and apply_augmentation_pipeline is not None:
                try:
                    numpy_rgb_image = np.array(image)
                    numpy_bgr_image = cv2.cvtColor(numpy_rgb_image, cv2.COLOR_RGB2BGR)
                    augmented_bgr_image = apply_augmentation_pipeline(numpy_bgr_image, self.augmentation_pipeline_config)
                    augmented_rgb_image = cv2.cvtColor(augmented_bgr_image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(augmented_rgb_image)
                except Exception as aug_e:
                    logger.error(f"Error during online augmentation for {image_path}: {aug_e}. Using original image.")
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)
            if not caption or not caption.strip():
                logger.warning(f"Item {idx} (Image: {image_path}) has empty caption after load. Skipping.")
                return default_return
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
                return_attention_mask=True
            )
            if 'input_ids' not in text_inputs or 'attention_mask' not in text_inputs:
                logger.error(f"Tokenizer failed for item {idx} (Image: {image_path}, Caption: '{caption}'). Tokenizer output keys: {text_inputs.keys()}. Skipping.")
                return default_return
            input_ids = text_inputs['input_ids'].squeeze(0)
            attention_mask = text_inputs['attention_mask'].squeeze(0)
            class_idx = self.class_names.index(caption)
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "class_idx": class_idx,
                "valid": torch.tensor(1, dtype=torch.bool)
            }
        except Exception as e:
            logger.error(f"Error loading/processing item {idx} (Image: {image_path_original}, Caption: '{caption}'): {e}")
            return default_return

    def collate_fn(self, batch):
        valid_samples = [item for item in batch if item["valid"]]
        if not valid_samples:
            return {
                "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
                "input_ids": torch.zeros((1, self.max_text_len), dtype=torch.long),
                "attention_mask": torch.zeros((1, self.max_text_len), dtype=torch.long),
                "class_idx": torch.zeros((1,), dtype=torch.long),
                "valid_batch": torch.tensor(False, dtype=torch.bool)
            }
        pixel_values = torch.stack([item["pixel_values"] for item in valid_samples])
        input_ids = torch.stack([item["input_ids"] for item in valid_samples])
        attention_mask = torch.stack([item["attention_mask"] for item in valid_samples])
        class_idx = torch.tensor([item["class_idx"] for item in valid_samples], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "class_idx": class_idx,
            "valid_batch": torch.tensor(True, dtype=torch.bool)
        }

# SigLIP Loss
def siglip_loss(image_features, text_features, logit_scale, logit_bias=None):
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    logits = torch.matmul(image_features, text_features.t()) * logit_scale.exp()
    if logit_bias is not None:
        logits += logit_bias
    n = logits.size(0)
    labels = torch.eye(n, device=logits.device) # Correct labels for pairwise loss
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum') / n
    return loss

# Validation Functions
@torch.no_grad()
def evaluate_zero_shot_classification(model, processor, tokenizer, val_loader, device, class_names, is_distributed=False, rank=0):
    if rank == 0:
        logger.info(f"[VALIDATION] Entered evaluate_zero_shot_classification with {len(val_loader)} batches and {len(class_names)} classes")
    
    model.eval()
    all_preds = []
    all_labels = [] # Store actual ground truth labels (indices corresponding to class_names)
    
    if not class_names:
        if rank == 0:
            logger.warning("[VALIDATION] No class names provided for zero-shot evaluation. Skipping.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    text_templates = [f"a photo of {class_name}" for class_name in class_names]
    if rank == 0:
        logger.info(f"[VALIDATION] Created {len(text_templates)} text templates for zero-shot evaluation")
    
    text_inputs_tokenized = tokenizer(
        text_templates,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)
    if rank == 0:
        logger.info("[VALIDATION] Tokenized text templates and moved to device")
    
    text_features_encoded = None
    unwrapped_model = model.module if hasattr(model, 'module') else model

    if hasattr(unwrapped_model, 'text_model'):
        text_outputs = unwrapped_model.text_model(**text_inputs_tokenized)
        text_features_encoded = text_outputs.pooler_output
        if rank == 0:
            logger.info(f"[VALIDATION] Processed text features, shape: {text_features_encoded.shape}")
    else:
        if rank == 0:
            logger.warning("[VALIDATION] Model does not have 'text_model' attribute. Cannot perform zero-shot eval.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    text_features_normalized = F.normalize(text_features_encoded, p=2, dim=1)
    
    # Create a mapping from caption string to class index for ground truth
    caption_to_class_idx = {name: i for i, name in enumerate(class_names)}

    # Add a synchronization point here to ensure all processes are ready to start validation
    if is_distributed:
        dist.barrier()
        logger.info(f"[VALIDATION] Rank {rank} starting batch processing")

    for i, batch in enumerate(tqdm(val_loader, desc="Zero-shot Eval", leave=False, disable=(rank != 0))):
        if rank == 0 and i % 5 == 0:  # Log progress every 5 batches on rank 0
            logger.info(f"[VALIDATION] Processing batch {i}/{len(val_loader)}")
        
        if not batch["valid_batch"]:
            logger.warning(f"[VALIDATION] Rank {rank} skipping invalid batch at step {i}")
            continue
        
        pixel_values = batch["pixel_values"].to(device)
        target_indices = batch["class_idx"].to(device)
        
        try:
            image_outputs = unwrapped_model.vision_model(pixel_values=pixel_values)
            image_features_encoded = image_outputs.pooler_output
            image_features_normalized = F.normalize(image_features_encoded, p=2, dim=1)
            
            logit_scale_val = unwrapped_model.logit_scale.exp()
            logits = torch.matmul(image_features_normalized, text_features_normalized.t()) * logit_scale_val
            
            preds = torch.argmax(logits, dim=1)
            
            # Only extend if we have corresponding labels.
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(target_indices.cpu().tolist())
            
        except Exception as e:
            logger.error(f"[VALIDATION] Rank {rank}, batch {i}: Error during feature extraction: {e}")
            continue # Skip batch on error
    
    # Log completion of batch processing
    logger.info(f"[VALIDATION] Rank {rank} finished processing {len(val_loader)} batches, collected {len(all_preds)} predictions")
    
    if is_distributed:
        # Add a barrier to ensure all processes have finished batch processing
        dist.barrier()
        logger.info(f"[VALIDATION] Rank {rank} reached barrier after batch processing")
        
        # Fix to prevent deadlocks with different tensor sizes
        # Gather the number of predictions from each rank first
        local_size = torch.tensor([len(all_preds)], dtype=torch.long, device=device)
        all_sizes = [torch.ones(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)
        
        if rank == 0:
            logger.info(f"[VALIDATION] All prediction sizes: {[size.item() for size in all_sizes]}")
        
        # If any process has zero predictions, skip the gathering process
        if 0 in [size.item() for size in all_sizes]:
            logger.warning(f"[VALIDATION] Rank {rank} detected at least one process with zero predictions. Skipping gather.")
            if rank == 0:
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            else:
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # All processes have predictions, proceed with gathering
        # Convert lists to tensors for gathering
        preds_tensor = torch.tensor(all_preds, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)
        
        # Perform gathering with proper error handling
        try:
            # Pad to maximum size to ensure all tensors have the same shape
            max_size = max([size.item() for size in all_sizes])
            padded_preds = torch.zeros(max_size, dtype=torch.long, device=device)
            padded_labels = torch.zeros(max_size, dtype=torch.long, device=device)
            
            # Copy actual data
            padded_preds[:len(all_preds)] = preds_tensor
            padded_labels[:len(all_labels)] = labels_tensor
            
            # Create lists to store gathered tensors
            gathered_preds_list = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            gathered_labels_list = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            
            # Gather the padded tensors
            dist.all_gather(gathered_preds_list, padded_preds)
            dist.all_gather(gathered_labels_list, padded_labels)
            
            logger.info(f"[VALIDATION] Rank {rank} completed all_gather operations")
            
            if rank == 0:
                # Reconstruct the original predictions by using the sizes
                final_preds = []
                final_labels = []
                for i, size in enumerate(all_sizes):
                    size = size.item()
                    final_preds.extend(gathered_preds_list[i][:size].cpu().tolist())
                    final_labels.extend(gathered_labels_list[i][:size].cpu().tolist())
                
                all_preds = final_preds
                all_labels = final_labels
                logger.info(f"[VALIDATION] Rank 0 reconstructed {len(all_preds)} predictions from all processes")
            else:
                # Non-rank 0 processes don't calculate metrics
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                
        except Exception as e:
            logger.error(f"[VALIDATION] Error during all_gather operation on rank {rank}: {e}")
            if rank == 0:
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            else:
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Add a final barrier to ensure all processes finish validation together
    if is_distributed:
        dist.barrier()
        logger.info(f"[VALIDATION] Rank {rank} reached final validation barrier")

    if rank == 0: # Metrics calculation only on rank 0
        if all_preds and all_labels and len(all_preds) == len(all_labels):
            try:
                accuracy = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )
                logger.info(f"[VALIDATION] Metrics: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            except Exception as e:
                logger.error(f"[VALIDATION] Error calculating metrics: {e}")
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            logger.warning(f"[VALIDATION] No predictions or labels available for metric calculation, or lengths mismatch. Preds: {len(all_preds)}, Labels: {len(all_labels)}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0} # For non-rank 0 in non-dist should not happen


# The Main Training Function
def train_vision_encoder(args, is_distributed, rank, world_size, device, wandb_run=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if rank == 0:
        logger.info(f"Starting training with {world_size} GPUs")
        logger.info(f"Args: {args}")
    
    try:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        if rank == 0:
            logger.info(f"Loaded model, processor, and tokenizer from {args.model_name}")
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        # No cleanup needed here as it's handled by main's finally
        raise
    
    try:
        full_dataset = ImageTextContrastiveDataset(
            json_path=args.train_json,
            image_root=args.image_root,
            processor=processor,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
            use_online_augmentation=args.use_online_augmentation,
            augmentation_pipeline_config=AUGMENTATION_PIPELINE if (args.use_online_augmentation and AUGMENTATION_PIPELINE) else None,
            image_root_2=args.image_root_2
        )
        
        if rank == 0:
            logger.info(f"Loaded dataset with {len(full_dataset)} samples. Using {len(full_dataset.class_names)} unique class names (captions).")
            if len(full_dataset) == 0:
                logger.error("Full dataset is empty. Cannot proceed with training.")
                raise ValueError("Dataset is empty.")

        val_share = 0.05 # 5% for validation
        val_size = int(len(full_dataset) * val_share)
        if val_size == 0 and len(full_dataset) > 0 : # Ensure val_size is at least 1 if dataset is not tiny
             if len(full_dataset) * val_share > 0 : val_size = 1 
        train_size = len(full_dataset) - val_size

        if train_size <=0 :
            logger.error(f"Train size is {train_size} which is not valid. Full dataset: {len(full_dataset)}, Val size: {val_size}")
            raise ValueError("Train dataset size is not positive.")

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        if rank == 0:
            logger.info(f"Split into {len(train_dataset)} training and {len(val_dataset)} validation samples")
            if len(val_dataset) == 0:
                logger.warning("Validation dataset is empty. Validation metrics will be zero.")
            
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed) if is_distributed else None
        
        class_names = full_dataset.class_names # Used for zero-shot classification prompts
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise
    
    try:
        # num_workers = min(4, os.cpu_count() // world_size if world_size > 0 else os.cpu_count())
        # num_workers = max(1, num_workers) # Ensure at least 1 worker
        # if rank == 0:
        #     logger.info(f"Using {num_workers} dataloader workers per GPU")
        if rank == 0:
             logger.info(f"Setting num_workers=0 for DataLoader debugging.")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=0, # SET TO 0 FOR DEBUGGING
            collate_fn=full_dataset.collate_fn,
            pin_memory=True, drop_last=True
        )

        # Only create val_loader if val_dataset is not empty
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                num_workers=0, # SET TO 0 FOR DEBUGGING
                collate_fn=full_dataset.collate_fn, pin_memory=True
            )
            if rank == 0: logger.info(f"Created val_loader with {len(val_loader)} batches (num_workers=0).")
        elif rank == 0:
            logger.warning("Validation dataset is empty, val_loader not created. Validation will be skipped or yield zero metrics.")

        if rank == 0:
            logger.info(f"Created train_loader with {len(train_loader)} batches (num_workers=0).")
            if not val_loader : logger.info("val_loader is None.")

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise
    
    model = model.to(device)
    unwrapped_model_for_setup = model # For accessing attributes before DDP
    
    if args.freeze_text_encoder and hasattr(unwrapped_model_for_setup, 'text_model'):
        for param in unwrapped_model_for_setup.text_model.parameters():
            param.requires_grad = False
        if rank == 0: logger.info("Froze text encoder parameters")
            
    if args.freeze_logit_scale and hasattr(unwrapped_model_for_setup, 'logit_scale'):
        unwrapped_model_for_setup.logit_scale.requires_grad = False
        if rank == 0: logger.info("Froze logit_scale parameter")

    # Vision encoder layer freezing (example)
    if hasattr(unwrapped_model_for_setup, 'vision_model') and args.freeze_layers_ratio > 0.0:
        if hasattr(unwrapped_model_for_setup.vision_model, 'encoder') and hasattr(unwrapped_model_for_setup.vision_model.encoder, 'layers'):
            layers = unwrapped_model_for_setup.vision_model.encoder.layers
            num_layers_to_freeze = int(len(layers) * args.freeze_layers_ratio)
            if rank == 0: logger.info(f"Attempting to freeze {num_layers_to_freeze} out of {len(layers)} vision encoder layers.")
            for i, layer in enumerate(layers):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            if rank == 0: logger.info(f"Froze first {num_layers_to_freeze} layers of vision encoder.")
        elif rank == 0:
            logger.warning(f"Could not freeze vision encoder layers: model.vision_model.encoder.layers not found. Current ratio: {args.freeze_layers_ratio}")


    if is_distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True) # find_unused_parameters can be True if text encoder is frozen
        if rank == 0: logger.info("Wrapped model with DistributedDataParallel")
                
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        if trainable_params == 0:
            logger.error("No trainable parameters found! Check freezing logic.")
            raise ValueError("No trainable parameters found!")
                
    try:
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay)
        
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        max_train_steps = args.num_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(args.warmup_ratio * max_train_steps)
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
        )
        
        if rank == 0:
            logger.info(f"Optimizer: AdamW with lr={args.learning_rate}")
            logger.info(f"LR Scheduler: cosine with warmup, warmup_steps={num_warmup_steps}, total_steps={max_train_steps}")
    except Exception as e:
        logger.error(f"Error setting up optimizer and scheduler: {e}")
        raise
    
    # Training Loop
    global_step = 0
    best_val_accuracy = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Starting training for {args.num_epochs} epochs")
            
    for epoch in range(args.num_epochs):
        # --- Start of Epoch Logging ---
        logger.info(f"Rank {rank} starting Epoch {epoch+1}/{args.num_epochs}")

        if train_sampler is not None:
            logger.info(f"Rank {rank} calling train_sampler.set_epoch({epoch}) for Epoch {epoch+1}")
            train_sampler.set_epoch(epoch)
            logger.info(f"Rank {rank} finished train_sampler.set_epoch({epoch}) for Epoch {epoch+1}")
        
        # val_sampler set_epoch is not strictly needed if shuffle=False, but good practice
        if val_sampler is not None: 
            # No need to log this one extensively unless validation also hangs
            val_sampler.set_epoch(epoch) 
            
        model.train()
        logger.info(f"Rank {rank} set model to train mode for Epoch {epoch+1}")
        epoch_loss = 0.0
        processed_batches_in_epoch = 0

        # --- Start of Train Loader Iteration Logging ---
        logger.info(f"Rank {rank} entering train_loader iteration for Epoch {epoch+1}")
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(rank != 0), dynamic_ncols=True)

        for step, batch in enumerate(train_iter):
            # --- Start of Step Logging (only for first step) ---
            if step == 0:
                logger.info(f"Rank {rank} received first batch (step 0) for Epoch {epoch+1}")
            
            if not batch["valid_batch"]: # Skip if collate_fn marked batch as invalid
                if rank == 0: logger.warning(f"Skipping invalid batch at epoch {epoch+1}, step {step}")
                continue
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            try:
                # --- Forward Pass Logging (only for first step) ---
                if step == 0:
                    logger.info(f"Rank {rank} starting forward pass for Epoch {epoch+1}, Step 0")
                
                # Forward pass
                current_model_to_call = model.module if is_distributed else model
                
                outputs = current_model_to_call(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=False, # We calculate SigLIP loss manually
                    return_dict=True
                )
                
                image_features = outputs.image_embeds if hasattr(outputs, 'image_embeds') else None
                if image_features is None and hasattr(outputs, 'vision_model_output') and hasattr(outputs.vision_model_output, 'pooler_output'):
                    image_features = outputs.vision_model_output.pooler_output
                        
                text_features = outputs.text_embeds if hasattr(outputs, 'text_embeds') else None
                if text_features is None and hasattr(outputs, 'text_model_output') and hasattr(outputs.text_model_output, 'pooler_output'):
                    text_features = outputs.text_model_output.pooler_output

                if image_features is None or text_features is None:
                    logger.error(f"Epoch {epoch+1}, Step {step}: Could not extract image or text features. Skipping batch.")
                    if rank==0: logger.info(f"Output keys: {outputs.keys() if outputs else 'None'}")
                    continue
                
                loss = siglip_loss(
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=current_model_to_call.logit_scale, # Access from unwrapped model
                    logit_bias=getattr(current_model_to_call, 'logit_bias', None)
                )
                loss = loss / args.gradient_accumulation_steps

                if step == 0:
                    logger.info(f"Rank {rank} finished forward pass and loss calculation for Epoch {epoch+1}, Step 0")

            except Exception as e:
                logger.error(f"Error in forward/loss pass at epoch {epoch+1}, step {step}: {e}", exc_info=True)
                continue # Skip batch on error
            
            # --- Backward Pass Logging (only for first step) ---
            if step == 0:
                 logger.info(f"Rank {rank} calling loss.backward() for Epoch {epoch+1}, Step 0")
            
            # Backward and optimize
            try:
                loss.backward()
                if step == 0:
                    logger.info(f"Rank {rank} finished loss.backward() for Epoch {epoch+1}, Step 0")
            except Exception as backward_e:
                logger.error(f"Rank {rank} Error during loss.backward() at epoch {epoch+1}, step {step}: {backward_e}", exc_info=True)
                # Decide if we should break or continue, continuing might lead to more errors
                # For now, let's re-raise to halt execution on backward error
                raise backward_e

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps # Scale back for logging
            processed_batches_in_epoch += 1
            global_step += 1
            
            if global_step % args.logging_steps == 0 and rank == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch+1}, Step {global_step} - Loss: {loss.item() * args.gradient_accumulation_steps:.4f}, LR: {current_lr:.6e}")
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + (step +1) / len(train_loader) , # Fractional epoch
                        "train/step": global_step
                    })
            train_iter.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
        
        # End of epoch
        if processed_batches_in_epoch > 0:
            avg_epoch_loss = epoch_loss / processed_batches_in_epoch
        else:
            avg_epoch_loss = 0.0
            if rank == 0: logger.warning(f"Epoch {epoch+1} had no processed batches.")
            
        if rank == 0:
            logger.info(f"Epoch {epoch+1} completed - Avg Train Loss: {avg_epoch_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"train/epoch_loss": avg_epoch_loss, "train/epoch_completed": epoch + 1})
        
        # Run validation
        if val_loader is not None and len(val_loader) > 0:
            if rank == 0:
                logger.info(f"[VALIDATION] === Starting validation at end of epoch {epoch+1} ===")
                logger.info(f"[VALIDATION] val_loader has {len(val_loader)} batches.")
            
            try:
                val_metrics = evaluate_zero_shot_classification(
                    model=model, processor=processor, tokenizer=tokenizer, val_loader=val_loader,
                    device=device, class_names=class_names, is_distributed=is_distributed, rank=rank
                )
                
                if rank == 0:
                    logger.info(f"[VALIDATION] === Finished validation at end of epoch {epoch+1} === Metrics: {val_metrics}")
                    if wandb_run is not None:
                        wandb_run.log({
                            "val/accuracy": val_metrics["accuracy"], "val/precision": val_metrics["precision"],
                            "val/recall": val_metrics["recall"], "val/f1": val_metrics["f1"],
                            "val/epoch": epoch + 1
                        })
                    
                    # Save model if validation accuracy improved (and it's a valid metric)
                    if val_metrics["accuracy"] > 0 and val_metrics["accuracy"] > best_val_accuracy: # Ensure accuracy is somewhat meaningful
                        best_val_accuracy = val_metrics["accuracy"]
                        best_model_dir = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_model_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(best_model_dir)
                        processor.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir) # Save tokenizer too
                        logger.info(f"Saved new best model with accuracy: {best_val_accuracy:.4f} to {best_model_dir}")
            except Exception as e:
                logger.error(f"[VALIDATION] Error during validation process: {e}", exc_info=True)
                logger.warning("[VALIDATION] Continuing training despite validation failure")
                
                # Synchronize processes to avoid deadlock after validation error
                if is_distributed:
                    try:
                        dist.barrier()
                        logger.info(f"Rank {rank} synchronized after validation error")
                    except Exception as barrier_e:
                        logger.error(f"Error during barrier synchronization: {barrier_e}")
        elif rank == 0:
            logger.info(f"Skipping validation for epoch {epoch+1} as val_loader is None or empty.")

        # --- Synchronization Barrier BEFORE Saving ---
        # Ensure all processes wait here after validation is done by all ranks
        # before Rank 0 proceeds with potentially long saving operations.
        if is_distributed:
            logger.info(f"Rank {rank} reaching synchronization barrier BEFORE saving after epoch {epoch+1}")
            dist.barrier()
            logger.info(f"Rank {rank} passed synchronization barrier BEFORE saving after epoch {epoch+1}")

        # --- Save Best Model (Rank 0 only) ---
        # This now happens AFTER the barrier, so other ranks wait if Rank 0 saves.
        if rank == 0:
            # Check if validation happened and produced metrics
            if val_loader is not None and len(val_loader) > 0 and 'val_metrics' in locals():
                 if val_metrics["accuracy"] > 0 and val_metrics["accuracy"] > best_val_accuracy: # Ensure accuracy is somewhat meaningful
                        best_val_accuracy = val_metrics["accuracy"]
                        best_model_dir = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_model_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        try:
                            model_to_save.save_pretrained(best_model_dir)
                            processor.save_pretrained(best_model_dir)
                            tokenizer.save_pretrained(best_model_dir) # Save tokenizer too
                            logger.info(f"Saved new best model with accuracy: {best_val_accuracy:.4f} to {best_model_dir}")
                        except Exception as save_e:
                            logger.error(f"Error saving best model to {best_model_dir}: {save_e}")
            elif val_loader is None or len(val_loader) == 0:
                # Log if saving is skipped because validation didn't run
                logger.info(f"Skipping best model check for epoch {epoch+1} as validation did not run.")
            # Add any other necessary checks if val_metrics might not be defined

        # --- Save Checkpoint (Rank 0 only) ---
        # Also happens AFTER the barrier.
        if rank == 0 and args.save_every_n_epochs > 0 and ((epoch + 1) % args.save_every_n_epochs == 0 or epoch == args.num_epochs - 1):
            if (epoch + 1) >= args.min_save_epoch :
                checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                try:
                    model_to_save.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_dir}")
                except Exception as save_e:
                     logger.error(f"Error saving checkpoint model to {checkpoint_dir}: {save_e}")
                
        # --- [Barrier at the very end of the loop removed] ---
        # The primary barrier is now before saving.
    
    if rank == 0:
        logger.info("Training complete!")
        logger.info(f"Best validation accuracy achieved: {best_val_accuracy:.4f} (Note: accuracy metric for ZS classification needs careful label setup)")

# Main Function
def main():
    args = parse_args()
    
    is_distributed, rank, world_size, device = init_distributed()
    wandb_run = None # Initialize
    
    try:
        if rank == 0 : # Only rank 0 should try to init wandb
            wandb_run = setup_wandb(args, rank)
        train_vision_encoder(args, is_distributed, rank, world_size, device, wandb_run)
    except Exception as e:
        logger.error(f"Critical error in training process (rank {rank}): {e}", exc_info=True)
        # No explicit cleanup() here, finally block will handle it
        # Re-raise to ensure process exits with error status if not handled by torchrun
        raise 
    finally:
        if wandb_run is not None and rank == 0: # Ensure only rank 0 finishes wandb
            wandb_run.finish()
        if is_distributed:
            cleanup() # Destroy process group

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a vision encoder using SigLIP loss (Stage 0)")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name (e.g., google/siglip-so400m-patch14-224)")
    parser.add_argument("--train_json", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for image files")
    parser.add_argument("--image_root_2", type=str, default=None, help="Secondary root directory for image files")
    parser.add_argument("--output_dir", type=str, default="./trained_vision_encoder_stage0_contrastive", help="Directory to save model and logs")
    parser.add_argument("--max_text_len", type=int, default=77, help="Maximum sequence length for text tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler")
    parser.add_argument("--freeze_layers_ratio", type=float, default=0.0, help="Ratio of initial vision encoder layers to freeze (0.0 trains all)")
    parser.add_argument("--freeze_text_encoder", action=argparse.BooleanOptionalAction, default=True, help="Freeze text encoder. Default: True.")
    parser.add_argument("--freeze_logit_scale", action=argparse.BooleanOptionalAction, default=True, help="Freeze logit_scale. Default: True.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Allow loading models with custom code from Hugging Face Hub.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every N epochs (0 disables epoch checkpoints, only best/final)")
    parser.add_argument("--min_save_epoch", type=int, default=1, help="Minimum epoch to start saving checkpoints (applies if save_every_n_epochs > 0)")
    parser.add_argument("--wandb_project", type=str, default="vision_encoder_siglip_stage0", help="WandB project name")
    parser.add_argument("--log_with", type=str, default="wandb", choices=["wandb", "tensorboard", "all", "none"], help="Logging tracker")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every N global steps.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (defaults to W&B generated name)")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable WandB logging explicitly.")
    parser.add_argument("--use_online_augmentation", action='store_true', help="Enable on-the-fly data augmentation.")
    return parser.parse_args()

if __name__ == "__main__":
    main() 