import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import os
import json
import copy
import random
from tqdm import tqdm
# --- Augmentation Pipeline Definition (Moved to Global Scope) ---
SHIFT_X_MIN, SHIFT_X_MAX = -10, 10 # -5, 5
SHIFT_Y_MIN, SHIFT_Y_MAX = -10, 10 # -5, 5
SCALE_MIN, SCALE_MAX = 0.9, 1.1 # 0.95, 1.05
CONTRAST_MIN, CONTRAST_MAX = 0.8, 1.2 # 0.9, 1.1
ELASTIC_ALPHA_MIN, ELASTIC_ALPHA_MAX = 10, 20
ELASTIC_SIGMA_MIN, ELASTIC_SIGMA_MAX = 2, 3

# Forward declare functions used in AUGMENTATION_PIPELINE
def scale_image(image, zoom_factor):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Input image must be a 3-channel NumPy array (H, W, 3).")
    if not isinstance(zoom_factor, (float, int)) or zoom_factor <= 0:
        raise ValueError("zoom_factor must be a positive number.")
    height, width, _ = image.shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    if zoom_factor == 1.0:
        return image.copy()
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    if zoom_factor > 1.0:
        center_x, center_y = new_width // 2, new_height // 2
        start_x = max(0, center_x - width // 2)
        start_y = max(0, center_y - height // 2)
        cropped_image = resized_image[start_y:start_y + height, start_x:start_x + width, :]
        if cropped_image.shape[0] != height or cropped_image.shape[1] != width:
            cropped_image = cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_LINEAR)
        return cropped_image
    else:
        delta_w = width - new_width
        delta_h = height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if padded_image.shape[0] != height or padded_image.shape[1] != width:
            padded_image = cv2.resize(padded_image, (width,height), interpolation=cv2.INTER_LINEAR)
        return padded_image

def flip_image(image, direction='horizontal'):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Input image must be a 3-channel NumPy array (H, W, 3).")
    if direction == 'horizontal': return cv2.flip(image, 1)
    elif direction == 'vertical': return cv2.flip(image, 0)
    elif direction == 'both': return cv2.flip(image, -1)
    else: raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'.")

def shift_image(image, shift_x, shift_y, padding_mode='reflect'):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Input image must be a 3-channel NumPy array (H, W, 3).")
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    border_mode_map = {
        'reflect': cv2.BORDER_REFLECT_101, 'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE, 'wrap': cv2.BORDER_WRAP
    }
    cv_border_mode = border_mode_map.get(padding_mode.lower(), cv2.BORDER_REFLECT_101)
    border_value = [0, 0, 0]
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv_border_mode, borderValue=border_value)

def adjust_contrast(image, factor, method='random'): # method is unused but kept for signature consistency
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Input image must be a 3-channel NumPy array (H, W, 3).")
    if not isinstance(factor, (float, int)) or factor < 0:
        raise ValueError("Contrast factor must be a non-negative number.")
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def elastic_deformation(image, alpha, sigma):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Input image must be a 3-channel NumPy array (H, W, 3).")
    shape_img = image.shape
    shape_spatial = image.shape[:2]
    dx = (np.random.rand(*shape_spatial) * 2 - 1)
    dy = (np.random.rand(*shape_spatial) * 2 - 1)
    dx = cv2.GaussianBlur(dx, (0,0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0,0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape_spatial[1]), np.arange(shape_spatial[0]))
    indices_x = (x + dx).reshape(-1)
    indices_y = (y + dy).reshape(-1)
    distorted_image = np.zeros_like(image, dtype=image.dtype)
    for i in range(shape_img[2]):
        distorted_image[..., i] = map_coordinates(image[..., i], [indices_y, indices_x], order=1, mode='reflect').reshape(shape_spatial)
    return distorted_image

AUGMENTATION_PIPELINE = [
    {
        'name': 'RandomHorizontalFlip',
        'function': flip_image,
        'probability': 0.5, 
        'params_config': {'direction': 'horizontal'}
    },
    {
        'name': 'RandomScale',
        'function': scale_image,
        'probability': 1.0, 
        'params_config': {'zoom_factor': {'min': SCALE_MIN, 'max': SCALE_MAX}}
    },
    {
        'name': 'RandomShift',
        'function': shift_image,
        'probability': 1.0, 
        'params_config': {
            'shift_x': {'min': SHIFT_X_MIN, 'max': SHIFT_X_MAX},
            'shift_y': {'min': SHIFT_Y_MIN, 'max': SHIFT_Y_MAX},
            'padding_mode': 'reflect'
        }
    },
    {
        'name': 'RandomContrast',
        'function': adjust_contrast,
        'probability': 0.3, 
        'params_config': {'factor': {'min': CONTRAST_MIN, 'max': CONTRAST_MAX}}
    },
    {
        'name': 'ElasticTransform',
        'function': elastic_deformation,
        'probability': 0.2, 
        'params_config': {
            'alpha': {'min': ELASTIC_ALPHA_MIN, 'max': ELASTIC_ALPHA_MAX},
            'sigma': {'min': ELASTIC_SIGMA_MIN, 'max': ELASTIC_SIGMA_MAX}
        }
    }
]
# --- End of Augmentation Pipeline Definition ---

# --- Pipeline Application Function ---
def apply_augmentation_pipeline(image_data, pipeline_definition):
    augmented_image = image_data.copy()
    for transform_step in pipeline_definition:
        if random.random() < transform_step['probability']:
            params_to_apply = {}
            for param_name, config_value in transform_step.get('params_config', {}).items():
                if isinstance(config_value, dict) and 'min' in config_value and 'max' in config_value:
                    if isinstance(config_value['min'], float) or isinstance(config_value['max'], float):
                        params_to_apply[param_name] = random.uniform(config_value['min'], config_value['max'])
                    else: # Assume int for randint
                        params_to_apply[param_name] = random.randint(config_value['min'], config_value['max'])
                else:
                    params_to_apply[param_name] = config_value
            
            try:
                # print(f"Applying {transform_step['name']} with params: {params_to_apply}")
                augmented_image = transform_step['function'](augmented_image, **params_to_apply)
            except Exception as e:
                print(f"Error applying {transform_step['name']}: {e}. Skipping this step for the current image.")
                # Potentially return original image or last successful state if a step fails critically
                # For now, it continues with the image state before this failed step for the next transform
                pass # Or `continue` if this transform should not affect subsequent ones if it fails
    return augmented_image

# --- Main Data Processing Function (Modified for Pipeline) ---
def process_images_with_pipeline(input_json_path, 
                                 image_root_dir, 
                                 base_augmented_image_output_dir, 
                                 augmentation_pipeline_config):
    print(f"Loading JSON from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}"); return

    all_output_json_entries = []
    total_images = len(data)
    output_subdir = "pipelined" # Subdirectory for pipelined augmentations

    print(f"Starting pipelined augmentation for {total_images} images.")

    for sample in tqdm(data, total=total_images, desc="Processing images with pipeline"):
        original_relative_path = sample.get('image')
        if not original_relative_path:
            print(f"Warning: Skipping sample {sample.get('id', 'Unknown ID')} due to missing 'image' key.")
            all_output_json_entries.append(copy.deepcopy(sample))
            continue

        all_output_json_entries.append(copy.deepcopy(sample)) # Add original entry
        original_image_full_path = os.path.join(image_root_dir, original_relative_path)

        try:
            original_image_data = cv2.imread(original_image_full_path)
            if original_image_data is None: raise FileNotFoundError(f"Img not found: {original_image_full_path}")
            if original_image_data.ndim != 3 or original_image_data.shape[2] != 3:
                print(f"Warning: Skip augment for {original_relative_path}. Expected 3Ch img, got {original_image_data.shape}.")
                continue
        except Exception as e:
            print(f"Error loading image {original_image_full_path}: {e}. Skipping augmentations."); continue

        try:
            pipelined_augmented_img = apply_augmentation_pipeline(original_image_data, augmentation_pipeline_config)
            
            new_relative_path_for_json = os.path.join("Augmentation", output_subdir, original_relative_path)
            augmented_image_save_path = os.path.join(base_augmented_image_output_dir, output_subdir, original_relative_path)
            
            os.makedirs(os.path.dirname(augmented_image_save_path), exist_ok=True)
            cv2.imwrite(augmented_image_save_path, pipelined_augmented_img)
            
            new_sample_entry = copy.deepcopy(sample)
            new_sample_entry['image'] = new_relative_path_for_json
            new_sample_entry['augmentation_type'] = 'pipelined' # Indicate augmentation type
            all_output_json_entries.append(new_sample_entry)

        except Exception as e:
            print(f"Error during pipeline augmentation for {original_relative_path}: {e}")
        
    output_json_dir = os.path.dirname(input_json_path)
    output_json_filename = "pipelined_augmented_" + os.path.basename(input_json_path)
    output_json_full_path = os.path.join(output_json_dir, output_json_filename)
    
    print(f"Saving updated JSON to: {output_json_full_path}")
    try:
        with open(output_json_full_path, 'w') as f:
            json.dump(all_output_json_entries, f, indent=4)
        print("Pipelined augmentation process completed and new JSON file saved.")
    except Exception as e:
        print(f"Error saving JSON: {e}")
