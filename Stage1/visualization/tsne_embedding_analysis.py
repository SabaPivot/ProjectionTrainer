# MULT-LABEL 샘플 제거
# ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion'] 클래스만 포함
# 4개 class 중 수량이 제일 작은 class 숫자에 맞춰서 t-sne 그래프 제작
# Cursor에게 "Enable Projector t-sne라고 명령하면 projector t-sne 그래프도 생성"
import os
import argparse
import json
import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModel
import sys

# allow importing projectors from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from projectors import MLPProjector


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE on image and projector embeddings")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the data JSON file")
    parser.add_argument("--images_root", type=str, required=True, help="Root directory for image files")
    parser.add_argument("--model_name", type=str, default="StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", help="Pretrained XraySigLIP model name")
    parser.add_argument("--projector_ckpt", type=str, default=None, help="Path to trained projector checkpoint (.bin)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output plots")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores for MulticoreTSNE (-1 uses all)")
    return parser.parse_args()


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Filter to keep only specified single classes
    # ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion'] 클래스만 포함
    allowed_classes = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion"]
    initial_count = len(df)
    # Ensure the column exists and is string type before filtering
    if 'normal_caption' in df.columns:
        df['normal_caption'] = df['normal_caption'].astype(str).str.strip() # Clean whitespace
        df = df[df['normal_caption'].isin(allowed_classes)]

    filtered_count = len(df)
    print(f"Kept {filtered_count} rows belonging to classes: {allowed_classes}. Filtered out {initial_count - filtered_count}.")

    # Balance the dataset
    if filtered_count > 0:
        class_counts = df['normal_caption'].value_counts()
        min_samples = class_counts.min()
        print(f"Balancing classes to {min_samples} samples each.")
        # Use random_state for reproducible sampling
        df_balanced = df.groupby('normal_caption').sample(n=min_samples, random_state=42)
        df = df_balanced.reset_index(drop=True)
        print(f"Final balanced dataset size: {len(df)}")
    else:
        print("Warning: No data left after filtering, cannot balance.")

    # compute number of abnormalities: assume 'No Finding' means 0, else count comma-separated
    def count_ab(x):
        if isinstance(x, str) and x.lower().strip() == 'no finding':
            return 0
        return len([c for c in x.split(',') if c.strip()])
    df['num_abnormalities'] = df['normal_caption'].apply(count_ab)
    return df


def compute_image_embeddings(df, images_root, model, processor, device, batch_size=32):
    all_feats = []
    img_paths = df['image'].tolist()
    print(f"Computing embeddings for {len(img_paths)} images...")
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Computing Image Embeddings"):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        valid_indices_in_batch = [] # Keep track of which images loaded successfully
        for p in batch_paths:
            full = os.path.join(images_root, p)
            try:
                img = Image.open(full).convert('RGB')
                imgs.append(img)
                valid_indices_in_batch.append(True)
            except Exception as e:
                print(f"Failed to load image {full}: {e}")
                valid_indices_in_batch.append(False)
        if valid_indices_in_batch.count(True) > 0:
            inputs = processor(images=imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                # Call the vision sub-model directly
                vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
                # Use the pooled output for the final image embedding
                feats = vision_outputs.pooler_output
            feats = feats.cpu().numpy()
            all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats


def load_projector(ckpt_path, vision_dim, llm_dim):
    # load checkpoint
    state = torch.load(ckpt_path, map_location='cpu')
    # infer expansion factor
    w0 = state['model.0.weight']  # shape [inter, vision]
    expansion = w0.shape[0] // w0.shape[1]
    print(f"Expansion factor: {expansion}")
    projector = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim, expansion_factor=expansion)
    projector.load_state_dict(state)
    projector.eval()
    return projector


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    df = load_data(args.json_path)

    # load model and processor
    # Use AutoModel and AutoProcessor for SigLIP
    model, processor = AutoModel.from_pretrained(args.model_name, trust_remote_code=True), \
                      AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device).eval()

    # compute raw image embeddings
    print("Computing raw image embeddings...")
    raw_feats = compute_image_embeddings(df, args.images_root, model, processor, device, args.batch_size)
    print(f"Raw embeddings shape: {raw_feats.shape}")

    # if projector specified, compute projected embeddings
    proj_feats = None
    # if args.projector_ckpt:
    #     # load fine-tuned projector (supports directory or file)
    #     print("Loading fine-tuned projector...")
    #     proj_path = args.projector_ckpt
    #     # determine checkpoint file
    #     if os.path.isdir(proj_path):
    #         # check for safetensors or pytorch bin
    #         st_path = os.path.join(proj_path, 'model.safetensors')
    #         pt_path = os.path.join(proj_path, 'pytorch_model.bin')
    #         if os.path.exists(st_path): ckpt_file = st_path
    #         elif os.path.exists(pt_path): ckpt_file = pt_path
    #         else: ckpt_file = None
    #     else:
    #         ckpt_file = proj_path
    #     if ckpt_file is None:
    #         raise FileNotFoundError(f"No projector checkpoint found in {proj_path}")
    #     # load state_dict
    #     if ckpt_file.endswith('.safetensors'):
    #         from safetensors.torch import load_file as load_safetensors
    #         state_dict = load_safetensors(ckpt_file, device='cpu')
    #     else:
    #         state_dict = torch.load(ckpt_file, map_location='cpu')
    #     # adjust keys (remove module., add model. prefix)
    #     if all(k.startswith('module.') for k in state_dict):
    #         state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    #     if any(not k.startswith('model.') for k in state_dict):
    #         state_dict = {('model.' + k if not k.startswith('model.') else k): v for k, v in state_dict.items()}
    #     # infer dims and expansion
    #     w0 = state_dict['model.0.weight']  # [inter, vision]
    #     w2 = state_dict['model.2.weight']  # [llm, inter]
    #     vision_dim = w0.shape[1]
    #     expansion = w0.shape[0] // w0.shape[1]
    #     llm_dim = w2.shape[0]
    #     # init and load projector
    #     projector = MLPProjector(vision_dim=vision_dim, llm_dim=llm_dim, expansion_factor=expansion)
    #     projector.load_state_dict(state_dict)
    #     projector = projector.to(device).eval()
    #     # compute projected embeddings
    #     print("Computing projected embeddings...")
    #     with torch.no_grad():
    #         x = torch.from_numpy(raw_feats).to(device)
    #         proj_feats = projector(x).cpu().numpy()
    #     print(f"Projected embeddings shape: {proj_feats.shape}")

    # run t-SNE
    print("Running t-SNE on raw features...")
    n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count() # Handle -1 for n_jobs
    print(f"Using {n_jobs} cores for t-SNE...")
    # MulticoreTSNE doesn't support random_state in constructor
    tsne_raw = TSNE(n_components=2, n_jobs=n_jobs).fit_transform(raw_feats)
    # if proj_feats is not None:
    #     print("Running t-SNE on projected features...")
    #     tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(proj_feats)

    # prepare DataFrame for plotting
    df_plot = df.copy()
    df_plot['tsne_raw_1'] = tsne_raw[:, 0]
    df_plot['tsne_raw_2'] = tsne_raw[:, 1]
    # if proj_feats is not None:
    #     df_plot['tsne_proj_1'] = tsne_proj[:, 0]
    #     df_plot['tsne_proj_2'] = tsne_proj[:, 1]

    # Map class names to integers for coloring
    unique_classes = df_plot['normal_caption'].unique()
    class_to_int = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    df_plot['class_int'] = df_plot['normal_caption'].map(class_to_int)
    n_classes = len(unique_classes)

    # plot number of abnormalities
    # fig, axes = plt.subplots(1, 2 if proj_feats is not None else 1, figsize=(14, 6))
    # if proj_feats is not None:
    #     ax_raw, ax_proj = axes
    # else:
    #     ax_raw = axes

    # Plot only raw embeddings
    fig, ax_raw = plt.subplots(1, 1, figsize=(8, 6)) # Adjusted figure size

    # Use a qualitative colormap like 'tab10' or 'viridis'/'plasma' if many classes
    cmap = plt.get_cmap('viridis', n_classes)
    sc1 = ax_raw.scatter(df_plot['tsne_raw_1'], df_plot['tsne_raw_2'],
                         c=df_plot['class_int'], cmap=cmap, alpha=0.7)
    ax_raw.set_title('t-SNE (Raw Image Embeddings)')
    ax_raw.set_xlabel('t-SNE 1')
    ax_raw.set_ylabel('t-SNE 2')

    # if proj_feats is not None:
    #     sc2 = ax_proj.scatter(df_plot['tsne_proj_1'], df_plot['tsne_proj_2'],
    #                            c=df_plot['class_int'], cmap=cmap, alpha=0.7)
    #     ax_proj.set_title('t-SNE (Projected Embeddings)')
    #     ax_proj.set_xlabel('t-SNE 1')
    #     ax_proj.set_ylabel('t-SNE 2')

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls_name,
                          markerfacecolor=cmap(i), markersize=8) for i, cls_name in enumerate(unique_classes)]
    # if proj_feats is not None:
    #     fig.legend(handles=handles, title="Class", loc='center right', bbox_to_anchor=(1.05, 0.5))

    # Place legend inside the axes
    ax_raw.legend(handles=handles, title="Class", loc='best')

    plt.tight_layout() # Use standard tight_layout
    out_path = os.path.join(args.output_dir, 'tsne_class_distribution_raw_only.png') # Change output filename
    plt.savefig(out_path)
    print(f"Saved t-SNE plot to {out_path}")

if __name__ == '__main__':
    main()
