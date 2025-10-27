#!/usr/bin/env python3
# coding: utf-8
import os
import torch
from torch.utils.data import Dataset
from einops import rearrange
from tqdm import trange
from twa.config import Config
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 36  # Global font enlargement
})
import numpy as np
import colorsys
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

slot_mask_cache = {}

def load_model_exclude_refiner(model, ckpt):
    state_dict = ckpt['model']
    # Filter out refiner module parameters
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('xxx')}

    # Load filtered parameters
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    print(f"[Info] Loaded model excluding refiner.")
    if missing_keys:
        print(f"[Warning] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys: {unexpected_keys}")

def get_mask_probs(epoch, max_epoch=10, max_strong=0.4, max_soft=0.6):
    # Linear increase until max_epoch
    ratio = min(epoch / max_epoch, 1.0)
    p_strong = ratio * max_strong
    p_soft = ratio * max_soft
    return p_strong, p_soft

# Removed unused build_slot_causal_mask and check_model_parameters functions

class LatentDataset(Dataset):
    def __init__(self, df, latent_dir, sam_dir=None):
        self.df = df.reset_index(drop=True)
        self.latent_dir = latent_dir
        self.sam_dir = sam_dir
        self.vid2id = {v: i for i, v in enumerate(self.df['filename'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid = self.df.iloc[idx]['filename']
        path = os.path.join(self.latent_dir, f"{vid}")
        sam_latent={'areas':None, 'masks':None}
        with open(path, 'rb') as f:
            latent = torch.load(f, map_location='cpu')  # Auto close file handle # [L, C, T, H, W]
        if self.sam_dir is not None:
            path = os.path.join(self.sam_dir, f"{vid}")
            with open(path, 'rb') as f:
                sam_latent = torch.load(f, map_location='cpu')

       # vid_id = self.vid2id[vid]
        return latent, vid, sam_latent['areas'], sam_latent['masks']

def custom_collate(batch):
    sam_masks_stacked=None
    sam_area_stacked=None
    latents, vid_ids, sam_area, sam_masks = zip(*batch)

    if Config.batch_size > 1:
        min_L = min([latent.shape[0] for latent in latents])
        latents = [latent[:min_L] for latent in latents]
    latents_stacked = torch.stack(latents, dim=0)
    if not sam_area:
        sam_area_stacked = torch.stack(sam_area, dim=0)
        sam_masks_stacked = torch.stack(sam_masks, dim=0)

    return latents_stacked, vid_ids, sam_area_stacked, sam_masks_stacked

# Removed unused check_model_parameters function

def init_slot_prototype(num_slots: int, dim: int):
    """
    Initialize slot prototype
    Args:
        num_slots: number of slot prototypes
        dim: slot vector dimension
    Returns:
        [num_slots, dim] tensor, normalized and fixed on sphere
    """
    torch.manual_seed(42)
    # Step 1: Gaussian sampling
    z = torch.randn(num_slots, dim)
    # z is the chaotic distribution of structural seeds mapping the world ontology
    return z # [N, D]

def get_param_groups(model, base_lr, weight_decay, layer_decay):
    param_groups = []
    assigned = set()

    # Fallback group for other unfrozen, unassigned parameters
    other_params = [p for n, p in model.named_parameters()
                    if id(p) not in assigned and p.requires_grad]

    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    print("[Debug] Parameter gradient status:")
    for name, param in model.named_parameters():
        print(f"{name:60s} | requires_grad = {param.requires_grad}")

    return param_groups

def debug_print_param_groups(model, param_groups, max_lines=10):
    """
    Print detailed information of param_groups for verifying LLRD grouping is correct.
    Args:
        model: nn.Module
        param_groups: optimizer param_groups (from get_param_groups_named)
        max_lines: maximum number of parameters to display per group
    """
    print("\n===== 🔍 Param Groups Debug Info =====")

    # Reverse mapping param → name
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    seen_ids = set()

    for i, group in enumerate(param_groups):
        lr = group.get("lr", "N/A")
        wd = group.get("weight_decay", "N/A")
        print(f"\n[Group {i}] lr={lr:.3e}, weight_decay={wd}, #params={len(group['params'])}")

        param_names = []
        for p in group["params"]:
            pid = id(p)
            if pid in seen_ids:
                param_names.append(f"(DUPLICATE) {param_to_name.get(pid, '<unnamed>')}")
            else:
                param_names.append(param_to_name.get(pid, "<unnamed>"))
                seen_ids.add(pid)

        # Print partial parameter names
        for name in param_names[:max_lines]:
            print(f"    - {name}")
        if len(param_names) > max_lines:
            print(f"    ... (+{len(param_names) - max_lines} more)")

    # Find missing parameters
    all_ids = set(id(p) for p in model.parameters() if p.requires_grad)
    missing = all_ids - seen_ids
    if missing:
        print(f"\n⚠️ WARNING: {len(missing)} parameters NOT assigned to any group!")
        for n, p in model.named_parameters():
            if id(p) in missing:
                print(f"    ⚠️ Unassigned: {n}")
    else:
        print("\n✅ All parameters are assigned without duplication.")

    print("=======================================\n")

def visualize_and_save_slots_heatmap(
    s_dmask: torch.Tensor,  # [B*T, H, N, P]
    save_dir: str,
    B: int,
    T: int,
    patch_hw: tuple = (40, 22),
    top_frames: int = 3,
    level_ratio: float = 0.8,  # Relative to max(attn) contour threshold
    max_slots=None,

):
    """
    Draw contour attention regions for all slots on each frame.

    Args:
        s_dmask: [B*T, H, N, P], attention distribution
        save_dir: save directory
        B, T: batch, time step
        patch_hw: patch resolution (H, W)
        top_frames: maximum number of frames to display
        max_slots: maximum number of slots to display
        level_ratio: relative threshold ratio for contour (0.6 means max(attn)*0.6)
    """
    os.makedirs(save_dir, exist_ok=True)
    Hpatch, Wpatch = patch_hw
    _, T, N, H, W = s_dmask.shape
    assert H * W == Hpatch * Wpatch

    s_attn = s_dmask

    cmap = plt.get_cmap('tab20')
    slot_colors = [cmap(i % 20) for i in range(max_slots)]

    for b in range(B):
        for t in range(min(T, top_frames)):
            fig, ax = plt.subplots(figsize=(Wpatch//2, Hpatch//2))
            ax.set_aspect('auto')
            drawn_any = False

            for n in range(min(N, max_slots)):
                attn_map = s_attn[b, t, n].cpu().numpy()  # [H, W]
                attn_max = attn_map.max()
                if attn_max < 1e-4:
                    continue  # Skip invalid slot

                level = attn_max * level_ratio
                attn_map = np.flipud(attn_map)
                cs = ax.contourf(
                    attn_map,
                    levels=[level,attn_max],
                    colors=[slot_colors[n]],
                    linewidths=1.5,
                    linestyles='solid',
                    alpha=0.45
                )
                ax.clabel(cs, inline=False, fontsize=26, fmt={level: f'slot {n}'})
                drawn_any = True

            if not drawn_any:
                ax.text(0.5, 0.5, "No slot activation above threshold", ha='center', va='center')

            ax.set_title(f"Contour Slot Attn - B{b}T{t}")
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'slot_contour_b{b}_t{t}.png'))
            plt.close(fig)
