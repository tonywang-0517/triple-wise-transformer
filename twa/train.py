#!/usr/bin/env python3
# coding: utf-8
import os
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from einops import rearrange, repeat
import math
import pandas as pd
from tqdm import tqdm, trange
from transformers import get_cosine_schedule_with_warmup
from twa.modules.utils import LatentDataset, custom_collate, get_param_groups, debug_print_param_groups, load_model_exclude_refiner, visualize_and_save_slots_heatmap, get_mask_probs
from twa.modules.modules import SlotBasedVideoModel
from twa.config import Config
from datetime import datetime

eps=1e-4
P = Config.blocks_per_frame
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
Config.tb_log_dir = f"{Config.tb_log_dir}_{timestamp}"

os.makedirs(Config.checkpoint_dir, exist_ok=True)
writer = SummaryWriter(Config.tb_log_dir)
scaler = GradScaler()

torch.autograd.set_detect_anomaly(False)

def train():
    model = SlotBasedVideoModel(Config, pre_trained=True).to(device)
    df = pd.read_csv(Config.metadata_csv).sample(frac=1, random_state=0)
    train_df = df.iloc[:int(0.95 * len(df))]
    val_df   = df.iloc[int(0.95 * len(df)):]
    train_loader = DataLoader(
        LatentDataset(train_df, Config.latent_dir),
        batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=True,
        collate_fn = custom_collate,
    )
    val_set = LatentDataset(val_df, Config.latent_dir)
    train_set = LatentDataset(train_df, Config.latent_dir)

    # optimizer, scheduler
    param_groups = get_param_groups(
        model=model,
        base_lr=Config.lr,
        weight_decay=Config.weight_decay,
        layer_decay=Config.layer_decay,
    )
    debug_print_param_groups(model, param_groups)
    optim = AdamW(param_groups)
    t_max = Config.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=int(0.1 * t_max),
        num_training_steps=t_max
    )

    # Checkpoint loading
    ckpt_path = os.path.join(Config.checkpoint_dir, Config.ckpt_name)
    start_epoch = 1
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        load_model_exclude_refiner(model, ckpt)
        # === Your original optimizer / scheduler recovery (keep comments) ===
        #optim.load_state_dict(ckpt['opt'])
        #scheduler.load_state_dict(ckpt['sch'])
        #start_epoch = ckpt['epoch']

    if Config.val_interval:
        model.eval()
        #latent_val, vid, _, _ = val_set[8]  # latent_val: [L, C, T, H, W]
        latent_val, vid, _, _ = train_set[666]  # latent_val: [L, C, T, H, W]

        latent_val = latent_val.unsqueeze(0).to(device)  # [1, L, C, T, H, W]

        latent_val = rearrange(latent_val, "b l c t h w -> b l (c t h w)")  # [1, L0, D]

        S = Config.val_steps  # rollout steps
        # pre-allocate buffer for context + rollout
        print(f"Starting autoregressive rollout for {S} steps ...")
        with torch.inference_mode(), autocast('cuda'):
            # current context = all available history
            #preds_flat, slots, confident_score= model(latent_val)  # [B, L, D]
            slots, confident_score = model.slot_encoder(latent_val)  # [B, T, N, D]
            x_noise = model.slot_decoder(slots)

        visualize_and_save_slots_heatmap(
            confident_score,
            save_dir="./confident_heatmaps",
            B=Config.batch_size,
            T=confident_score.shape[0],
            patch_hw=(Config.blocks_per_col, Config.blocks_per_row),
            top_frames=10,
            max_slots=Config.num_slot_pool,
            level_ratio=0.2
        )

        print(f"{vid} Heat map Generated")
        #B, T, N, H, W = confident_score.shape

       # tokens_out = tokens_out[:,:S, :]
       # create_video(tokens_out, vid)

        #preds_flat = (preds_flat * confident_score).sum(dim=1)
        print(mse_loss(x_noise, latent_val))
        preds_flat = x_noise[:, :880*4, :]
        # create_video(preds_flat, vid)  # Comment out video creation for now
        exit()

    global_step = 0
    model.train()

    for epoch in range(start_epoch, Config.num_epochs + 1):
        # === Weight health diagnosis ===
        #check_model_parameters(model)
        # === Training ===
        epoch_loss = 0.0
        with (tqdm(train_loader, desc=f"Epoch {epoch}") as bar):
            for latent_seq, vid, sam_area, sam_masks in bar:
                #latent_seq, vid = train_set[64]  ####
                latent_seq = latent_seq.to(device)
                #sam_masks = sam_masks.to(device)
                #latent_seq.unsqueeze_(0)

                tokens = rearrange(latent_seq, 'b l c t h w -> b l (c t h w)')
                #tokens = rearrange(latent_seq, "b c t (H h) (W w) -> b (t H W) (c h w)", w=4, h=4)  # [1, L0, D] lean
                B, L, D = tokens.shape
                strong_noise = torch.rand_like(tokens) * 2 - 1
                soft_noise = (torch.randn_like(tokens) * 0.32) * 2 - 1  # mse 0.11
                x_clean = tokens.clone().detach()
                x_mixed = x_clean.detach()
                z_true = x_clean
                # === Rollout ===
                loss_sum = 0.0
                recon_total = 0.0
                #rollout_times = 3 if epoch > Config.rollout_epoch else 1
                rollout_times = 1

                for rollout_index in range(rollout_times):
                    with (autocast(device_type='cuda')):
                        with torch.no_grad():
                            if rollout_index == 2:
                                x_mixed = x_clean
                                p_strong, p_soft = get_mask_probs(epoch)
                                # soft_noise_block is the version after adding noise
                                soft_noise_block = x_clean + soft_noise
                                # Generate mask and expand dimensions
                                rand_val = torch.rand(B, L, 1, device=device)
                                mask_strong = (rand_val < p_strong).expand(-1, -1, D)  # [B, L, D]
                                mask_soft = ((rand_val >= p_strong) & (rand_val < p_strong + p_soft)).expand(-1, -1, D)  # [B, L, D]
                                # --- 3. Construct x_mixed ---
                                # Apply different types of noise
                                # Replace mixed input
                                x_mixed[mask_strong] = strong_noise[mask_strong]
                                x_mixed[mask_soft] = soft_noise_block[mask_soft]
                                x_mixed[:, :Config.blocks_per_frame, :] = x_clean[:, :Config.blocks_per_frame, :]
                                x_mixed = x_mixed.detach()

                        slots, confident_score = model.slot_encoder(tokens)  # [B, T, N, D]
                        x_noise = model.slot_decoder(slots)

                        # === Reconstruction Loss Only ===
                        loss_recon = mse_loss(x_noise, z_true)

                        # === Total Loss ===
                        loss_total = loss_recon

                        # === Health Metrics ===
                        scaler.scale(loss_total).backward()

                        loss_sum += loss_total.item()
                        recon_total += loss_recon.item()

                        scaler.unscale_(optim)
                        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optim)
                        scaler.update()
                        optim.zero_grad()

                scheduler.step()
                global_step += 1
                epoch_loss += loss_total
                if global_step % 10 == 0:
                    writer.add_scalar('train/recon_loss', recon_total / rollout_times, global_step)
                    writer.add_scalar('train/loss_total', loss_sum / rollout_times, global_step)
                    writer.add_scalar('train/grad_norm', grad_norm, global_step)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

                    bar.set_postfix(loss=f"{loss_total:.4f}")

                if global_step % 500 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': model._orig_mod.state_dict() if Config.use_compile else model.state_dict(),
                        'opt': optim.state_dict(),
                        'sch': scheduler.state_dict()
                    }, ckpt_path)
                    print('model saved')

        writer.add_scalar('train/loss_epoch', epoch_loss / len(train_loader), epoch)
        torch.save({
            'epoch': epoch+1,
            'model': model._orig_mod.state_dict() if Config.use_compile else model.state_dict(),
            'opt': optim.state_dict(),
            'sch': scheduler.state_dict()
        }, ckpt_path)
        print('model saved')
    writer.close()

if __name__ == '__main__':
    train()
