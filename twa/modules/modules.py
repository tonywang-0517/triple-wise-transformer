#!/usr/bin/env python3
# coding: utf-8
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
# Removed unused RoPE3D and RoPE2D imports
import math
from flash_attn import flash_attn_func
from torch.distributions import Normal
from typing import Tuple
from twa.config import Config
import torch.cuda.amp as amp
from .utils import init_slot_prototype
from .rope import Sinusoidal2DPositionEmbed
from torch.utils.checkpoint import checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TripleWiseAttention(nn.Module):
    """
    Triple-wise attention mechanism that splits K and V into K1,K2 and V1,V2
    """
    def __init__(self, dim: int, num_heads: int, head_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Separate projection matrices for K1, K2, V1, V2
        # Each projection outputs dim//2 to maintain total dim after concat
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k1_proj = nn.Linear(dim, dim // 2, bias=False)
        self.k2_proj = nn.Linear(dim, dim // 2, bias=False)
        self.v1_proj = nn.Linear(dim, dim // 2, bias=False)
        self.v2_proj = nn.Linear(dim, dim // 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
            causal: whether to use causal attention
        Returns:
            [B, L, D] output tensor
        """
        B, L, D = x.shape
        
        # Project to Q, K1, K2, V1, V2
        q = self.q_proj(x)  # [B, L, D]
        k1 = self.k1_proj(x)  # [B, L, D]
        k2 = self.k2_proj(x)  # [B, L, D]
        v1 = self.v1_proj(x)  # [B, L, D]
        v2 = self.v2_proj(x)  # [B, L, D]
        
        # Reshape to multi-head format
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads, d=self.head_dim)
        k1 = rearrange(k1, 'b l (h d) -> b l h d', h=self.num_heads, d=self.head_dim//2)
        k2 = rearrange(k2, 'b l (h d) -> b l h d', h=self.num_heads, d=self.head_dim//2)
        v1 = rearrange(v1, 'b l (h d) -> b l h d', h=self.num_heads, d=self.head_dim//2)
        v2 = rearrange(v2, 'b l (h d) -> b l h d', h=self.num_heads, d=self.head_dim//2)
        
        # Merge K1, K2 and V1, V2 using concatenation
        k_merged = torch.cat([k1, k2], dim=-1)  # Concatenate along head dimension
        v_merged = torch.cat([v1, v2], dim=-1)  # Concatenate along head dimension
        
        # Reshape for flash attention
        q = rearrange(q, 'b l h d -> b h l d')
        k_merged = rearrange(k_merged, 'b l h d -> b h l d')
        v_merged = rearrange(v_merged, 'b l h d -> b h l d')
        
        # Apply flash attention
        attn_out = flash_attn_func(q, k_merged, v_merged, causal=causal)
        
        # Reshape back and project output
        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        out = self.out_proj(attn_out)
        
        return out

# Removed unused SlotPredBlockV1 class

class CrossAttentionBlock(nn.Module):
    """
    Post-Norm Cross-Attention + Gated MLP Block.
    Supports both standard attention and triple-wise attention.

    1) Slot-to-token cross-attention
    2) DeepSeek-style gated feed-forward
    """
    def __init__(
        self,
        dim: int = Config.token_dim,
        drop_path_rate: float = 0.01,
        attn_dropout: float = 0.01,
        mlp_dropout: float = 0.01,
        mlp_ratio: float = Config.mlp_ratio,
        use_triple_wise: bool = Config.use_triple_wise_attention,
    ):
        super().__init__()
        self.num_heads = dim//64
        self.head_dim  = 64
        self.use_triple_wise = use_triple_wise

        # Post-attention LayerNorm
        self.attn_norm = nn.LayerNorm(dim)
        # Post-MLP LayerNorm
        self.mlp_norm  = nn.LayerNorm(dim)

        if use_triple_wise:
            # Triple-wise attention projections
            self.query_proj = nn.Linear(dim, dim, bias=False)
            self.k1_proj = nn.Linear(dim, dim // 2, bias=False)
            self.k2_proj = nn.Linear(dim, dim // 2, bias=False)
            self.v1_proj = nn.Linear(dim, dim // 2, bias=False)
            self.v2_proj = nn.Linear(dim, dim // 2, bias=False)
        else:
            # Standard attention projections
            self.query_proj = nn.Linear(dim, dim, bias=False)
            self.kv_proj    = nn.Linear(dim, 2 * dim, bias=False)

        # Dropout & Droppath
        self.attn_dropout      = nn.Dropout(attn_dropout)
        self.mlp_dropout       = nn.Dropout(mlp_dropout)
        self.drop_path_attn    = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path_mlp     = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Gated MLP
        hidden_dim     = int(dim * mlp_ratio)
        self.mlp_fc1   = nn.Linear(dim, hidden_dim)
        self.mlp_fc2   = nn.Linear(dim, hidden_dim)
        self.mlp_proj  = nn.Linear(hidden_dim, dim)

    def forward(self, slots: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots:  Tensor[B, T, K, D]  — slot queries
            tokens: Tensor[B, T, P, D]  — token features (memory)
        Returns:
            Tensor[B, T, K, D]
        """
        B, T, P, D = tokens.shape
        _, _, K, _ = slots.shape

        # --- Cross-Attention ---
        q = self.query_proj(slots)  # [B, T, K, D]
        q = rearrange(q, 'b t k (h d) -> (b t) k h d', h=self.num_heads, d=self.head_dim)

        if self.use_triple_wise:
            # Triple-wise attention: separate K1, K2, V1, V2 projections
            k1 = self.k1_proj(tokens)  # [B, T, P, D]
            k2 = self.k2_proj(tokens)  # [B, T, P, D]
            v1 = self.v1_proj(tokens)  # [B, T, P, D]
            v2 = self.v2_proj(tokens)  # [B, T, P, D]
            
            # Reshape to multi-head format
            k1 = rearrange(k1, 'b t p (h d) -> (b t) p h d', h=self.num_heads, d=self.head_dim//2)
            k2 = rearrange(k2, 'b t p (h d) -> (b t) p h d', h=self.num_heads, d=self.head_dim//2)
            v1 = rearrange(v1, 'b t p (h d) -> (b t) p h d', h=self.num_heads, d=self.head_dim//2)
            v2 = rearrange(v2, 'b t p (h d) -> (b t) p h d', h=self.num_heads, d=self.head_dim//2)
            
            # Merge K1, K2 and V1, V2 using concatenation
            k_merged = torch.cat([k1, k2], dim=-1)  # Concatenate along head dimension
            v_merged = torch.cat([v1, v2], dim=-1)  # Concatenate along head dimension
            
            attn_out = flash_attn_func(q, k_merged, v_merged, causal=False)  # [B*T, K, H, Dh]
        else:
            # Standard attention
            kv = self.kv_proj(tokens)  # [B, T, P, 2*D]
            kv = rearrange(
                kv, 'b t p (two h d) -> two (b t) p h d',
                two=2, h=self.num_heads, d=self.head_dim
            )
            k, v = kv
            attn_out = flash_attn_func(q, k, v, causal=False)  # [B*T, K, H, Dh]

        attn_out = self.attn_dropout(attn_out)
        attn_out = rearrange(
            attn_out, '(b t) k h d -> b t k (h d)',
            b=B, t=T
        )

        slot_attn = slots + self.drop_path_attn(attn_out)
        slot_attn = self.attn_norm(slot_attn)

        # --- Gated MLP ---
        u = self.mlp_fc1(slot_attn)
        v2 = self.mlp_fc2(slot_attn)
        gated = F.silu(u) * v2
        gated = self.mlp_dropout(gated)
        mlp_out = self.mlp_proj(gated)

        out = slot_attn + self.drop_path_mlp(mlp_out)
        out = self.mlp_norm(out)

        return out

class SlotControlBlock(nn.Module):
    """
    Post-Norm Cross-Attention (tokens→slots) + Gated MLP block.
    Supports both standard attention and triple-wise attention.
    1) Slots attend over token features.
    2) DeepSeek-style gated feed-forward.
    """
    def __init__(
        self,
        dim: int = Config.slot_dim,
        num_heads: int = Config.num_heads,
        mlp_ratio: float = Config.mlp_ratio,
        attn_dropout: float = 0.01,
        mlp_dropout: float = 0.01,
        drop_path_rate: float = 0.01,
        use_triple_wise: bool = Config.use_triple_wise_attention,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.use_triple_wise = use_triple_wise

        if use_triple_wise:
            # Triple-wise attention projections
            self.query_proj = nn.Linear(dim, dim, bias=False)
            self.k1_proj = nn.Linear(dim, dim // 2, bias=False)
            self.k2_proj = nn.Linear(dim, dim // 2, bias=False)
            self.v1_proj = nn.Linear(dim, dim // 2, bias=False)
            self.v2_proj = nn.Linear(dim, dim // 2, bias=False)
        else:
            # Standard attention projections
            self.query_proj = nn.Linear(dim, dim, bias=False)
            self.kv_proj    = nn.Linear(dim, 2 * dim, bias=False)

        # Attention & MLP dropouts
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.mlp_dropout    = nn.Dropout(mlp_dropout)

        # Stochastic depth
        self.drop_path_attn = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path_mlp  = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Post-attention and post-MLP LayerNorm
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm  = nn.LayerNorm(dim)

        # Gated MLP
        hidden_dim    = int(dim * mlp_ratio)
        self.fc1      = nn.Linear(dim, hidden_dim)
        self.fc2      = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.sin2d_pos = Sinusoidal2DPositionEmbed(dim=dim)

    def forward(self, tokens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor[B, L, D] — input token features
            slots:  Tensor[B, N, D] — slot queries
        Returns:
            Tensor[B, N, D] — updated slots
        """
        # --- Cross-Attention: slots query over tokens ---
        q = self.query_proj(tokens)  # [B, N, D]
        q = q + self.sin2d_pos()
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)

        if self.use_triple_wise:
            # Triple-wise attention: separate K1, K2, V1, V2 projections
            k1 = self.k1_proj(slots)   # [B, L, D]
            k2 = self.k2_proj(slots)   # [B, L, D]
            v1 = self.v1_proj(slots)   # [B, L, D]
            v2 = self.v2_proj(slots)   # [B, L, D]
            
            # Reshape to multi-head format
            k1 = rearrange(k1, 'b n (h d) -> b n h d', h=self.num_heads, d=self.head_dim//2)
            k2 = rearrange(k2, 'b n (h d) -> b n h d', h=self.num_heads, d=self.head_dim//2)
            v1 = rearrange(v1, 'b n (h d) -> b n h d', h=self.num_heads, d=self.head_dim//2)
            v2 = rearrange(v2, 'b n (h d) -> b n h d', h=self.num_heads, d=self.head_dim//2)
            
            # Merge K1, K2 and V1, V2 using concatenation
            k_merged = torch.cat([k1, k2], dim=-1)  # Concatenate along head dimension
            v_merged = torch.cat([v1, v2], dim=-1)  # Concatenate along head dimension
            
            attn_out = flash_attn_func(
                q, k_merged, v_merged,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=False,
                return_attn_probs=False
            )  # [B*N, H, Dh]
        else:
            # Standard attention
            kv = self.kv_proj(slots)   # [B, L, 2*D]
            kv = rearrange(
                kv, 'b n (two h d) -> two b n h d', two=2, h=self.num_heads)
            k, v = kv       # both [B*L, H, Dh]

            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=False,
                return_attn_probs=False
            )  # [B*N, H, Dh]

        attn_out = rearrange(attn_out, 'b l h d -> b l (h d)')

        # Residual + DropPath + LayerNorm
        residual = tokens + self.drop_path_attn(attn_out)
        residual = self.attn_norm(residual)

        # --- Gated MLP ---
        u = self.fc1(residual)
        v2 = self.fc2(residual)
        gated = F.silu(u) * v2
        gated = self.mlp_dropout(gated)
        mlp_out = self.out_proj(gated)

        # Residual + DropPath + LayerNorm
        x = residual + self.drop_path_mlp(mlp_out)
        x = self.mlp_norm(x)

        return x

class SlotEncoder(nn.Module):
    def __init__(
        self,
        tokens_per_frame: int = Config.blocks_per_frame,
        token_dim: int = Config.token_dim,
        slot_dim: int = Config.slot_dim,
        num_prototypes: int = Config.slot_book_size,
        num_selected: int = Config.num_slot_pool,
        fixed_return: bool = False,
        ema_beta: float = 0.01,
        noise_sigma: float = 0.2,
        mlp_ratio: int = Config.mlp_ratio,
        slot_iters: int = Config.slot_iters
    ):
        """
        tokens_per_frame: tokens per frame P
        token_dim: input token dimension
        slot_dim: prototype & slot dimension D
        num_prototypes: global prototype pool size N
        num_selected: number of prototypes selected per frame/per token K
        fixed_return: whether to skip noise sampling
        ema_beta: EMA decay coefficient
        noise_sigma: noise standard deviation
        """
        super().__init__()
        self.tokens_per_frame:int  = tokens_per_frame
        self.slot_dim:int = slot_dim
        self.num_prototypes:int  = num_prototypes
        self.num_selected :int = num_selected
        self.fixed_return = fixed_return
        self.ema_beta = ema_beta
        self.noise_sigma = noise_sigma
        self.ln_slot = nn.LayerNorm(slot_dim)
        self.ln_token = nn.LayerNorm(slot_dim)

        self.mask_ln = nn.LayerNorm(tokens_per_frame)

        # 1) Input projection
        self.input_norm = nn.LayerNorm(token_dim)
        self.input_proj = nn.Linear(token_dim, slot_dim)

        self.register_buffer('slot_prototypes', init_slot_prototype(num_selected, slot_dim))
        # 3) Core SlotAttention
        self.slot_attention_blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(slot_iters)])
        #self.self_attn = SelfAttention()
        mlp_dim  = slot_dim + tokens_per_frame
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(mlp_dim * mlp_ratio, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, slot_dim)
        )

        self.logit_mlp = nn.Sequential(
            nn.Linear(tokens_per_frame, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )

        # 4) Output normalization
        self.output_norm = nn.LayerNorm(slot_dim,elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(mlp_dim)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, T*P, token_dim]
        Returns:
          refined_slots: [B, T, num_selected, slot_dim]
          selected_idx:  [B, T, num_selected]
          attn_logits:   attention logits
          attn_mask:     attention mask
        """
        B, total_len, D = tokens.shape
        P = self.tokens_per_frame
        assert total_len % P == 0, f"{total_len} must be divisible by P={P}"
        T = total_len // P
        tokens = self.input_proj(tokens) # dimension up
        tokens_norm = self.input_norm(tokens)
        #x_norm = self.self_attn(tokens_norm) # first fuse spatiotemporal
        # === Execute spatial Self-Attn fusion per frame ===
        x_norm = rearrange(tokens_norm, 'b (t hw) d -> b t hw d', t=T)  # [B, T, HW, D]
        # === Initialize K slot queries per frame ===
        q = repeat(self.slot_prototypes, 'k d -> b t k d', b=B, t=T)  # [B, T, K, D]
        # === Multi-layer slot-token cross-attention ===
        for blk in self.slot_attention_blocks:
            q = blk(q, x_norm)  # [B, T, K, D]

        # === slot × token dot product to generate mask logit ===
        attn_logits = torch.einsum('btkd,bthd->btkh', q, x_norm) / (D ** 0.5)  # [B, T, K, HW]
        masks = F.softmax(attn_logits, dim=2)  # softmax over K slots smooth style
        masks = rearrange(masks, 'b t k (h w) -> b t k h w', h=Config.blocks_per_col, w=Config.blocks_per_row)
        normed_attn_logits = self.mask_ln(attn_logits)
        merged_slots = q + self.logit_mlp(normed_attn_logits)
        normed_merged_slots = self.output_norm(merged_slots)

        # — Optional noise injection —
        if not self.fixed_return:
            normed_merged_slots = Normal(loc=normed_merged_slots, scale=self.noise_sigma).rsample()

        return normed_merged_slots, masks

class SlotDecoder(nn.Module):
    def __init__(
            self,
            slot_dim: int = Config.slot_dim,
            num_patches: int = Config.blocks_per_frame,
            mlp_ratio: float = Config.mlp_ratio,
            drop: float = 0.01,
            drop_path: float = 0.01,
            token_dim: int = Config.token_dim,
            batch_size: int = Config.batch_size,
            token_refine_steps: int = Config.token_refine_steps,
            num_slot_pool:int = Config.num_slot_pool,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.num_slot_pool = num_slot_pool
        self.token_query = nn.Parameter(torch.empty(1, num_patches, slot_dim))
        self.slot_ln = nn.LayerNorm(slot_dim)
        nn.init.xavier_uniform_(self.token_query)
        # Multi-layer Cross-Attn Block (stackable)
        self.cross_blocks = nn.ModuleList([
            SlotControlBlock()
            for _ in range(token_refine_steps)
        ])
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.output_proj = nn.Linear(token_dim, token_dim)

    def forward(self, slots: torch.Tensor):
        """
        Args:
            slots: [B, N, D] structure latent slot
            B: batch size
        Returns:
            tokens: [B, 880, D] predicted token
        """
        slots = rearrange(slots, "b t n d -> (b t) n d")
        slots = self.slot_ln(slots)
        q = self.token_query.expand(slots.shape[0], -1, -1).contiguous()
        # Stack multiple Cross-Attn Blocks
        for blk in self.cross_blocks:
            q = blk(q, slots)

        token = self.output_proj(q)
        token = token * self.scale  # [B*T, P, 256]
        token = rearrange(token, '(b t) p d -> b (t p) d', b=self.batch_size)

        return token  # [B, T*P, D]

class SlotBasedVideoModel(nn.Module):
    def __init__(self, cfg, pre_trained=True, token_refine_steps = Config.token_refine_steps, val_interval=Config.val_interval):
        super().__init__()
        self.cfg = cfg
        self.pre_trained = pre_trained
        self.token_refine_steps = token_refine_steps
        self.blocks_per_frame = cfg.blocks_per_frame
        self.batch_size = cfg.batch_size
        self.training = not val_interval
        self.slot_encoder = SlotEncoder(fixed_return=(not pre_trained))
        self.slot_decoder = SlotDecoder()

    def forward(self, tokens, exact_slots=None,run_decoder=True):
        slots=None
        tokens_out=None
        confident_score=None
        run_slot_encoder = self.training or (exact_slots is None)
        if run_slot_encoder:
            if self.pre_trained:
                slots, confident_score = self.slot_encoder(tokens)  # [B, T, N, D]
            else:
                with torch.inference_mode():
                    slots, confident_score = self.slot_encoder(tokens)  # [B, T, N, D]

        if self.pre_trained:
            tokens_out = self.slot_decoder(slots)
            return tokens_out, slots, confident_score

        input_slots = exact_slots if exact_slots is not None else slots
        if run_decoder:
            tokens_out = self.slot_decoder(input_slots)  # [B, T*P, D]
        return tokens_out, input_slots, slots, confident_score
