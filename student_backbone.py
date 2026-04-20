"""
StudentBackbone: MobileNetV3-Large (image) + MobileNetV3-Small (LiDAR)
with INTERLEAVED BidirCrossAttn fusion replacing the teacher's GPT blocks.

Interleaved means fused features from stage i feed into stage i+1 of both
encoders — same hierarchical cross-modal refinement as the teacher's GPT.

    img_stem ──► img_stage0 ──► FUSE ──► img_stage1 ──► FUSE ──► ...
   lidar_stem ──► lidar_stage0 ──►      ──► lidar_stage1 ──►      ──► ...

BidirCrossAttn anchor budget per stage (student vs teacher):
    Teacher GPT  self-attn : 12×32 + 8×8  = 448 tokens → O(448²) ≈ 200 K
    Student cross-attn     :  4×8  + 4×4  =  48 tokens → O(32×16)×2 ≈   1 K
                                                          ~200× cheaper

ONNX-safe: no Python control flow on tensor values, no dynamic shapes.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from . import transfuser_utils as t_u


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_encoder_into_stages(encoder, n_stages: int):
    """
    Split a timm MobileNetV3Features encoder (features_only=True) into
    a stem module and n_stages sequential stage modules.

    The encoder must have been created with out_indices covering exactly
    n_stages feature levels starting from index 0.

    MobileNetV3Features internal layout:
        conv_stem → bn1 → act1   (stem)
        blocks[0] → blocks[1] → ...  (block groups, each a nn.Sequential)

    feature_info.info[i]['module'] = 'blocks.N' tells which block group
    is tapped for feature level i.  We slice encoder.blocks accordingly.

    Returns:
        stem:   nn.Sequential
        stages: nn.ModuleList of n_stages nn.Sequential modules
    """
    stem = nn.Sequential(encoder.conv_stem, encoder.bn1, encoder.act1)

    def _block_idx(module_str: str):
        parts = module_str.split('.')
        if parts[0] == 'blocks' and len(parts) >= 2:
            return int(parts[1])
        return None  # conv_head or similar — beyond blocks

    block_idxs = [
        _block_idx(encoder.feature_info.info[i]['module'])
        for i in range(n_stages)
    ]

    stages = nn.ModuleList()
    cursor = 0
    for bi in block_idxs:
        if bi is not None:
            sub = nn.Sequential(*list(encoder.blocks[cursor : bi + 1]))
            cursor = bi + 1
        elif cursor == 0:
            # Feature is the stem output itself (e.g. module='bn1' in MobileNetV3-Small).
            # The stem already produced this tensor; this stage is a no-op pass-through.
            sub = nn.Identity()
        else:
            # Feature is produced by conv_head (after all blocks)
            remaining = list(encoder.blocks[cursor:])
            head = []
            if hasattr(encoder, 'conv_head'):
                head = [encoder.conv_head, encoder.bn2, encoder.act2]
            sub = nn.Sequential(*(remaining + head))
        stages.append(sub)

    return stem, stages


# ──────────────────────────────────────────────────────────────────────────────
# Fusion block
# ──────────────────────────────────────────────────────────────────────────────

class BidirCrossAttn(nn.Module):
    """
    Single-scale bidirectional pooled cross-attention between image and LiDAR.

    Pipeline:
        1. Pool both maps to fixed anchor grids via AdaptiveAvgPool2d.
        2. Project to shared channel dim (rounded up to n_heads multiple).
        3. img ← lidar cross-attn  (img tokens query lidar tokens).
        4. lidar ← img cross-attn  (lidar tokens query img tokens).
        5. Project residuals back to original branch channels.
        6. Bilinear upsample to original spatial size and add residual.
    """

    def __init__(self, img_ch: int, lidar_ch: int, n_heads: int,
                 img_anchor_h: int, img_anchor_w: int,
                 lidar_anchor_h: int, lidar_anchor_w: int):
        super().__init__()
        self.n_heads = n_heads
        self.img_anchor   = (img_anchor_h,   img_anchor_w)
        self.lidar_anchor = (lidar_anchor_h, lidar_anchor_w)

        raw = max(img_ch, lidar_ch)
        self.shared_ch = math.ceil(raw / n_heads) * n_heads

        self.img_in   = nn.Conv2d(img_ch,   self.shared_ch, kernel_size=1)
        self.lidar_in = nn.Conv2d(lidar_ch, self.shared_ch, kernel_size=1)

        # img ← lidar
        self.img_q_proj    = nn.Linear(self.shared_ch, self.shared_ch)
        self.lidar_kv_proj = nn.Linear(self.shared_ch, self.shared_ch * 2)
        self.img_attn_out  = nn.Linear(self.shared_ch, self.shared_ch)
        self.img_norm      = nn.LayerNorm(self.shared_ch)

        # lidar ← img
        self.lidar_q_proj = nn.Linear(self.shared_ch, self.shared_ch)
        self.img_kv_proj  = nn.Linear(self.shared_ch, self.shared_ch * 2)
        self.lidar_attn_out = nn.Linear(self.shared_ch, self.shared_ch)
        self.lidar_norm     = nn.LayerNorm(self.shared_ch)

        self.img_out   = nn.Conv2d(self.shared_ch, img_ch,   kernel_size=1)
        self.lidar_out = nn.Conv2d(self.shared_ch, lidar_ch, kernel_size=1)

        self.img_pool   = nn.AdaptiveAvgPool2d(self.img_anchor)
        self.lidar_pool = nn.AdaptiveAvgPool2d(self.lidar_anchor)

    def _mhca(self, q_proj, kv_proj, out_proj, norm, q_feat, kv_feat):
        B, T_q, C = q_feat.shape
        T_kv = kv_feat.shape[1]
        hd = C // self.n_heads

        Q  = q_proj(q_feat).view(B, T_q,  self.n_heads, hd).transpose(1, 2)
        kv = kv_proj(kv_feat)
        K, V = kv.chunk(2, dim=-1)
        K = K.view(B, T_kv, self.n_heads, hd).transpose(1, 2)
        V = V.view(B, T_kv, self.n_heads, hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        return norm(q_feat + out_proj(out))

    def forward(self, img_feat: torch.Tensor, lidar_feat: torch.Tensor):
        B = img_feat.shape[0]
        ih, iw = img_feat.shape[2],   img_feat.shape[3]
        lh, lw = lidar_feat.shape[2], lidar_feat.shape[3]

        img_p   = self.img_pool(img_feat)
        lidar_p = self.lidar_pool(lidar_feat)

        img_s   = self.img_in(img_p)
        lidar_s = self.lidar_in(lidar_p)

        img_seq   = img_s.flatten(2).permute(0, 2, 1)
        lidar_seq = lidar_s.flatten(2).permute(0, 2, 1)

        img_att = self._mhca(self.img_q_proj, self.lidar_kv_proj,
                              self.img_attn_out, self.img_norm,
                              img_seq, lidar_seq)
        lidar_att = self._mhca(self.lidar_q_proj, self.img_kv_proj,
                                self.lidar_attn_out, self.lidar_norm,
                                lidar_seq, img_seq)

        img_att   = img_att.permute(0, 2, 1).view(
            B, self.shared_ch, *self.img_anchor)
        lidar_att = lidar_att.permute(0, 2, 1).view(
            B, self.shared_ch, *self.lidar_anchor)

        img_delta   = F.interpolate(self.img_out(img_att),
                                    size=(ih, iw), mode='bilinear',
                                    align_corners=False)
        lidar_delta = F.interpolate(self.lidar_out(lidar_att),
                                    size=(lh, lw), mode='bilinear',
                                    align_corners=False)

        return img_feat + img_delta, lidar_feat + lidar_delta


# ──────────────────────────────────────────────────────────────────────────────
# Student Backbone
# ──────────────────────────────────────────────────────────────────────────────

class StudentBackbone(nn.Module):
    """
    Interleaved dual-encoder backbone for the student model.

    Image  encoder : mobilenetv3_large_100  (ImageNet-pretrained)
                     out_indices=(0,1,2,3) → 4 clean stages, no conv_head bloat
    LiDAR  encoder : mobilenetv3_large_100  (random init, same arch as image)
                     lidar_input_proj: Conv2d(C_lidar→3) before encoder stem
    Fusion         : BidirCrossAttn × 4 stages, INTERLEAVED
    BEV FPN        : same 3-conv top-down structure as teacher

    Interleaved forward:
        img_stem  ──► img_stage0 ──► FUSE ──► img_stage1 ──► FUSE ──► ...
        lidar_stem ──► lidar_stage0 ──►       lidar_stage1 ──►         ...

    Output signature matches TransfuserBackbone:
        bev_feature_grid   [B, bev_features_chanels, 64, 64]  (training only)
        fused_features     [B, lidar_s3_ch, h_l, w_l]
        image_feature_grid [B, img_s3_ch,  h_i, w_i]
    """

    IMG_ARCH       = 'mobilenetv3_large_100'
    LIDAR_ARCH     = 'mobilenetv3_large_100'
    IMG_ANCHOR_H   = 4
    IMG_ANCHOR_W   = 8
    LIDAR_ANCHOR_H = 4
    LIDAR_ANCHOR_W = 4
    FUSION_HEADS   = 4
    N_STAGES       = 4

    def __init__(self, config):
        super().__init__()
        self.config = config

        C_lidar = (2 * config.lidar_seq_len if config.use_ground_plane
                   else config.lidar_seq_len)

        # ── encoders ──────────────────────────────────────────────────────────
        # out_indices=(0,1,2,3): 4 stages, avoids the 960-ch conv_head level
        _img_enc = timm.create_model(
            self.IMG_ARCH, pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3))

        # timm's features_only doesn't reliably forward in_chans for MobileNetV3,
        # so we use an explicit 1×1 projection before the encoder stem.
        self.lidar_input_proj = nn.Conv2d(C_lidar, 3, kernel_size=1, bias=False)
        _lidar_enc = timm.create_model(
            self.LIDAR_ARCH, pretrained=False, features_only=True,
            out_indices=(0, 1, 2, 3))

        # ── channel info (query before splitting) ─────────────────────────────
        img_chs   = [_img_enc.feature_info.info[i]['num_chs']
                     for i in range(self.N_STAGES)]
        lidar_chs = [_lidar_enc.feature_info.info[i]['num_chs']
                     for i in range(self.N_STAGES)]

        # ── split into interleaved stem + stage modules ───────────────────────
        # Parameters are owned by img_stem/img_stages — no longer need _img_enc
        self.img_stem,   self.img_stages   = _split_encoder_into_stages(
            _img_enc,   self.N_STAGES)
        self.lidar_stem, self.lidar_stages = _split_encoder_into_stages(
            _lidar_enc, self.N_STAGES)

        # ── 4× bidirectional cross-attention fusion ───────────────────────────
        self.fusions = nn.ModuleList([
            BidirCrossAttn(img_chs[i], lidar_chs[i], self.FUSION_HEADS,
                           self.IMG_ANCHOR_H,   self.IMG_ANCHOR_W,
                           self.LIDAR_ANCHOR_H, self.LIDAR_ANCHOR_W)
            for i in range(self.N_STAGES)
        ])

        # ── channel counts exposed to join / KD projectors ───────────────────
        self.num_features       = lidar_chs[3]   # → change_channel input
        self.num_image_features = img_chs[3]     # → image KD signal

        # ── perspective upsample factor (API compat, no perspective head) ─────
        self.perspective_upsample_factor = (
            _img_enc.feature_info.info[3]['reduction']
            // config.perspective_downsample_factor)

        # ── BEV FPN (same 3-conv top-down structure as teacher) ───────────────
        ch = config.bev_features_chanels
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=config.bev_upsample_factor,
            mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(
            size=(config.lidar_resolution_height // config.bev_down_sample_factor,
                  config.lidar_resolution_width  // config.bev_down_sample_factor),
            mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(ch, ch, kernel_size=(3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=(3, 3), padding=1)
        self.c5_conv  = nn.Conv2d(lidar_chs[3], ch, kernel_size=(1, 1))

    # ------------------------------------------------------------------
    def top_down(self, x: torch.Tensor) -> torch.Tensor:
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, lidar: torch.Tensor,
                timer=None):
        _t = timer

        img_feat   = (t_u.normalize_imagenet(image)
                      if self.config.normalize_imagenet else image)
        lidar_feat = self.lidar_input_proj(lidar)

        # ── stems ──────────────────────────────────────────────────────────
        if _t:
            with _t.measure('student_backbone/img_stem'):
                img_feat   = self.img_stem(img_feat)
            with _t.measure('student_backbone/lidar_stem'):
                lidar_feat = self.lidar_stem(lidar_feat)
        else:
            img_feat   = self.img_stem(img_feat)
            lidar_feat = self.lidar_stem(lidar_feat)

        # ── 4 interleaved stages: encode → fuse → encode → fuse → ... ──────
        for i in range(self.N_STAGES):
            if _t:
                with _t.measure(f'student_backbone/img_stage{i}'):
                    img_feat   = self.img_stages[i](img_feat)
                with _t.measure(f'student_backbone/lidar_stage{i}'):
                    lidar_feat = self.lidar_stages[i](lidar_feat)
                with _t.measure(f'student_backbone/bidir_fusion{i}'):
                    img_feat, lidar_feat = self.fusions[i](img_feat, lidar_feat)
            else:
                img_feat   = self.img_stages[i](img_feat)
                lidar_feat = self.lidar_stages[i](lidar_feat)
                img_feat, lidar_feat = self.fusions[i](img_feat, lidar_feat)

        # ── outputs ────────────────────────────────────────────────────────
        # BEV feature grid: training only (for KD + FPN); skipped at inference
        if self.training:
            if _t:
                with _t.measure('student_backbone/fpn_top_down'):
                    bev_feature_grid = self.top_down(lidar_feat)
            else:
                bev_feature_grid = self.top_down(lidar_feat)
        else:
            bev_feature_grid = None

        return bev_feature_grid, lidar_feat, img_feat
