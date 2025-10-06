# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, MaskDecoderDraft, PromptEncoder, TwoWayTransformer
from .modeling import SamDraft as Sam

def build_sam_vit_h(checkpoint=None, args=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        args=args,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, args=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        args=args,
    )


def build_sam_vit_b(checkpoint=None, args=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        args=args,
    )

from timm.models import create_model


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    args=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            in_chans=4,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                vatt_init=args.vatt_init,
                vatt_pos=args.vatt_pos,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        draft_decoder=MaskDecoderDraft(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                vatt_init=args.vatt_init,
                vatt_pos=args.vatt_pos,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53, 114.495],
        pixel_std=[58.395, 57.12, 57.375, 57.63],
        args=args,
    )
    sam.eval()
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        state_dict.pop('pixel_mean', None)
        state_dict.pop('pixel_std', None)

        new_state_dict = {}
        for key in state_dict.keys():
            if "mask_decoder." in key:
                new_key = key.replace("mask_decoder", "draft_decoder")
                new_state_dict[new_key] = state_dict[key]
            new_state_dict[key] = state_dict[key]
        
        # 拿到预训练的权重
        proj_weight = new_state_dict["image_encoder.patch_embed.proj.weight"]  # [768, 3, 16, 16]

        # 扩展到 4 通道
        if proj_weight.shape[1] == 3:
            extra_channel = proj_weight.mean(dim=1, keepdim=True)  # 用平均代替D通道的初始权重
            proj_weight_4 = torch.cat([proj_weight, extra_channel], dim=1)  # [768, 4, 16, 16]
            new_state_dict["image_encoder.patch_embed.proj.weight"] = proj_weight_4

        sam.load_state_dict(new_state_dict, strict=False)
    return sam

