"""Loads baseline / pruned / finetuned InternVL 3.5 checkpoints.

This mirrors the loader in ``InternVL/evaluate_internvl_pruned.py`` but is
kept self-contained so that the ``lavis_eval`` folder has no cross-file
dependencies on the rest of the InternVL code.
"""
from __future__ import annotations

import json
import logging
import os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

NUM_VIT_LAYERS = 24
NUM_LLM_LAYERS = 28


def _decoupled_internvit_attn_forward(self, x):
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    head_dim = q.shape[-1]
    scale = head_dim ** -0.5
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def decouple_internvit_qkv(model, device: str = "cpu") -> None:
    for name, module in model.vision_model.encoder.layers.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            qkv = module.qkv
            in_feat = qkv.in_features
            out_feat = qkv.out_features // 3
            has_bias = qkv.bias is not None

            q = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)
            k = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)
            v = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)

            q.weight.data = qkv.weight.data[:out_feat, :]
            k.weight.data = qkv.weight.data[out_feat : out_feat * 2, :]
            v.weight.data = qkv.weight.data[out_feat * 2 :, :]
            if has_bias:
                q.bias.data = qkv.bias.data[:out_feat]
                k.bias.data = qkv.bias.data[out_feat : out_feat * 2]
                v.bias.data = qkv.bias.data[out_feat * 2 :]

            module.q = q
            module.k = k
            module.v = v
            del module.qkv
            module.forward = partial(_decoupled_internvit_attn_forward, module)


def load_internvl(
    model_name: str,
    pruned_ckpt: Optional[str] = None,
    device: str = "cuda",
):
    """Load an InternVL 3.5 model, optionally substituting a pruned/finetuned ckpt.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id used for the base architecture and tokenizer
        (e.g. ``OpenGVLab/InternVL3_5-1B``).
    pruned_ckpt : Optional[str]
        Directory containing ``model.pt`` (state dict) and optionally
        ``pruned_shapes.json``. If ``None``, the baseline HF model is used.
    device : str
        Device to place the model on.
    """
    if pruned_ckpt is not None:
        ckpt_path = pruned_ckpt
        if os.path.isdir(ckpt_path):
            sd_path = os.path.join(ckpt_path, "model.pt")
            shapes_path = os.path.join(ckpt_path, "pruned_shapes.json")
        else:
            sd_path = ckpt_path
            shapes_path = None

        logger.info(f"Loading base architecture from: {model_name}")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )

        decouple_internvit_qkv(model, device="cpu")

        if shapes_path and os.path.exists(shapes_path):
            with open(shapes_path) as f:
                shape_map = {k: tuple(v) for k, v in json.load(f).items()}
            logger.info(f"Loaded shape metadata ({len(shape_map)} params)")
        else:
            tmp = torch.load(sd_path, map_location="cpu", weights_only=True)
            shape_map = {k: tuple(v.shape) for k, v in tmp.items()}
            del tmp

        for name, param in list(model.named_parameters()):
            if name not in shape_map:
                continue
            pruned_shape = shape_map[name]
            if tuple(param.shape) == pruned_shape:
                continue
            parts = name.rsplit(".", 1)
            parent = model
            for p in parts[0].split("."):
                parent = getattr(parent, p)
            attr = parts[1]
            if attr == "weight" and isinstance(parent, nn.Linear):
                parent.out_features = pruned_shape[0]
                parent.in_features = pruned_shape[1]
                parent.weight = nn.Parameter(torch.empty(pruned_shape, dtype=param.dtype))
                if parent.bias is not None and f"{parts[0]}.bias" in shape_map:
                    parent.bias = nn.Parameter(torch.empty(pruned_shape[0], dtype=param.dtype))
            elif attr == "bias" and isinstance(parent, nn.Linear):
                parent.bias = nn.Parameter(torch.empty(pruned_shape, dtype=param.dtype))
            elif attr == "weight" and hasattr(parent, "normalized_shape"):
                parent.normalized_shape = pruned_shape
                parent.weight = nn.Parameter(torch.empty(pruned_shape, dtype=param.dtype))
                if parent.bias is not None:
                    parent.bias = nn.Parameter(torch.empty(pruned_shape, dtype=param.dtype))
            else:
                setattr(parent, attr, nn.Parameter(torch.empty(pruned_shape, dtype=param.dtype)))

        state_dict = torch.load(sd_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        del state_dict

        vit_config = model.vision_model.config
        orig_vit_head_dim = vit_config.hidden_size // vit_config.num_attention_heads
        for i in range(NUM_VIT_LAYERS):
            attn = model.vision_model.encoder.layers[i].attn
            if hasattr(attn, "q") and orig_vit_head_dim > 0:
                attn.num_heads = attn.q.weight.shape[0] // orig_vit_head_dim

        for i in range(NUM_LLM_LAYERS):
            attn = model.language_model.model.layers[i].self_attn
            head_dim = attn.head_dim
            if head_dim > 0:
                new_num_heads = attn.q_proj.weight.shape[0] // head_dim
                new_num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
                attn.num_heads = new_num_heads
                attn.num_key_value_heads = new_num_kv_heads
                if hasattr(attn, "config"):
                    attn.config.num_attention_heads = new_num_heads
                    attn.config.num_key_value_heads = new_num_kv_heads

        logger.info("Pruned model loaded.")
    else:
        logger.info(f"Loading full model: {model_name}")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Total parameters: {total_params:,}")
    return model, tokenizer
