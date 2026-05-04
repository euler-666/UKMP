"""
UKMP Pruning for InternVL 3.5 1B
Adapts the UKMP structured pruning framework to InternVL (InternViT-300M + MLP + Qwen3-0.6B).
"""

import argparse
import json
import os
import random
from functools import partial

import internvl_lib.compression.torch_pruning as tp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from internvl_lib.common.logger import LoggerWithDepth
from internvl_lib.compression.pruners import mask_pruner as mask_pruner
from internvl_lib.compression.pruners.mask_pruner import (
    gqa_attention_head_mask_pruner,
    layer_norm_mask_pruner,
    linear_mask_pruner,
    param_mask_pruner,
    rms_norm_mask_pruner,
)

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

NUM_VIT_LAYERS = 24
NUM_LLM_LAYERS = 28
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_pil(image, input_size=448, max_num=1):
    """Load pixel values from a PIL Image with optional multi-tile preprocessing."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_param(model):
    param_counts = {"out": {}, "in": {}, "total": {}}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_counts["out"][name] = module.weight.shape[0]
            param_counts["in"][name] = module.weight.shape[1]
            param_counts["total"][name] = module.weight.numel()
    return param_counts


# ---------------------------------------------------------------------------
# InternViT QKV decoupling (fused qkv -> separate q, k, v)
# ---------------------------------------------------------------------------
def decoupled_internvit_attn_forward(self, x):
    """Replacement forward for InternAttention after decoupling fused qkv."""
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    scale = (C // self.num_heads) ** -0.5
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def decouple_internvit_qkv(model, device):
    """Split fused qkv linear in each InternViT attention layer into separate q, k, v."""
    for name, module in model.vision_model.encoder.layers.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            qkv = module.qkv
            in_feat = qkv.in_features
            out_feat = qkv.out_features // 3
            has_bias = qkv.bias is not None

            q = nn.Linear(in_feat, out_feat, bias=has_bias).to(device)
            k = nn.Linear(in_feat, out_feat, bias=has_bias).to(device)
            v = nn.Linear(in_feat, out_feat, bias=has_bias).to(device)

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
            module.forward = partial(decoupled_internvit_attn_forward, module)

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Selected causal-LM loss (UKMI-aligned: focus on poorly-fitted tokens)
# ---------------------------------------------------------------------------
def selected_causal_lm_forward(original_forward, self, *args, **kwargs):
    """Wraps InternVLChatModel.forward to apply token-selected loss."""
    outputs = original_forward(*args, **kwargs)
    if outputs.loss is None:
        return outputs

    logits = outputs.logits
    labels = kwargs.get("labels", None)
    if labels is None and len(args) > 0:
        labels = args[-1] if isinstance(args[-1], torch.Tensor) else None
    if labels is None:
        return outputs

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Only consider answer token positions (where labels != -100)
    # for threshold computation, matching T5's decoder-only behavior.
    valid_mask = shift_labels != -100
    sm_p = F.softmax(shift_logits, dim=-1)
    max_values, _ = torch.max(sm_p, dim=-1)

    valid_max = max_values[valid_mask]
    if valid_max.numel() == 0:
        return outputs
    thresh = max(0.4, torch.min(valid_max).item())

    select_mask = (max_values <= thresh) & valid_mask
    indices = select_mask.nonzero(as_tuple=True)
    selected_logits = shift_logits[indices[0], indices[1], :]
    selected_labels = shift_labels[indices[0], indices[1]]

    if selected_logits.numel() == 0:
        return outputs

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="mean")
    loss = loss_fct(
        selected_logits.view(-1, selected_logits.size(-1)), selected_labels.view(-1)
    )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------
class CalibrationDataset(Dataset):
    """Simple calibration dataset for InternVL pruning.
    Expects a JSON file with entries like: [{"image": "path/to/img.jpg", "conversations": [...]}]
    or the CC595K format: [{"image": "path", "caption": "text"}].
    """

    def __init__(self, data_path, image_root, tokenizer, model, max_samples=1000,
                 max_length=512, image_size=448):
        with open(data_path, "r") as f:
            raw = json.load(f)
        self.data = raw[:max_samples]
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item.get("image", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = load_image_pil(image, input_size=self.image_size, max_num=1)
        except Exception:
            pixel_values = torch.zeros(1, 3, self.image_size, self.image_size)

        if "conversations" in item:
            question_parts = []
            answer_parts = []
            for turn in item["conversations"]:
                val = turn.get("value", turn.get("content", ""))
                val = val.replace("<image>", "").strip()
                if turn.get("from") == "human":
                    question_parts.append(val)
                else:
                    answer_parts.append(val)
            question = " ".join(question_parts) if question_parts else "Describe this image."
            text = " ".join(answer_parts) if answer_parts else "Describe this image."
        else:
            question = "Describe this image."
            text = item.get("caption", item.get("text", ""))

        return {"pixel_values": pixel_values, "text": text, "question": question}


def build_calibration_loader(args, tokenizer, model):
    """Build a DataLoader that yields batches ready for InternVL forward pass."""
    dataset = CalibrationDataset(
        data_path=args.calib_data_path,
        image_root=args.calib_image_root,
        tokenizer=tokenizer,
        model=model,
        max_samples=args.num_examples,
        image_size=448,
    )

    def collate_fn(batch):
        return batch  # return list of dicts; we process per-sample in the loop

    return DataLoader(
        dataset,
        batch_size=args.calibration_bs,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )


def _build_prompt_from_template(model, question, answer=None):
    """Build a prompt string using the model's own conversation template.

    Mirrors how model.chat() constructs prompts so that train/eval formatting
    is always consistent with the HuggingFace-shipped template.

    Returns (full_prompt, assistant_prompt).
    """
    try:
        template_name = model.template
    except AttributeError:
        template_name = model.config.template
    from importlib import import_module
    mod = import_module(type(model).__module__.rsplit(".", 1)[0] + ".conversation")
    template = mod.get_conv_template(template_name)

    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], answer)
    full_prompt = template.get_prompt()

    assistant_role = template.roles[1]
    if answer is not None:
        assistant_prompt = assistant_role + answer + template.sep
    else:
        assistant_prompt = assistant_role

    return full_prompt, assistant_prompt


def prepare_internvl_inputs(samples, model, tokenizer, device):
    """Convert a list of {pixel_values, text, question} dicts into model-ready inputs with labels."""
    pixel_values_list = []
    input_ids_list = []
    labels_list = []
    num_patches_list = []

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    for sample in samples:
        pv = sample["pixel_values"]
        text = sample["text"]
        user_question = sample.get("question", "Describe this image.")

        pixel_values_list.append(pv)
        num_patches = pv.shape[0]
        num_patches_list.append(num_patches)

        num_image_tokens = model.num_image_token * num_patches
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens + IMG_END_TOKEN
        question = f"{image_tokens}\n{user_question}"

        full_prompt, assistant_prompt = _build_prompt_from_template(
            model, question, text
        )

        tokenized = tokenizer(full_prompt, return_tensors="pt", truncation=True,
                              max_length=512, padding="max_length")
        input_ids = tokenized["input_ids"].squeeze(0)

        unpadded_ids = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        assistant_ids = tokenizer(assistant_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        prefix_len = len(unpadded_ids) - len(assistant_ids)

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
    input_ids = torch.stack(input_ids_list).to(device)
    labels = torch.stack(labels_list).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    image_flags = torch.ones(pixel_values.shape[0], 1, device=device, dtype=torch.long)

    return {
        "pixel_values": pixel_values.to(model.dtype),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_flags": image_flags,
    }


def prepare_internvl_inputs_simple(model, tokenizer, device, batch_size=1):
    """Prepare minimal dummy inputs for FLOPs analysis / graph tracing."""
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if img_context_token_id is None:
        img_context_token_id = 0
    model.img_context_token_id = img_context_token_id

    num_image_tokens = model.num_image_token  # 256 per tile, 1 tile minimum
    pixel_values = torch.randn(1, 3, 448, 448, device=device, dtype=model.dtype)

    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens + IMG_END_TOKEN
    question = f"{image_tokens}\nDescribe this image."
    full_prompt, _ = _build_prompt_from_template(model, question, "A photo.")

    tokenized = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = input_ids.clone()
    image_flags = torch.ones(1, 1, device=device, dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_flags": image_flags,
    }


# ---------------------------------------------------------------------------
# Forward function for the pruner
# ---------------------------------------------------------------------------
def forward_fn(model, example_input):
    out = model(**example_input)
    return out.loss


# ---------------------------------------------------------------------------
# Identify Qwen3RMSNorm dynamically
# ---------------------------------------------------------------------------
def get_rms_norm_class(model):
    """Find the RMSNorm class used in the language model."""
    for module in model.language_model.modules():
        cls_name = type(module).__name__
        if "RMSNorm" in cls_name:
            return type(module)
    return None


def get_internvit_attn_class(model):
    """Find the InternAttention class used in the vision encoder."""
    for module in model.vision_model.encoder.layers.modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name and hasattr(module, "num_heads"):
            return type(module)
    return None


def get_qwen_attn_class(model):
    """Find the Qwen3Attention class used in the language model."""
    for module in model.language_model.modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name and hasattr(module, "num_heads"):
            return type(module)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir="pruned_checkpoint",
        setup_sublogger=True,
        sublogger_name=args.job_id,
    )

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    logger.log("Loading InternVL 3.5 1B...")
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=False,  # disable for tracing compatibility
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(args.device)
    model.eval()
    logger.log(f"Model loaded: {type(model).__name__}")

    # ------------------------------------------------------------------
    # 2. Prepare dummy input for FLOPs / graph tracing
    # ------------------------------------------------------------------
    example_input = prepare_internvl_inputs_simple(model, tokenizer, args.device)

    try:
        raw_flop = FlopCountAnalysis(model, example_input)
        logger.log(flop_count_table(raw_flop, max_depth=2, show_param_shapes=True))
        logger.log("Total Flops: " + str(raw_flop.total() / 1e9))
    except Exception as e:
        logger.log(f"FLOPs analysis failed (non-critical): {e}")
        raw_flop = None

    # ------------------------------------------------------------------
    # 3. Decouple InternViT fused QKV
    # ------------------------------------------------------------------
    logger.log("Decoupling InternViT fused QKV...")
    decouple_internvit_qkv(model, args.device)

    # ------------------------------------------------------------------
    # 4. Setup importance estimator
    # ------------------------------------------------------------------
    pruner_type = args.pruner_type.lower()
    if pruner_type == "taylor":
        imp = mask_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy,
            normalizer=args.imp_normalizer,
            taylor=args.taylor,
            model=model,
        )
    elif pruner_type == "taylor+knowledge":
        imp1 = mask_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy,
            normalizer=args.imp_normalizer,
            taylor=args.taylor,
        )
        imp2 = mask_pruner.KnowledgeImportance(group_reduction="mean", normalizer=None)
        imp = [imp1, imp2]
    else:
        raise NotImplementedError(f"Unknown pruner type: {pruner_type}")

    # ------------------------------------------------------------------
    # 5. Enable gradients
    # ------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad_(True)

    before_pruning_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    before_pruning_visual_parameters = sum(
        p.numel() for p in model.vision_model.parameters() if p.requires_grad
    )
    before_pruning_language_parameters = sum(
        p.numel() for p in model.language_model.parameters() if p.requires_grad
    )
    raw_param_counts = count_param(model)

    for name, param in model.named_parameters():
        param.global_name = name

    # ------------------------------------------------------------------
    # 6. Build calibration data loader
    # ------------------------------------------------------------------
    logger.log("Building calibration data loader...")
    data_loader = build_calibration_loader(args, tokenizer, model)

    if args.num_examples == 0:
        args.num_examples = len(data_loader)

    if args.pruning_ratio <= 0.2:
        args.channel_per_step = 100

    # ------------------------------------------------------------------
    # 7. Identify model-specific classes
    # ------------------------------------------------------------------
    RMSNormClass = get_rms_norm_class(model)
    InternVitAttnClass = get_internvit_attn_class(model)
    QwenAttnClass = get_qwen_attn_class(model)

    logger.log(f"RMSNorm class: {RMSNormClass}")
    logger.log(f"InternViT Attention class: {InternVitAttnClass}")
    logger.log(f"Qwen Attention class: {QwenAttnClass}")

    # ------------------------------------------------------------------
    # 8. Build pruner (block-granularity structured pruning)
    # ------------------------------------------------------------------
    if args.granularity != "block":
        raise NotImplementedError(f"Only block granularity is supported, got: {args.granularity}")

    # Build customized pruners map
    customized_pruners = {
        nn.Linear: linear_mask_pruner,
        nn.LayerNorm: layer_norm_mask_pruner,
    }
    if RMSNormClass is not None:
        customized_pruners[RMSNormClass] = rms_norm_mask_pruner

    # Build root instances: ViT attention + MLP, LLM attention + MLP
    root_instances = []

    # InternViT: 24 layers with decoupled q,k,v and mlp.fc1
    for i in range(NUM_VIT_LAYERS):
        layer = model.vision_model.encoder.layers[i]
        root_instances.append(layer.attn.q)
        root_instances.append(layer.mlp.fc1)

    # Qwen3: 28 layers with q_proj, gate_proj, up_proj
    for i in range(NUM_LLM_LAYERS):
        layer = model.language_model.model.layers[i]
        root_instances.append(layer.self_attn.q_proj)
        root_instances.append(layer.mlp.gate_proj)
        root_instances.append(layer.mlp.up_proj)

    # Build num_heads mapping for attention head pruning
    num_heads = {}
    for i in range(NUM_VIT_LAYERS):
        attn = model.vision_model.encoder.layers[i].attn
        num_heads[attn.q] = attn.num_heads

    for i in range(NUM_LLM_LAYERS):
        attn = model.language_model.model.layers[i].self_attn
        # For GQA: use num_attention_heads (Q heads) as the grouping for Q proj
        num_heads[attn.q_proj] = attn.config.num_attention_heads

    # q_norm and k_norm are per-head RMSNorms (shape=[head_dim]) that the
    # dependency graph incorrectly links to the full hidden_dim axis of q_proj/k_proj.
    # Pruning them with hidden_dim indices causes out-of-bounds errors.
    ignored_layers = []
    for i in range(NUM_LLM_LAYERS):
        attn = model.language_model.model.layers[i].self_attn
        if hasattr(attn, 'q_norm'):
            ignored_layers.append(attn.q_norm)
        if hasattr(attn, 'k_norm'):
            ignored_layers.append(attn.k_norm)

    kwargs = {
        "importance": imp,
        "global_pruning": args.global_pruning,
        "iterative_steps": args.iterative_steps,
        "ch_sparsity": args.pruning_ratio,
        "ignored_layers": ignored_layers,
        "max_pruning_ratio": args.max_pruning_ratio,
        "customized_pruners": customized_pruners,
        "root_module_types": [],
        "channel_per_step": args.channel_per_step,
        "prune_num_heads": True,
        "prune_head_dims": False,
        "multimodal": args.multimodal,
        "root_instances": root_instances,
        "num_heads": num_heads,
    }
    if args.imp_normalizer is None:
        kwargs["group_collect"] = "sum"

    # Build mask_operation mapping
    mask_operation = {
        linear_mask_pruner.prune_in_channels: linear_mask_pruner.operate_in_masks,
        linear_mask_pruner.prune_out_channels: linear_mask_pruner.operate_out_masks,
        layer_norm_mask_pruner.prune_in_channels: layer_norm_mask_pruner.operate_in_masks,
        layer_norm_mask_pruner.prune_out_channels: layer_norm_mask_pruner.operate_out_masks,
        param_mask_pruner.prune_in_channels: param_mask_pruner.operate_in_masks,
        param_mask_pruner.prune_out_channels: param_mask_pruner.operate_out_masks,
    }
    if RMSNormClass is not None:
        mask_operation[rms_norm_mask_pruner.prune_in_channels] = rms_norm_mask_pruner.operate_in_masks
        mask_operation[rms_norm_mask_pruner.prune_out_channels] = rms_norm_mask_pruner.operate_out_masks
    kwargs["mask_operation"] = mask_operation

    # Build trigger for GQA attention head pruning in Qwen3
    trigger_prune_module = {}
    for i in range(NUM_LLM_LAYERS):
        attn = model.language_model.model.layers[i].self_attn
        trigger_prune_module[attn.q_proj] = (
            attn,
            gqa_attention_head_mask_pruner.operate_out_masks,
        )
    kwargs["trigger"] = trigger_prune_module

    logger.log("Building MaskPruner...")
    pruner = tp.pruner.MaskPruner(
        model,
        example_input,
        forward_fn=forward_fn,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 9. Knowledge importance via entropy (optional)
    # Note: must run BEFORE select_loss monkey-patch to avoid OOM —
    # the entropy pass only needs hook activations, not the loss.
    # ------------------------------------------------------------------
    if args.entropy_importance:

        def new_process_imp_list(self, group, imp_list, ch_groups, remain_channels):
            _is_attn, qkv_layers = self._is_attn_group(group)
            group_size = len(imp_list[0]) // ch_groups
            if _is_attn and self.prune_num_heads:
                for i in range(len(imp_list)):
                    if imp_list[i] is None:
                        continue
                    if self.group_collect == "mean":
                        imp_list[i] = imp_list[i].view(ch_groups, -1).mean(1)
                    elif self.group_collect == "sum":
                        imp_list[i] = imp_list[i].view(ch_groups, -1).sum(1)
            if self.is_visual_part(group):
                imp = imp_list[0]
            else:
                imp = imp_list[0] * imp_list[1]

            if _is_attn and self.prune_num_heads:
                remain_channels = remain_channels.view(ch_groups, -1)[:, 0].view(-1)
            imp[remain_channels == 0] = float("inf")
            return imp, group_size

        pruner.process_imp_list = partial(new_process_imp_list, pruner)

        answer_start_idx = [0]

        def forward_projection_save_hook(module, input, output):
            weights = module.weight.t()
            x_flat = input[0].view(-1, input[0].size(-1))
            ans_start = answer_start_idx[0]
            if ans_start > 0 and ans_start < x_flat.shape[0]:
                x_flat = x_flat[ans_start:]
            weights_norm = F.normalize(weights, p=2, dim=0)
            x_norm = F.normalize(x_flat, p=2, dim=1)
            sim = torch.matmul(x_norm, weights_norm)
            module.out_dim_vals.append(sim.cpu())

        logger.log("Computing knowledge importance (entropy)...")
        with torch.no_grad():
            module_name_filters = [".q_proj", ".k_proj", ".v_proj", "gate_proj", "up_proj"]

            real_lm_head = model.language_model.lm_head
            model.language_model.lm_head = nn.Identity()

            assistant_token_ids = tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            assistant_len = len(assistant_token_ids)

            module_list, hook_list = [], []
            for name, module in model.language_model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                for subname in module_name_filters:
                    if subname in name:
                        module_list.append(module)
                        module.out_dim_vals = []
                        hook_list.append(
                            module.register_forward_hook(forward_projection_save_hook)
                        )
                        break

            num_batches = min(args.num_examples // args.calibration_bs, len(data_loader))
            logger.log(f"  {len(module_list)} modules hooked, {num_batches} batches...")
            loader_iter = iter(data_loader)
            for i in range(num_batches):
                batch_samples = next(loader_iter)
                inputs = prepare_internvl_inputs(
                    batch_samples, model, tokenizer, args.device
                )
                inputs.pop("labels", None)

                ids = inputs["input_ids"][0].tolist()
                ans_pos = 0
                for j in range(len(ids) - assistant_len + 1):
                    if ids[j:j + assistant_len] == assistant_token_ids:
                        ans_pos = j + assistant_len
                answer_start_idx[0] = ans_pos

                model(**inputs)
                if (i + 1) % 50 == 0 or i == num_batches - 1:
                    logger.log(f"    [{i + 1}/{num_batches}]")

            for module in module_list:
                out_dim_vals = torch.cat(module.out_dim_vals, dim=0).float().to(args.device)
                entropies = []
                for d in range(out_dim_vals.shape[-1]):
                    hist = torch.histc(out_dim_vals[:, d], bins=100, min=-1, max=1)
                    prob = hist / hist.sum(dim=0, keepdim=True)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-12), dim=0)
                    entropies.append(entropy)
                module.out_dim_entropy = torch.stack(entropies)
                del out_dim_vals, module.out_dim_vals
                torch.cuda.empty_cache()
            for hook_handle in hook_list:
                hook_handle.remove()
            torch.cuda.empty_cache()

            model.language_model.lm_head = real_lm_head
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 10. Selected loss (optional) — applied after entropy computation
    # ------------------------------------------------------------------
    if args.select_loss:
        original_forward = model.forward
        model.forward = partial(selected_causal_lm_forward, original_forward, model)

    # ------------------------------------------------------------------
    # 11. Compute gradient-based importance
    # ------------------------------------------------------------------
    model.zero_grad()
    logger.log("Computing gradient importance...")
    loader_iter = iter(data_loader)
    for i in range(min(args.num_examples // args.calibration_bs, len(data_loader))):
        batch_samples = next(loader_iter)
        inputs = prepare_internvl_inputs(batch_samples, model, tokenizer, args.device)
        out = model(**inputs)
        loss = out.loss
        if loss is not None:
            logger.log(f"Batch {i} loss = {loss.item():.4f}")
            loss.backward()

    # ------------------------------------------------------------------
    # 12. Run pruning
    # ------------------------------------------------------------------
    logger.log("Starting pruning...")
    if args.global_pruning:
        iter_steps, pruned_ratio = 0, 0.0
        prev_params = before_pruning_parameters
        while pruned_ratio < args.pruning_ratio:
            if pruned_ratio > 0.9 * args.pruning_ratio:
                pruner.channel_per_step = 100
            iter_steps += 1
            pruner.step()
            after_pruning_parameters = sum(
                torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks]))
                for p in model.parameters()
            )
            pruned_ratio = 1 - 1.0 * after_pruning_parameters / before_pruning_parameters
            after_pruning_visual_parameters = sum(
                torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks]))
                for p in model.vision_model.parameters()
            )
            visual_pruned_ratio = (
                1 - 1.0 * after_pruning_visual_parameters / before_pruning_visual_parameters
            )
            after_pruning_language_parameters = sum(
                torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks]))
                for p in model.language_model.parameters()
            )
            language_pruned_ratio = (
                1 - 1.0 * after_pruning_language_parameters / before_pruning_language_parameters
            )
            logger.log(
                f"Iter {iter_steps}: #params={after_pruning_parameters}, "
                f"ratio={pruned_ratio*100:.4f}%, "
                f"vision={visual_pruned_ratio*100:.2f}%, "
                f"language={language_pruned_ratio*100:.2f}%"
            )
            if after_pruning_parameters == prev_params:
                logger.log("No more channels to prune (all groups at max_pruning_ratio). Stopping early.")
                break
            prev_params = after_pruning_parameters
    else:
        pruner.step()
        after_pruning_parameters = sum(
            torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks]))
            for p in model.parameters()
        )
        pruned_ratio = 1 - 1.0 * after_pruning_parameters / before_pruning_parameters
        logger.log(
            f"After local pruning: #params={after_pruning_parameters}, ratio={pruned_ratio*100:.4f}%"
        )

    model.zero_grad()
    for name, module in model.named_parameters():
        if "weight" in name:
            module.grad = None

    # ------------------------------------------------------------------
    # 13. Compress weight matrices
    # ------------------------------------------------------------------
    logger.log("Compressing weight matrices...")
    pruner.compress_matrix()
    del pruner

    # ------------------------------------------------------------------
    # 14. Post-pruning fixup: update head counts and configs
    # ------------------------------------------------------------------
    logger.log("Post-pruning fixup: updating head counts...")

    # Update InternViT attention head counts
    for i in range(NUM_VIT_LAYERS):
        attn = model.vision_model.encoder.layers[i].attn
        if hasattr(attn, "q"):
            head_dim = attn.q.weight.shape[0] // max(1, attn.num_heads)
            if head_dim > 0:
                attn.num_heads = attn.q.weight.shape[0] // head_dim

    # Update Qwen3 attention head counts (GQA-aware)
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

    # ------------------------------------------------------------------
    # 15. Save masks, model, param counts
    # ------------------------------------------------------------------
    logger.log("Saving prune masks...")
    mask_dict = {}
    for name, param in model.named_parameters():
        if hasattr(param, "preserve_masks"):
            mask_dict[name] = [
                tensor.cpu().numpy().tolist() for tensor in param.preserve_masks
            ]
            del param.preserve_masks
    json.dump(
        mask_dict,
        open(os.path.join(logger.sub_dir, "prune_masks.json"), "w"),
    )

    after_pruning_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.log(
        f"#Param before: {before_pruning_parameters}, "
        f"#Param after: {after_pruning_parameters}, "
        f"#Ratio = {100.0*after_pruning_parameters/before_pruning_parameters:.4f}%"
    )

    # Undo selected loss patch before saving
    if args.select_loss:
        model.forward = type(model).forward.__get__(model, type(model))

    logger.log("Saving pruned model...")
    save_dir = os.path.join(logger.sub_dir, "pruned_model")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    shape_map = {name: list(p.shape) for name, p in model.named_parameters()}
    with open(os.path.join(save_dir, "pruned_shapes.json"), "w") as f:
        json.dump(shape_map, f)

    tokenizer.save_pretrained(save_dir)
    logger.log(f"Pruned model saved to {save_dir} (state_dict + shape metadata)")
    new_param_counts = count_param(model)
    save_dic = {"raw": raw_param_counts, "new": new_param_counts}
    json.dump(
        save_dic,
        open(os.path.join(logger.sub_dir, "param_counts.json"), "w"),
    )

    logger.log("Model after pruning:")
    logger.log(str(model))

    try:
        new_flop = FlopCountAnalysis(model, example_input)
        logger.log(flop_count_table(new_flop, max_depth=2, show_param_shapes=True))
        logger.log("Total Flops: " + str(new_flop.total() / 1e9))
        if raw_flop is not None:
            logger.log(
                f"FLOPs pruning ratio: {new_flop.total()*100.0/raw_flop.total():.4f}%"
            )
    except Exception as e:
        logger.log(f"Post-pruning FLOPs analysis failed (non-critical): {e}")

    logger.log("[FINISH] - Finish Pruning InternVL Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UKMP Pruning for InternVL")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID")
    parser.add_argument("--save_ckpt_log_name", type=str, default="internvl_prune",
                        help="Checkpoint and log save path name")
    parser.add_argument("--pruning_ratio", type=float, default=0.5, help="Pruning ratio")
    parser.add_argument("--max_pruning_ratio", type=float, default=0.9,
                        help="Max pruning ratio per group")
    parser.add_argument("--pruner_type", type=str, default="taylor",
                        help="Pruner type: taylor or taylor+knowledge")
    parser.add_argument("--granularity", type=str, default="block", help="Prune granularity")
    parser.add_argument("--imp_normalizer", type=str, default=None,
                        help="Importance normalizer")
    parser.add_argument("--iterative_steps", type=int, default=1,
                        help="Iteration steps for baseline pruning")
    parser.add_argument("--channel_per_step", type=int, default=1000,
                        help="Channels per step for iterative pruning")
    parser.add_argument("--grouping_strategy", type=str, default="sum",
                        help="Reduce method for grouping")
    parser.add_argument("--global_pruning", action="store_true",
                        help="Enable global pruning")
    parser.add_argument("--taylor", type=str, default="param_first",
                        help="Taylor variant: vectorize, param_second, param_first, param_mix")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Calibration dataset size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--select_loss", action="store_true",
                        help="Use selected loss (UKMI)")
    parser.add_argument("--entropy_importance", action="store_true",
                        help="Use entropy-based knowledge importance")
    parser.add_argument("--calibration_bs", type=int, default=1,
                        help="Calibration batch size")
    parser.add_argument("--multimodal", action="store_true",
                        help="Normalize importance for multimodal pruning")
    parser.add_argument("--calib_data_path", type=str, required=True,
                        help="Path to calibration data JSON (e.g. CC595K annotations)")
    parser.add_argument("--calib_image_root", type=str, required=True,
                        help="Root directory for calibration images")

    args = parser.parse_args()
    args.torch_version = float(".".join(torch.__version__.split(".")[:2]))

    main(args)
