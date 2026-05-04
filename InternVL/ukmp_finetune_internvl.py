"""
UKMP Recovery Fine-tuning for Pruned InternVL 3.5 1B
Loads a pruned InternVL checkpoint (state_dict + pruned_shapes.json) and
fine-tunes with LoRA + optional knowledge distillation from the full model.

Progressive distillation stages (--distill_mode):
  Stage 0 (τ1): β1 * Lmse(Ev_s, Ev_t)  — align vision hidden states
  Stage 1 (τ2): β1 * Lmse(Ev_s, Ev_t) + β2 * Lmse(El_s, El_t)
                — align vision + language hidden states
  Stage 2 (τ3): Ltask(ys, y) + Lkl(ps, pt)
                — CE task loss + KL divergence on logits

Weight Recalling (--wr_lora): PruneLora with parallel branches
  f(X) = (Wr + P1*Q1 + P2*Q2*Wp) * X  (paper Eq. 12)
"""

import argparse
import json
import os
import re
import random
import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from internvl_lib.common.logger import LoggerWithDepth
from internvl_lib.peft import (
    PruneLoraConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from internvl_lib.peft.tuners.prunelora.layer import Linear as PruneLoraLinear

NUM_VIT_LAYERS = 24
NUM_LLM_LAYERS = 28


def is_dist():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize DDP when launched via torchrun / torch.distributed.launch."""
    if "RANK" not in os.environ:
        return 0, torch.device("cuda")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device("cuda", local_rank)


# ---------------------------------------------------------------------------
# InternViT QKV decoupling (fused qkv -> separate q, k, v)
# Must be done before loading pruned weights since the pruning script saves
# with decoupled q,k,v keys.
# ---------------------------------------------------------------------------
def _decoupled_internvit_attn_forward(self, x):
    """Replacement forward for InternAttention after decoupling fused qkv."""
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


def decouple_internvit_qkv(model, device="cpu"):
    """Split fused qkv linear in each InternViT attention layer into q, k, v."""
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


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Image preprocessing (same as pruning / eval scripts)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


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
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_pil(image, input_size=448, max_num=1):
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_module_by_name(model, name):
    parts = name.split('.')
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


# ---------------------------------------------------------------------------
# Forward hooks for progressive distillation
# ---------------------------------------------------------------------------
def forward_save_feature_hook(module, input, output):
    """Save block output for MSE distillation. Handles tuple outputs (e.g. LLM layers)."""
    if isinstance(output, tuple):
        module.feature = output[0]
    else:
        module.feature = output


def register_hooks(model, model_label="model"):
    """Attach feature-saving hooks to all InternViT and Qwen3 layers."""
    hook_list = []

    for name, module in model.named_modules():
        m_vit = re.search(r'vision_model\.encoder\.layers\.(\d+)$', name)
        if m_vit:
            hook_list.append(module.register_forward_hook(forward_save_feature_hook))
            continue

        m_llm = re.search(r'language_model\.model\.layers\.(\d+)$', name)
        if m_llm:
            hook_list.append(module.register_forward_hook(forward_save_feature_hook))

    return hook_list


# ---------------------------------------------------------------------------
# Progressive distillation loss functions (paper Eq. 13-14)
# ---------------------------------------------------------------------------
mse_criterion = nn.MSELoss()


def _mse_feat(student_feat, teacher_feat, feature_norm):
    """Compute MSE between student and teacher features.
    When feature_norm=True, applies per-layer L2 normalization (paper Eq. 14):
      || Es,n/||Es,n||_2 - Et,n/||Et,n||_2 ||^2
    """
    if student_feat.shape != teacher_feat.shape:
        teacher_feat = teacher_feat[..., :student_feat.shape[-1]]
    if feature_norm:
        student_feat = F.normalize(student_feat.float(), p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat.float(), p=2, dim=-1)
    return mse_criterion(student_feat, teacher_feat)


def _enable_vit_grads(inputs):
    """Ensure pixel_values requires grad so gradient checkpointing in InternViT works."""
    if inputs["pixel_values"].dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].detach().requires_grad_(True)
    return inputs


def _get_student_base(student_model):
    m = student_model.module if isinstance(student_model, DDP) else student_model
    return m.model if type(m).__name__ == "PeftModel" else m


def _run_both(student_model, teacher_model, inputs):
    """Run student (with grads) and teacher (no grads) forward passes."""
    inputs = _enable_vit_grads(inputs)
    student_model(**inputs)
    with torch.no_grad():
        teacher_model(**inputs)


def _vision_mse(student_base, teacher_model, feature_norm, n_layers=NUM_VIT_LAYERS):
    """Eq. 14 for vision: sum of per-layer normalized MSE over ViT layers."""
    loss = 0.0
    for i in range(n_layers):
        s_layer = student_base.vision_model.encoder.layers[i]
        t_layer = teacher_model.vision_model.encoder.layers[i]
        if hasattr(s_layer, 'feature') and hasattr(t_layer, 'feature'):
            loss += _mse_feat(s_layer.feature, t_layer.feature, feature_norm)
    return loss


def _llm_mse(student_base, teacher_model, feature_norm, n_layers=NUM_LLM_LAYERS):
    """Eq. 14 for language: sum of per-layer normalized MSE over LLM layers."""
    loss = 0.0
    for i in range(n_layers):
        s_layer = student_base.language_model.model.layers[i]
        t_layer = teacher_model.language_model.model.layers[i]
        if hasattr(s_layer, 'feature') and hasattr(t_layer, 'feature'):
            loss += _mse_feat(s_layer.feature, t_layer.feature, feature_norm)
    return loss


def train_step_tau1(student_model, teacher_model, inputs,
                    feature_norm=True, beta1=1.0, **_kw):
    """Phase τ1 (Eq.13): β1 * Lmse(Ev_s, Ev_t) — align vision hidden states."""
    _run_both(student_model, teacher_model, inputs)
    student_base = _get_student_base(student_model)
    return beta1 * _vision_mse(student_base, teacher_model, feature_norm)


def train_step_tau2(student_model, teacher_model, inputs,
                    feature_norm=True, beta1=1.0, beta2=1.0, **_kw):
    """Phase τ2 (Eq.13): β1*Lmse(Ev_s,Ev_t) + β2*Lmse(El_s,El_t)
    — align vision + language hidden states."""
    _run_both(student_model, teacher_model, inputs)
    student_base = _get_student_base(student_model)
    v_loss = _vision_mse(student_base, teacher_model, feature_norm)
    l_loss = _llm_mse(student_base, teacher_model, feature_norm)
    return beta1 * v_loss + beta2 * l_loss


_tau3_log_counter = 0

def train_step_tau3(student_model, teacher_model, inputs,
                    kd_temperature=1.0, **_kw):
    """Phase τ3 (Eq.13): Ltask(ys, y) + Lkl(ps, pt)
    — task CE loss + KL divergence on logits.

    KL is computed ONLY over label-token positions (where labels != -100)
    so that prompt/padding tokens don't inflate the divergence.
    If a sample has zero label tokens, return a zero loss (no signal).
    """
    global _tau3_log_counter

    labels = inputs.get("labels", None)
    outputs = student_model(**inputs)
    ce_loss = outputs.loss

    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)

    student_logits = outputs.logits
    teacher_logits = teacher_outputs.logits
    min_len = min(student_logits.shape[1], teacher_logits.shape[1])

    # Build mask: True for positions where the model is actually supervised
    if labels is not None:
        label_mask = (labels[:, :min_len] != -100)   # (B, min_len)
        n_label_tokens = label_mask.sum().item()
    else:
        label_mask = None
        n_label_tokens = min_len

    # If no label tokens at all, CE is NaN — return a zero loss with grad
    if n_label_tokens == 0:
        _tau3_log_counter += 1
        if _tau3_log_counter <= 10 or _tau3_log_counter % 50 == 0:
            print(f"[tau3-diag step={_tau3_log_counter}] SKIP: n_label_tokens=0")
        return (student_logits.sum() * 0.0).requires_grad_(True)

    T = kd_temperature
    s_logits_f = student_logits[:, :min_len].float() / T
    t_logits_f = teacher_logits[:, :min_len].float() / T

    s_log = F.log_softmax(s_logits_f, dim=-1)        # (B, min_len, V)
    t_prob = F.softmax(t_logits_f, dim=-1)

    # Per-token KL: sum over vocab dimension -> (B, min_len)
    per_token_kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)

    # Mask to label positions only and average over them
    if label_mask is not None:
        per_token_kl = per_token_kl * label_mask.float()
    kl_loss = per_token_kl.sum() / max(n_label_tokens, 1) * (T ** 2)

    _tau3_log_counter += 1
    if _tau3_log_counter <= 10 or _tau3_log_counter % 50 == 0:
        with torch.no_grad():
            s_abs = student_logits[:, :min_len].float().abs()
            t_abs = teacher_logits[:, :min_len].float().abs()
            print(
                f"[tau3-diag step={_tau3_log_counter}] "
                f"seq_len={min_len}, "
                f"n_label_tokens={n_label_tokens}, "
                f"ce_loss={ce_loss.item():.4f}, "
                f"kl_loss={kl_loss.item():.4f}, "
                f"total={ce_loss.item() + kl_loss.item():.4f}, "
                f"student_logit max={s_abs.max().item():.1f} mean={s_abs.mean().item():.2f}, "
                f"teacher_logit max={t_abs.max().item():.1f} mean={t_abs.mean().item():.2f}"
            )

    return ce_loss + kl_loss.to(ce_loss.dtype)


def _compute_data_slices(total_samples):
    """Split dataset into 3 phases following the paper:
      τ1 (visual MSE):         ~6.7% of data  (40K/595K in the paper)
      τ2 (visual + LLM MSE):  ~47.1% of data  (280K/595K in the paper)
      τ3 (CE + KL):           ~46.2% of data  (275K/595K in the paper)
    """
    t1_end = round(total_samples * 40 / 595)
    t2_end = t1_end + round(total_samples * 280 / 595)
    return [
        (0, t1_end),
        (t1_end, t2_end),
        (t2_end, total_samples),
    ]


# ---------------------------------------------------------------------------
# Model loading helper (state_dict + pruned_shapes.json)
# ---------------------------------------------------------------------------
def load_pruned_model(model_name, pruned_ckpt_dir, device, dtype=torch.bfloat16):
    """Load a pruned model from state_dict + pruned_shapes.json.

    The pruning script decouples InternViT's fused qkv into separate q,k,v
    before pruning, so we must do the same here before loading the state dict.
    """
    sd_path = os.path.join(pruned_ckpt_dir, "model.pt")
    shapes_path = os.path.join(pruned_ckpt_dir, "pruned_shapes.json")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=True,
    )

    # Decouple fused qkv -> q, k, v so that pruned state dict keys match
    decouple_internvit_qkv(model, device="cpu")

    if os.path.exists(shapes_path):
        with open(shapes_path) as f:
            shape_map = {k: tuple(v) for k, v in json.load(f).items()}
    else:
        state_dict_tmp = torch.load(sd_path, map_location="cpu", weights_only=True)
        shape_map = {k: tuple(v.shape) for k, v in state_dict_tmp.items()}
        del state_dict_tmp

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

    # Post-load fixup: update attention head counts to match pruned dimensions.
    # Use original head_dim (before pruning) since prune_num_heads=True keeps
    # head_dim constant and only removes entire heads.
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

    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _load_data_file(data_path):
    """Load a JSON array or a JSONL file into a list of dicts."""
    if data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        with open(data_path, "r") as f:
            return json.load(f)


def load_sft_datasets(data_config_path):
    """Load multiple SFT datasets from a config JSON file.

    Config format (list of dataset specs):
    [
      {"annotation": "path/to/file.jsonl", "image_root": "path/to/images/", "max_samples": 10000},
      {"annotation": "path/to/another.jsonl", "image_root": "path/to/other_images/"},
      ...
    ]

    Returns a flat list of (item_dict, image_root) tuples.
    """
    with open(data_config_path, "r") as f:
        config = json.load(f)

    all_items = []
    for ds in config:
        annotation = ds["annotation"]
        image_root = ds["image_root"]
        max_samples = ds.get("max_samples", None)
        raw = _load_data_file(annotation)
        if max_samples is not None:
            random.shuffle(raw)
            raw = raw[:max_samples]
        for item in raw:
            all_items.append((item, image_root))
    random.shuffle(all_items)
    return all_items


class FinetuneDataset(Dataset):
    def __init__(self, data_path, image_root, max_samples=None,
                 image_size=448, max_num_tiles=1, data_config=None):
        if data_config is not None:
            paired = load_sft_datasets(data_config)
            if max_samples is not None:
                paired = paired[:max_samples]
            self.data = [item for item, _ in paired]
            self.image_roots = [root for _, root in paired]
            self.multi_root = True
        else:
            raw = _load_data_file(data_path)
            if max_samples is not None:
                random.shuffle(raw)
                raw = raw[:max_samples]
            self.data = raw
            self.image_roots = None
            self.multi_root = False
        self.image_root = image_root
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item.get("image", "")
        if not os.path.isabs(img_path):
            root = self.image_roots[idx] if self.multi_root else self.image_root
            img_path = os.path.join(root, img_path)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = load_image_pil(image, input_size=self.image_size,
                                          max_num=self.max_num_tiles)
        except Exception:
            pixel_values = torch.zeros(1, 3, self.image_size, self.image_size)

        if "conversations" in item:
            convs = item["conversations"]
            question = ""
            answer = ""
            for turn in convs:
                role = turn.get("from", turn.get("role", ""))
                val = turn.get("value", turn.get("content", ""))
                val = val.replace("<image>", "").strip()
                if role in ("human", "user"):
                    if not question:
                        question = val
                elif role in ("gpt", "assistant"):
                    if not answer:
                        answer = val
            if not question:
                question = "Describe this image."
            if not answer:
                answer = "This is an image."
        else:
            question = "Describe this image."
            answer = item.get("caption", item.get("text", "This is an image."))

        return {
            "pixel_values": pixel_values,
            "question": question,
            "answer": answer,
        }


def _build_prompt_from_template(model, question, answer=None):
    """Build a prompt string using the model's own conversation template.

    Mirrors how model.chat() constructs prompts so that train/eval formatting
    is always consistent with the HuggingFace-shipped template (system message,
    role tokens, separators, etc.).

    Returns (full_prompt, assistant_prompt) where assistant_prompt is the
    assistant turn only (used to locate where labels should start).
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


def build_training_batch(samples, model, tokenizer, device, num_image_token=256):
    """Build a training batch with proper chat-template formatting and labels."""
    pixel_values_list = []
    input_ids_list = []
    labels_list = []

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    base_model = model.model if type(model).__name__ == "PeftModel" else model
    nit = getattr(base_model, "num_image_token", num_image_token)

    for sample in samples:
        pv = sample["pixel_values"]
        pixel_values_list.append(pv)
        num_patches = pv.shape[0]

        n_image_tok = nit * num_patches
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * n_image_tok + IMG_END_TOKEN
        question_with_image = f"{image_tokens}\n{sample['question']}"

        full_prompt, assistant_prompt = _build_prompt_from_template(
            base_model, question_with_image, sample["answer"]
        )

        full_ids = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        assistant_ids = tokenizer(assistant_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

        prefix_len = full_ids.shape[0] - assistant_ids.shape[0]
        labels = full_ids.clone()
        labels[:prefix_len] = -100

        max_len = n_image_tok + 256
        input_ids = full_ids[:max_len]
        labels = labels[:max_len]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    max_seq = max(ids.shape[0] for ids in input_ids_list)
    padded_input_ids = []
    padded_labels = []
    for ids, lbl in zip(input_ids_list, labels_list):
        pad_len = max_seq - ids.shape[0]
        padded_input_ids.append(F.pad(ids, (0, pad_len), value=tokenizer.pad_token_id))
        padded_labels.append(F.pad(lbl, (0, pad_len), value=-100))

    pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
    input_ids = torch.stack(padded_input_ids).to(device)
    labels = torch.stack(padded_labels).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    image_flags = torch.ones(pixel_values.shape[0], 1, device=device, dtype=torch.long)

    dtype = next(model.parameters()).dtype
    return {
        "pixel_values": pixel_values.to(dtype),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_flags": image_flags,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    local_rank, device = setup_distributed()
    args.device = str(device)
    setup_seeds(args.seed + get_rank())

    if is_main_process():
        logger = LoggerWithDepth(
            env_name="{}".format(args.save_ckpt_log_name),
            config=args.__dict__,
            root_dir="tuned_checkpoint",
            setup_sublogger=True,
            sublogger_name=args.job_id,
        )
    else:
        class _SilentLogger:
            """No-op logger for non-rank-0 processes."""
            def __getattr__(self, _):
                return lambda *a, **kw: None
            sub_dir = "/tmp"
        logger = _SilentLogger()

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Load full (teacher) model if distillation is enabled
    # ------------------------------------------------------------------
    full_model = None
    hook_list = []
    if args.distill_mode:
        logger.log("Loading full InternVL model as teacher...")
        full_model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )
        decouple_internvit_qkv(full_model, device="cpu")
        full_model.to(device)
        full_model.eval()
        for param in full_model.parameters():
            param.requires_grad = False

        logger.log("Registering feature hooks on teacher...")
        hook_list += register_hooks(full_model, "teacher")

        for name, module in full_model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

    # ------------------------------------------------------------------
    # 3. Load pruned (student) model
    # ------------------------------------------------------------------
    logger.log(f"Loading pruned model from: {args.pruned_ckpt}")
    model = load_pruned_model(args.model_name, args.pruned_ckpt, device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    if args.distill_mode:
        if hasattr(model.vision_model, 'gradient_checkpointing_disable'):
            model.vision_model.gradient_checkpointing_disable()
        if hasattr(model.vision_model, 'encoder') and hasattr(model.vision_model.encoder, 'gradient_checkpointing'):
            model.vision_model.encoder.gradient_checkpointing = False
        logger.log("Disabled gradient checkpointing on student ViT for MSE distillation")

    # ------------------------------------------------------------------
    # 4. Load pruned mask and apply PruneLora (Weight Recalling) or plain LoRA
    # ------------------------------------------------------------------
    pruned_mask = None
    mask_data = None
    if args.pruned_mask is not None and os.path.exists(args.pruned_mask):
        logger.log(f"Loading pruned mask from: {args.pruned_mask}")
        mask_data = json.load(open(args.pruned_mask, 'r'))
        pruned_mask = {}
        for k in mask_data.keys():
            pruned_mask[k] = [torch.tensor(mask, dtype=torch.bool) for mask in mask_data[k]]

    lora_targets = [t.strip() for t in args.lora_target_modules.split(",")]

    if args.wr_lora and pruned_mask is not None:
        logger.log("Setting up PruneLora (Weight Recalling) structure...")

        pruned_linear_features = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name + '.weight'
                if key in mask_data:
                    pruned_linear_features[name] = [
                        (len(l) - int(sum(l))) for l in mask_data[key]
                    ]
                elif any(name.endswith('.' + t) for t in lora_targets):
                    pruned_linear_features[name] = [0, 0]

        config = PruneLoraConfig(
            r=args.lora_r,
            pruned_r=args.lora_pruned_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=args.lora_dropout,
            bias="none",
            init_lora_weights=True,
            pruned_features=pruned_linear_features,
        )
        if not hasattr(model, 'config') or model.config is None:
            from transformers import PretrainedConfig
            model.config = PretrainedConfig()
        model = get_peft_model(model, config).to(device)

        if full_model is None:
            logger.log("WARNING: --wr_lora requires a teacher model (--distill_mode) "
                        "to initialize weight recalling branches. Skipping WR init.")
        for _name, module in model.named_modules():
            if full_model is None:
                break
            if isinstance(module, PruneLoraLinear):
                name = _name.replace('base_model.model.', '').replace('.base_layer', '')
                full_module = get_module_by_name(full_model, name)
                if full_module is None:
                    continue
                key = name + '.weight'
                if key not in pruned_mask:
                    continue
                masks = pruned_mask[key]
                if module.input_base_layer is not None:
                    module.input_base_layer.weight.data = (
                        full_module.weight.data[masks[0]][:, ~masks[1]].clone()
                    )
                    module.input_base_layer.weight.requires_grad = False
                if module.output_base_layer is not None:
                    module.output_base_layer.weight.data = (
                        full_module.weight.data[~masks[0]][:, masks[1]].clone()
                    )
                    module.output_base_layer.weight.requires_grad = False
    else:
        logger.log("Setting up standard LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=args.lora_dropout,
            bias="none",
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
        model.to(device)

    if is_main_process():
        model.print_trainable_parameters()

    if args.distill_mode:
        logger.log("Registering feature hooks on student...")
        hook_list += register_hooks(
            model.model if type(model).__name__ == "PeftModel" else model,
            "student",
        )

    # ------------------------------------------------------------------
    # 5. Set img_context_token_id on both student and teacher
    # ------------------------------------------------------------------
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    student_base = model.model if type(model).__name__ == "PeftModel" else model
    student_base.img_context_token_id = img_context_token_id
    if full_model is not None:
        full_model.img_context_token_id = img_context_token_id

    # Save reference before DDP wrapping (needed for merge_and_unload later)
    old_state_dict = model.state_dict

    # ------------------------------------------------------------------
    # 5b. Wrap student model in DDP for multi-GPU training
    # ------------------------------------------------------------------
    if is_dist():
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)
        logger.log(f"Wrapped model in DDP across {get_world_size()} GPUs")

    # ------------------------------------------------------------------
    # 6. Build dataset
    # ------------------------------------------------------------------
    if not args.data_config and not args.data_path:
        raise ValueError("Either --data_config or --data_path must be provided.")

    full_dataset = FinetuneDataset(
        data_path=args.data_path,
        image_root=args.image_root or "",
        max_samples=args.max_samples,
        max_num_tiles=args.max_num_tiles,
        data_config=args.data_config,
    )
    logger.log(f"Image tiling: max_num_tiles={args.max_num_tiles}")
    total_num = len(full_dataset)

    # ------------------------------------------------------------------
    # 7. Define progressive distillation stages
    # ------------------------------------------------------------------
    distill_kwargs = dict(
        feature_norm=args.feature_norm,
        beta1=args.beta1,
        beta2=args.beta2,
        kd_temperature=args.kd_temperature,
    )

    if args.distill_mode and full_model is not None:
        stage_names = ["tau1_visual_mse", "tau2_visual+llm_mse", "tau3_ce+kl"]
        stage_fns = [
            partial(train_step_tau1, **distill_kwargs),
            partial(train_step_tau2, **distill_kwargs),
            partial(train_step_tau3, **distill_kwargs),
        ]
        data_slices = _compute_data_slices(total_num)
    else:
        stage_names = ["finetune"]
        stage_fns = [None]
        data_slices = [(0, total_num)]

    logger.log(f"Total dataset: {total_num} samples")
    logger.log(f"Number of stages: {len(stage_names)}")
    for i, (name, sl) in enumerate(zip(stage_names, data_slices)):
        logger.log(f"  Stage {i} ({name}): samples [{sl[0]}:{sl[1]}] ({sl[1]-sl[0]} samples)")

    # ------------------------------------------------------------------
    # 8. Run each stage
    # ------------------------------------------------------------------
    for stage_idx, (stage_name, stage_fn) in enumerate(zip(stage_names, stage_fns)):
        sl_start, sl_end = data_slices[stage_idx]
        if sl_start >= sl_end:
            logger.log(f"Stage {stage_idx} ({stage_name}): no data, skipping.")
            continue

        stage_dataset = deepcopy(full_dataset)
        stage_dataset.data = full_dataset.data[sl_start:sl_end]
        if getattr(full_dataset, 'multi_root', False):
            stage_dataset.image_roots = full_dataset.image_roots[sl_start:sl_end]

        sampler = DistributedSampler(stage_dataset, shuffle=True) if is_dist() else None
        dataloader = DataLoader(
            stage_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=2,
            collate_fn=lambda batch: batch,
            drop_last=True,
            pin_memory=True,
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        total_steps = max(1, len(dataloader) * args.num_epochs // args.gradient_accumulation_steps)

        def cosine_lr(step, _total=total_steps):
            if step >= _total:
                return 0.01
            return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * step / _total))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

        logger.log(f"=== Stage {stage_idx}: {stage_name} | "
                    f"{len(stage_dataset)} samples, {args.num_epochs} epoch(s), "
                    f"{get_world_size()} GPU(s) ===")

        global_step = 0
        optimizer.zero_grad()
        for epoch in range(args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            # τ1/τ2 compute loss from hooked features, not from the model's
            # forward output. DDP's reducer can't trace that gradient path,
            # so we use no_sync() and manually allreduce gradients instead.
            uses_hook_loss = stage_fn is not None and stage_name != "tau3_ce+kl"

            for batch_idx, batch_samples in enumerate(dataloader):
                base_model = student_base
                inputs = build_training_batch(batch_samples, base_model, tokenizer, device)

                if uses_hook_loss and is_dist():
                    with model.no_sync():
                        loss = stage_fn(model, full_model, inputs)
                elif stage_fn is not None:
                    loss = stage_fn(model, full_model, inputs)
                else:
                    outputs = model(**inputs)
                    loss = outputs.loss

                loss_val = loss.item()
                bad_loss = math.isnan(loss_val) or math.isinf(loss_val)

                if bad_loss:
                    # Zero the loss but keep the computation graph connected
                    # so DDP's allreduce still fires for all ranks.
                    loss = loss * 0.0
                    loss_val = 0.0

                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if not bad_loss:
                    epoch_loss += loss_val

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Check for NaN in gradients and skip the step if found
                    has_nan = any(
                        torch.isnan(p.grad).any() for p in trainable_params
                        if p.grad is not None
                    )
                    if has_nan:
                        optimizer.zero_grad()
                        global_step += 1
                        continue

                    if uses_hook_loss and is_dist():
                        for p in trainable_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (batch_idx + 1) % args.log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.log(
                        f"[{stage_name}] Epoch {epoch+1}/{args.num_epochs}, "
                        f"Step {batch_idx+1}/{len(dataloader)}, "
                        f"Loss: {avg_loss:.4f}, LR: {lr_now:.2e}"
                    )

            avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
            logger.log(f"[{stage_name}] Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

        del optimizer, scheduler, dataloader, stage_dataset
        if sampler is not None:
            del sampler
        torch.cuda.empty_cache()

    for h in hook_list:
        h.remove()

    # ------------------------------------------------------------------
    # 9. Merge LoRA and save (rank 0 only)
    # ------------------------------------------------------------------
    if is_dist():
        dist.barrier()

    # Unwrap DDP before merge
    if is_dist():
        model = model.module

    logger.log("Merging adapter weights into base model...")
    model.state_dict = old_state_dict
    model = model.merge_and_unload()
    model.eval()

    if is_main_process():
        save_dir = os.path.join(logger.sub_dir, "finetuned_model")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        shape_map = {name: list(p.shape) for name, p in model.named_parameters()}
        with open(os.path.join(save_dir, "pruned_shapes.json"), "w") as f:
            json.dump(shape_map, f)
        tokenizer.save_pretrained(save_dir)

        logger.log(f"Fine-tuned model saved to {save_dir}")
        logger.log("[FINISH] - Finish Fine-tuning Pruned InternVL Model")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UKMP Fine-tune Pruned InternVL")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B")
    parser.add_argument("--pruned_ckpt", type=str, required=True,
                        help="Path to pruned model directory (with model.pt + pruned_shapes.json)")
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--save_ckpt_log_name", type=str, default="internvl_finetune")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data (JSON array or JSONL). "
                             "Not needed when --data_config is used.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Root dir for images. Not needed when --data_config is used.")
    parser.add_argument("--data_config", type=str, default=None,
                        help="Path to a JSON config listing multiple SFT datasets. "
                             "Each entry: {annotation, image_root, max_samples(optional)}. "
                             "When set, --data_path/--image_root are ignored.")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,proj,fc1,fc2,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_num_tiles", type=int, default=6,
                        help="Max image tiles for dynamic preprocessing. "
                             "Higher = better OCR but more VRAM. (1=thumbnail only, 12=full)")
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--distill_mode", action="store_true",
                        help="Progressive distillation (paper Eq.13): "
                             "τ1 visual MSE, τ2 visual+LLM MSE, τ3 CE+KL")
    parser.add_argument("--kd_temperature", type=float, default=1.0,
                        help="Temperature T for KL divergence in phase τ3")
    parser.add_argument("--beta1", type=float, default=1.0,
                        help="Weight β1 for vision MSE loss (Eq.13)")
    parser.add_argument("--beta2", type=float, default=1.0,
                        help="Weight β2 for language MSE loss (Eq.13)")
    parser.add_argument("--feature_norm", action="store_true",
                        help="Per-layer L2-normalize features before MSE (paper Eq.14)")

    parser.add_argument("--wr_lora", action="store_true",
                        help="Use PruneLora weight recalling structure (requires --pruned_mask)")
    parser.add_argument("--pruned_mask", type=str, default=None,
                        help="Path to prune_masks.json from pruning step")
    parser.add_argument("--lora_pruned_r", type=int, default=8,
                        help="Rank for the weight recalling branch")

    args = parser.parse_args()
    main(args)
