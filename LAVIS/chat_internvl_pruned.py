"""
Interactive chat with a pruned (and optionally finetuned) InternVL model.

Usage:
  # With a finetuned checkpoint (directory containing model.pt + pruned_shapes.json):
  python chat_internvl_pruned.py \
    --pruned_ckpt tuned_checkpoint/internvl_finetune/recovery-v8-kd/finetuned_model/

  # With just a pruned checkpoint:
  python chat_internvl_pruned.py \
    --pruned_ckpt pruned_checkpoint/ukmp_prune_internvl/test-run-v8/pruned_model/

  # With the full unpruned model (for comparison):
  python chat_internvl_pruned.py

Then at the prompt, type an image path and a question:
  > /path/to/image.jpg  What is in this image?
  > /path/to/photo.png   How many people are there?
  > quit
"""

import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

NUM_VIT_LAYERS = 24
NUM_LLM_LAYERS = 28

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# InternViT QKV decoupling (needed for pruned checkpoints)
# ---------------------------------------------------------------------------
def _decoupled_internvit_attn_forward(self, x):
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    scale = q.shape[-1] ** -0.5
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def decouple_internvit_qkv(model, device="cpu"):
    for _, module in model.vision_model.encoder.layers.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            qkv = module.qkv
            in_feat = qkv.in_features
            out_feat = qkv.out_features // 3
            has_bias = qkv.bias is not None
            q = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)
            k = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)
            v = nn.Linear(in_feat, out_feat, bias=has_bias, device=device)
            q.weight.data = qkv.weight.data[:out_feat, :]
            k.weight.data = qkv.weight.data[out_feat:out_feat * 2, :]
            v.weight.data = qkv.weight.data[out_feat * 2:, :]
            if has_bias:
                q.bias.data = qkv.bias.data[:out_feat]
                k.bias.data = qkv.bias.data[out_feat:out_feat * 2]
                v.bias.data = qkv.bias.data[out_feat * 2:]
            module.q = q
            module.k = k
            module.v = v
            del module.qkv
            module.forward = partial(_decoupled_internvit_attn_forward, module)


# ---------------------------------------------------------------------------
# Image preprocessing (matches InternVL's dynamic resolution)
# ---------------------------------------------------------------------------
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
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name, pruned_ckpt, device):
    if pruned_ckpt is not None:
        if os.path.isdir(pruned_ckpt):
            sd_path = os.path.join(pruned_ckpt, "model.pt")
            shapes_path = os.path.join(pruned_ckpt, "pruned_shapes.json")
        else:
            sd_path = pruned_ckpt
            shapes_path = None

        print(f"Loading base architecture: {model_name}")
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=False, low_cpu_mem_usage=True,
        )
        decouple_internvit_qkv(model, device="cpu")

        if shapes_path and os.path.exists(shapes_path):
            with open(shapes_path) as f:
                shape_map = {k: tuple(v) for k, v in json.load(f).items()}
        else:
            sd_tmp = torch.load(sd_path, map_location="cpu", weights_only=True)
            shape_map = {k: tuple(v.shape) for k, v in sd_tmp.items()}
            del sd_tmp

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

        print("Pruned model loaded.")
    else:
        print(f"Loading full model: {model_name}")
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, use_flash_attn=False, low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model, tokenizer


def parse_input(user_input):
    """Split input into image_path and question.

    Supports:
      /path/to/image.jpg  What is this?
      "/path/with spaces/image.jpg"  Describe this.
    """
    user_input = user_input.strip()
    if user_input.startswith('"'):
        end_quote = user_input.index('"', 1)
        image_path = user_input[1:end_quote]
        question = user_input[end_quote + 1:].strip()
    else:
        parts = user_input.split(None, 1)
        image_path = parts[0]
        question = parts[1] if len(parts) > 1 else "Describe this image."
    return os.path.expanduser(image_path), question


def main():
    parser = argparse.ArgumentParser(description="Chat with pruned InternVL")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B")
    parser.add_argument("--pruned_ckpt", type=str, default=None,
                        help="Path to pruned/finetuned model dir (with model.pt + pruned_shapes.json)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_num_tiles", type=int, default=6,
                        help="Max image tiles for dynamic preprocessing (match training setting)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model(args.model_name, args.pruned_ckpt, device)
    dtype = next(model.parameters()).dtype

    print("\n" + "=" * 60)
    print("InternVL Interactive Chat")
    print("=" * 60)
    print("Enter: <image_path>  <question>")
    print('  e.g.  photo.jpg  What objects are in this image?')
    print("Commands: quit/exit, clear (reset history)")
    print("=" * 60 + "\n")

    history = []
    current_image = None
    current_pixels = None

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history = []
            current_image = None
            current_pixels = None
            print("-- History cleared --\n")
            continue

        # If input looks like it starts with a file path, parse image + question
        first_token = user_input.split('"')[1] if user_input.startswith('"') else user_input.split()[0]
        first_token = os.path.expanduser(first_token)
        if os.path.exists(first_token) or any(first_token.lower().endswith(ext)
                                                for ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff')):
            image_path, question = parse_input(user_input)

            if not os.path.exists(image_path):
                print(f"  [Error] File not found: {image_path}\n")
                continue

            try:
                current_pixels = load_image(image_path, input_size=448, max_num=args.max_num_tiles).to(device).to(dtype)
                current_image = image_path
                history = []
                print(f"  [Loaded image: {image_path}]")
            except Exception as e:
                print(f"  [Error loading image: {e}]\n")
                continue
        else:
            question = user_input

        if current_pixels is None:
            print("  [No image loaded. Provide an image path first.]\n")
            continue

        prompt = f"<image>\n{question}"
        generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

        with torch.no_grad():
            response, history = model.chat(
                tokenizer, current_pixels, prompt, generation_config,
                history=history, return_history=True,
            )

        print(f"\n  {response}\n")


if __name__ == "__main__":
    main()
