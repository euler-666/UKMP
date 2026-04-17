"""
UKMP Recovery Fine-tuning for Pruned InternVL 3.5 1B
Loads a pruned InternVL checkpoint (state_dict + pruned_shapes.json) and
fine-tunes with LoRA + optional knowledge distillation from the full model.
"""

import argparse
import json
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from lavis.common.logger import LoggerWithDepth
from lavis.peft import LoraConfig, get_peft_model


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
# Model loading helper (state_dict + pruned_shapes.json)
# ---------------------------------------------------------------------------
def load_pruned_model(model_name, pruned_ckpt_dir, device, dtype=torch.bfloat16):
    """Load a pruned model from state_dict + pruned_shapes.json."""
    sd_path = os.path.join(pruned_ckpt_dir, "model.pt")
    shapes_path = os.path.join(pruned_ckpt_dir, "pruned_shapes.json")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=True,
    )

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
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FinetuneDataset(Dataset):
    def __init__(self, data_path, image_root, max_samples=None,
                 image_size=448, max_num_tiles=1):
        with open(data_path, "r") as f:
            raw = json.load(f)
        if max_samples is not None:
            random.shuffle(raw)
            raw = raw[:max_samples]
        self.data = raw
        self.image_root = image_root
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item.get("image", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

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
                    question = val
                elif role in ("gpt", "assistant"):
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


def build_training_batch(samples, model, tokenizer, device, num_image_token=256):
    """Build a training batch with proper chat-template formatting and labels."""
    pixel_values_list = []
    input_ids_list = []
    labels_list = []

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    nit = getattr(model, "num_image_token",
                  getattr(getattr(model, "model", None), "num_image_token", num_image_token))

    for sample in samples:
        pv = sample["pixel_values"]
        pixel_values_list.append(pv)
        num_patches = pv.shape[0]

        n_image_tok = nit * num_patches
        image_tokens = IMG_CONTEXT_TOKEN * n_image_tok
        question_with_image = f"<img>{image_tokens}</img>\n{sample['question']}"

        user_part = f"<|im_start|>user\n{question_with_image}<|im_end|>\n"
        assistant_part = f"<|im_start|>assistant\n{sample['answer']}<|im_end|>"

        user_ids = tokenizer(user_part, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        assistant_ids = tokenizer(assistant_part, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

        input_ids = torch.cat([user_ids, assistant_ids], dim=0)
        labels = torch.cat([
            torch.full_like(user_ids, -100),
            assistant_ids,
        ], dim=0)

        max_len = n_image_tok + 256
        input_ids = input_ids[:max_len]
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
    setup_seeds(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir="tuned_checkpoint",
        setup_sublogger=True,
        sublogger_name=args.job_id,
    )

    device = torch.device(args.device)

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
    if args.distill_mode:
        logger.log("Loading full InternVL model as teacher...")
        full_model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )
        full_model.to(device)
        full_model.eval()
        for param in full_model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # 3. Load pruned (student) model
    # ------------------------------------------------------------------
    logger.log(f"Loading pruned model from: {args.pruned_ckpt}")
    model = load_pruned_model(args.model_name, args.pruned_ckpt, device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    # ------------------------------------------------------------------
    # 4. Apply LoRA
    # ------------------------------------------------------------------
    lora_targets = [t.strip() for t in args.lora_target_modules.split(",")]
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
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 5. Build dataset and dataloader
    # ------------------------------------------------------------------
    dataset = FinetuneDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        max_samples=args.max_samples,
        max_num_tiles=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: batch,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 6. Optimizer and scheduler
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dataloader) * args.num_epochs // args.gradient_accumulation_steps)

    def cosine_lr(step):
        if step >= total_steps:
            return 0.01
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

    # ------------------------------------------------------------------
    # 7. Set img_context_token_id on both student and teacher
    # ------------------------------------------------------------------
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    student_base = model.model if type(model).__name__ == "PeftModel" else model
    student_base.img_context_token_id = img_context_token_id
    if full_model is not None:
        full_model.img_context_token_id = img_context_token_id

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    logger.log(f"Starting fine-tuning: {args.num_epochs} epochs, "
               f"{len(dataset)} samples, bs={args.batch_size}, "
               f"grad_accum={args.gradient_accumulation_steps}")
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch_samples in enumerate(dataloader):
            base_model = student_base

            inputs = build_training_batch(batch_samples, base_model, tokenizer, device)

            outputs = model(**inputs)
            loss = outputs.loss

            if args.distill_mode and full_model is not None:
                with torch.no_grad():
                    teacher_outputs = full_model(**inputs)
                student_logits = outputs.logits
                teacher_logits = teacher_outputs.logits
                min_len = min(student_logits.shape[1], teacher_logits.shape[1])
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits[:, :min_len] / args.kd_temperature, dim=-1),
                    F.softmax(teacher_logits[:, :min_len] / args.kd_temperature, dim=-1),
                    reduction="batchmean",
                ) * (args.kd_temperature ** 2)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr_now = optimizer.param_groups[0]["lr"]
                logger.log(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.4f}, LR: {lr_now:.2e}"
                )

        avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
        logger.log(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # 9. Merge LoRA and save
    # ------------------------------------------------------------------
    logger.log("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    save_dir = os.path.join(logger.sub_dir, "finetuned_model")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    shape_map = {name: list(p.shape) for name, p in model.named_parameters()}
    with open(os.path.join(save_dir, "pruned_shapes.json"), "w") as f:
        json.dump(shape_map, f)
    tokenizer.save_pretrained(save_dir)

    logger.log(f"Fine-tuned model saved to {save_dir}")
    logger.log("[FINISH] - Finish Fine-tuning Pruned InternVL Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UKMP Fine-tune Pruned InternVL")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B")
    parser.add_argument("--pruned_ckpt", type=str, required=True,
                        help="Path to pruned model directory (with model.pt + pruned_shapes.json)")
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--save_ckpt_log_name", type=str, default="internvl_finetune")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--distill_mode", action="store_true",
                        help="Use full model as teacher for KD")
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
    