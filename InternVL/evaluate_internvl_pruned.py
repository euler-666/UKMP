"""
Evaluation script for pruned (and optionally fine-tuned) InternVL 3.5 1B.
Supports multiple VLM benchmark tasks: VQAv2, GQA, TextVQA, OKVQA, etc.
Can also run free-form generation quality checks.
"""

import argparse
import json
import logging
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_VIT_LAYERS = 24
NUM_LLM_LAYERS = 28


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
# Evaluation datasets
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
    """Generic VQA evaluation dataset.
    Expects JSON with entries: {"image": "path", "question": "...", "answer": "..."(optional)}
    """

    def __init__(self, data_path, image_root, max_samples=None):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item.get("image", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (448, 448))

        # Support both flat {"question","answer"} and LLaVA conversations format
        if "conversations" in item:
            convs = item["conversations"]
            question = convs[0]["value"].replace("<image>", "").replace("\n", " ").strip() if len(convs) > 0 else "Describe this image."
            answer = convs[1]["value"].strip() if len(convs) > 1 else ""
        else:
            question = item.get("question", item.get("text", "Describe this image."))
            answer = item.get("answer", item.get("answers", ""))
        qid = item.get("question_id", item.get("id", idx))

        category = item.get("category", "")

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "question_id": qid,
            "category": category,
        }


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
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_pil(image, input_size=448, max_num=12):
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def generate_response(model, tokenizer, image, question, device, max_new_tokens=256):
    """Generate a response from the InternVL model for a single image+question."""
    dtype = next(model.parameters()).dtype
    pixel_values = load_image_pil(image, input_size=448, max_num=12).to(device).to(dtype)

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    prompt = f"<image>\n{question}"

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response


def evaluate_vqa(model, tokenizer, dataset, device, output_path=None):
    """Run VQA evaluation: generate answers and compute exact-match accuracy."""
    results = []
    correct = 0
    total = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        gt_answer = str(sample["answer"]).lower().strip()
        qid = sample["question_id"]

        pred = generate_response(model, tokenizer, image, question, device)
        pred_clean = pred.lower().strip()

        is_correct = gt_answer in pred_clean or pred_clean in gt_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question_id": qid,
            "question": question,
            "prediction": pred,
            "answer": gt_answer,
            "correct": is_correct,
        })

        if (idx + 1) % 100 == 0:
            logger.info(
                f"Progress: {idx+1}/{len(dataset)}, "
                f"Accuracy so far: {correct/total*100:.2f}%"
            )

    accuracy = correct / total * 100 if total > 0 else 0
    logger.info(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total})")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return accuracy, results


def evaluate_generation(model, tokenizer, dataset, device, output_path=None, num_samples=50):
    """Run free-form generation and save outputs for qualitative inspection."""
    results = []
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"]
        question = sample.get("question", "Describe this image in detail.")

        pred = generate_response(model, tokenizer, image, question, device)
        results.append({
            "question_id": sample.get("question_id", idx),
            "question": question,
            "prediction": pred,
        })
        logger.info(f"[{idx+1}] Q: {question[:80]}... | A: {pred[:80]}...")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Generation results saved to {output_path}")

    return results


def _ocrbench_match(answers, predict, category):
    """OCRBench scoring: answer substring containment (VLMEvalKit convention)."""
    if category == "Handwritten Mathematical Expression Recognition":
        for ans in answers:
            a = ans.strip().replace("\n", " ").replace(" ", "")
            p = predict.strip().replace("\n", " ").replace(" ", "")
            if a in p:
                return True
    else:
        for ans in answers:
            a = ans.lower().strip().replace("\n", " ")
            p = predict.lower().strip().replace("\n", " ")
            if a in p:
                return True
    return False


OCRBENCH_AGG = {
    "Text Recognition": [
        "Regular Text Recognition", "Irregular Text Recognition",
        "Artistic Text Recognition", "Handwriting Recognition",
        "Digit String Recognition", "Non-Semantic Text Recognition",
    ],
    "Scene Text-centric VQA": ["Scene Text-centric VQA"],
    "Doc-oriented VQA": ["Doc-oriented VQA"],
    "Key Information Extraction": ["Key Information Extraction"],
    "Handwritten Math Expr Recognition": [
        "Handwritten Mathematical Expression Recognition"
    ],
}


def evaluate_ocrbench(model, tokenizer, dataset, device, output_path=None):
    """Run full OCRBench evaluation with per-category scoring.

    Expects dataset items to have 'answer' (list of acceptable strings)
    and 'category'.
    """
    per_cat = {}
    per_cat_total = {}
    results = []
    total_correct = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        qid = sample["question_id"]

        raw_answer = sample.get("answer", "")
        if isinstance(raw_answer, list):
            answers = [str(a) for a in raw_answer]
        elif isinstance(raw_answer, str):
            try:
                parsed = json.loads(raw_answer)
                if isinstance(parsed, list):
                    answers = [str(a) for a in parsed]
                else:
                    answers = [str(parsed)]
            except (json.JSONDecodeError, ValueError):
                answers = [raw_answer] if raw_answer else []
        else:
            answers = [str(raw_answer)]

        category = sample.get("category", "unknown")

        pred = generate_response(model, tokenizer, image, question, device)
        correct = _ocrbench_match(answers, pred, category)

        per_cat[category] = per_cat.get(category, 0) + int(correct)
        per_cat_total[category] = per_cat_total.get(category, 0) + 1
        if correct:
            total_correct += 1

        results.append({
            "question_id": qid,
            "question": question,
            "prediction": pred,
            "answer": answers,
            "category": category,
            "correct": correct,
        })

        if (idx + 1) % 100 == 0:
            logger.info(
                f"Progress: {idx+1}/{len(dataset)}, "
                f"Score so far: {total_correct}/{idx+1}"
            )

    agg_scores = {}
    for agg_name, sub_cats in OCRBENCH_AGG.items():
        agg_scores[agg_name] = sum(per_cat.get(c, 0) for c in sub_cats)
    agg_scores["Final Score"] = sum(agg_scores.values())
    agg_scores["Final Score Norm"] = agg_scores["Final Score"] / 10.0

    logger.info("=" * 60)
    logger.info("OCRBench Results:")
    for k, v in agg_scores.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    if output_path:
        out = {
            "scores": agg_scores,
            "per_category": {c: {"correct": per_cat.get(c, 0),
                                 "total": per_cat_total.get(c, 0)}
                             for c in per_cat_total},
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"OCRBench results saved to {output_path}")

    return agg_scores, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    setup_seeds(args.seed)

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load the model
    # ------------------------------------------------------------------
    if args.pruned_ckpt is not None:
        ckpt_path = args.pruned_ckpt

        if os.path.isdir(ckpt_path):
            sd_path = os.path.join(ckpt_path, "model.pt")
            shapes_path = os.path.join(ckpt_path, "pruned_shapes.json")
        else:
            sd_path = ckpt_path
            shapes_path = None

        logger.info(f"Loading base architecture from: {args.model_name}")
        model = AutoModel.from_pretrained(
            args.model_name,
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
        logger.info(f"Loading full model: {args.model_name}")
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Total parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 2. Load evaluation dataset
    # ------------------------------------------------------------------
    dataset = VQADataset(
        data_path=args.eval_data_path,
        image_root=args.eval_image_root,
        max_samples=args.max_eval_samples,
    )
    logger.info(f"Evaluation dataset loaded: {len(dataset)} samples")

    # ------------------------------------------------------------------
    # 3. Run evaluation
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_mode == "vqa":
        output_path = os.path.join(args.output_dir, f"vqa_results_{args.job_id}.json")
        accuracy, results = evaluate_vqa(model, tokenizer, dataset, device, output_path)
        logger.info(f"VQA Accuracy: {accuracy:.2f}%")

    elif args.eval_mode == "generation":
        output_path = os.path.join(args.output_dir, f"gen_results_{args.job_id}.json")
        results = evaluate_generation(
            model, tokenizer, dataset, device, output_path, args.max_eval_samples or 50
        )

    elif args.eval_mode == "ocrbench":
        output_path = os.path.join(args.output_dir, f"ocrbench_results_{args.job_id}.json")
        scores, results = evaluate_ocrbench(model, tokenizer, dataset, device, output_path)

    else:
        raise ValueError(f"Unknown eval mode: {args.eval_mode}")

    logger.info("[FINISH] - Evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pruned InternVL")

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B",
                        help="HuggingFace model name (for tokenizer and optionally full model)")
    parser.add_argument("--pruned_ckpt", type=str, default=None,
                        help="Path to pruned model checkpoint (.bin)")
    parser.add_argument("--job_id", type=str, default="eval",
                        help="Job ID for output naming")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation data
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="Path to evaluation data JSON")
    parser.add_argument("--eval_image_root", type=str, required=True,
                        help="Root directory for evaluation images")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Max evaluation samples (None = all)")

    # Eval config
    parser.add_argument("--eval_mode", type=str, default="vqa",
                        choices=["vqa", "generation", "ocrbench"],
                        help="Evaluation mode: vqa | generation | ocrbench")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for results")

    args = parser.parse_args()
    main(args)
