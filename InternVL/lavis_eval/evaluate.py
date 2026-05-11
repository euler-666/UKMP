"""Evaluate InternVL on LAVIS-style VQA tasks (VQAv2, OK-VQA, GQA).

Usage (single-task, single-checkpoint):

    cd ~/UKMP/InternVL
    python -u lavis_eval/evaluate.py \\
        --task okvqa \\
        --model_name OpenGVLab/InternVL3_5-1B \\
        --pruned_ckpt tuned_checkpoint/.../finetuned_model \\
        --ann_path /home/rishabh/UKMP/LAVIS/data/datasets/okvqa/annotations/vqa_val_eval.json \\
        --image_root /home/rishabh/UKMP/LAVIS/data/datasets/coco/images/ \\
        --apply_lemmatizer \\
        --max_samples 1000 \\
        --output_dir eval_results/lavis_vqa/okvqa_finetuned_0.2_mm \\
        --job_id finetuned_0.2_mm

Scoring matches LAVIS BLIP-2 evaluation:
    - VQAv2 / OK-VQA: official VQAEval (10-answer avg, processPunctuation +
      processDigitArticle). OK-VQA additionally lemmatizes predictions.
    - GQA: exact match after VQAEval normalization (single GT answer).

The prompt uses InternVL's native chat structure (``<image>\\n{question}``) —
no BLIP-style ``Short answer:`` suffix is added.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Make the package importable when running this file directly.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from datasets import GQAEvalDataset, VQAv2EvalDataset  # noqa: E402
from generation import generate_response  # noqa: E402
from model_loader import load_internvl  # noqa: E402
from vqa_scoring import Lemmatizer, exact_match, vqa_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lavis_eval")


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def _build_dataset(task: str, ann_path: str, image_root: str, max_samples):
    if task in ("vqav2", "okvqa"):
        return VQAv2EvalDataset(ann_path, image_root, max_samples)
    if task == "gqa":
        return GQAEvalDataset(ann_path, image_root, max_samples)
    raise ValueError(f"Unknown task: {task}")


def _evaluate(
    task: str,
    model,
    tokenizer,
    dataset,
    device,
    apply_lemmatizer: bool,
    max_new_tokens: int,
    num_beams: int,
    max_tiles: int,
    short_hint: bool,
    log_every: int,
):
    """Run generation + scoring. Returns (overall_metric, results)."""
    lemmatizer = Lemmatizer() if (task == "okvqa" and apply_lemmatizer) else None
    if lemmatizer is not None:
        logger.info("OK-VQA lemmatizer enabled (matches LAVIS apply_lemmatizer=True)")

    results = []
    sum_score = 0.0
    n = len(dataset)
    t0 = time.time()

    for idx in range(n):
        sample = dataset[idx]
        pred = generate_response(
            model,
            tokenizer,
            sample["image"],
            sample["question"],
            device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            max_tiles=max_tiles,
            short_hint=short_hint,
        )

        pred_scored = pred
        if lemmatizer is not None:
            pred_scored = lemmatizer(pred_scored)

        if task in ("vqav2", "okvqa"):
            score = vqa_score(pred_scored, sample["answers"])
            entry_gt = sample["answers"]
        else:  # gqa
            score = exact_match(pred_scored, sample["answer"])
            entry_gt = sample["answer"]

        sum_score += score

        results.append({
            "question_id": sample["question_id"],
            "question": sample["question"],
            "prediction": pred,
            "prediction_for_score": pred_scored if lemmatizer is not None else None,
            "answer": entry_gt,
            "score": score,
        })

        if (idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            logger.info(
                f"[{task}] {idx+1}/{n} | running acc = {100*sum_score/(idx+1):.2f}% "
                f"| {rate:.2f} samples/s"
            )

    overall = 100.0 * sum_score / n if n > 0 else 0.0
    return overall, results


def main():
    parser = argparse.ArgumentParser(description="LAVIS-style VQA eval for InternVL")
    parser.add_argument("--task", required=True, choices=["vqav2", "okvqa", "gqa"])
    parser.add_argument("--model_name", default="OpenGVLab/InternVL3_5-1B")
    parser.add_argument("--pruned_ckpt", default=None,
                        help="Path to a pruned/finetuned checkpoint directory containing "
                             "model.pt + pruned_shapes.json. Omit for baseline.")

    parser.add_argument("--ann_path", required=True, help="LAVIS-format annotations JSON")
    parser.add_argument("--image_root", required=True, help="Root directory holding images")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on number of samples to evaluate (None = full)")

    parser.add_argument("--apply_lemmatizer", action="store_true",
                        help="Lemmatize predictions before scoring (used for OK-VQA)")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--max_tiles", type=int, default=12,
                        help="Max InternVL image tiles (12 for OCRBench-style, 6 is also common)")
    short_group = parser.add_mutually_exclusive_group()
    short_group.add_argument(
        "--short_hint", dest="short_hint", action="store_true",
        help="Append 'Answer the question using a single word or phrase.' to the prompt "
             "(matches the InternVL VQA evaluation prompt used in VLMEvalKit). Default ON.")
    short_group.add_argument(
        "--no_short_hint", dest="short_hint", action="store_false",
        help="Use only the bare '<image>\\n{question}' prompt (typically scores ~0 on "
             "VQAv2/OK-VQA/GQA because answers come out verbose).")
    parser.set_defaults(short_hint=True)

    parser.add_argument("--job_id", default="eval", help="Used for output filename")
    parser.add_argument("--output_dir", default="eval_results/lavis_vqa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    setup_seeds(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Task              : {args.task}")
    logger.info(f"Model             : {args.model_name}")
    logger.info(f"Pruned ckpt       : {args.pruned_ckpt}")
    logger.info(f"Annotations       : {args.ann_path}")
    logger.info(f"Image root        : {args.image_root}")
    logger.info(f"Max samples       : {args.max_samples}")
    logger.info(f"num_beams         : {args.num_beams}")
    logger.info(f"max_new_tokens    : {args.max_new_tokens}")
    logger.info(f"apply_lemmatizer  : {args.apply_lemmatizer}")
    logger.info(f"short_hint        : {args.short_hint}")

    model, tokenizer = load_internvl(args.model_name, args.pruned_ckpt, args.device)

    dataset = _build_dataset(args.task, args.ann_path, args.image_root, args.max_samples)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    overall, results = _evaluate(
        task=args.task,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        apply_lemmatizer=args.apply_lemmatizer,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        max_tiles=args.max_tiles,
        short_hint=args.short_hint,
        log_every=args.log_every,
    )

    out_file = os.path.join(args.output_dir, f"{args.task}_results_{args.job_id}.json")
    payload = {
        "task": args.task,
        "model_name": args.model_name,
        "pruned_ckpt": args.pruned_ckpt,
        "job_id": args.job_id,
        "num_samples": len(results),
        "accuracy": overall,
        "apply_lemmatizer": args.apply_lemmatizer,
        "short_hint": args.short_hint,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"[{args.task}/{args.job_id}] Final score: {overall:.2f}% "
        f"({len(results)} samples) -> {out_file}"
    )


if __name__ == "__main__":
    main()
