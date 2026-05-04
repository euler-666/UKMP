"""
Run OCRBench (1000 samples) on baseline and/or finetuned InternVL checkpoints.
Usage: cd InternVL && python scripts/eval_ocrbench.py
"""
import subprocess
import sys

GPU = "0"
MODEL_NAME = "OpenGVLab/InternVL3_5-1B"
EVAL_DATA_PATH = "configs/ocrbench_full.json"
EVAL_IMAGE_ROOT = "~/LMUData/images/OCRBench"

FINETUNED_CKPT = (
    "tuned_checkpoint/internvl_finetune/"
    "internvl-1000data-taylor+knowledge-param_first-param_norm-0.05-"
    "blockwise-global-select_loss-multimodal+finetune-wr_lora-distill-sft-tiled-40pct/"
    "finetuned_model"
)

PRUNED_CKPT = (
    "pruned_checkpoint/ukmp_prune_internvl/"
    "internvl-1000data-taylor+knowledge-param_first-param_norm-0.05-"
    "blockwise-global-select_loss-multimodal/pruned_model"
)


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"[START] {desc}")
    print(f"{'='*70}")
    print(f"CMD: {cmd}\n")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"[FAILED] {desc} (exit code {ret})")
        sys.exit(ret)
    print(f"[DONE] {desc}\n")


# --- Baseline (full unpruned model) ---
run(
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id baseline"
    f" --output_dir eval_results/ocrbench_baseline",
    "OCRBench — Baseline (unpruned)"
)

# --- Pruned (5%) ---
run(
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {PRUNED_CKPT}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id pruned_0.05"
    f" --output_dir eval_results/ocrbench_pruned_0.05",
    "OCRBench — Pruned 5%"
)

# --- Finetuned (pruned 5% + SFT recovery) ---
run(
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {FINETUNED_CKPT}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id finetuned_0.05"
    f" --output_dir eval_results/ocrbench_finetuned_0.05",
    "OCRBench — Finetuned (pruned 5% + SFT recovery)"
)

# --- Summary ---
import json
import os

print(f"\n{'='*70}")
print("OCRBench COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<45} {'Score':>8} {'Norm':>8}")
print("-" * 65)

for label, path in [
    ("Baseline (unpruned)", "eval_results/ocrbench_baseline/ocrbench_results_baseline.json"),
    ("Pruned 5%", "eval_results/ocrbench_pruned_0.05/ocrbench_results_pruned_0.05.json"),
    ("Finetuned (5% + SFT)", "eval_results/ocrbench_finetuned_0.05/ocrbench_results_finetuned_0.05.json"),
]:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        scores = data["scores"]
        print(f"{label:<45} {scores['Final Score']:>6}/1000 {scores['Final Score Norm']:>7.1f}")
    else:
        print(f"{label:<45} {'(not found)':>8}")

print(f"{'='*70}\n")
