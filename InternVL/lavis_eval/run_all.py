"""Run LAVIS-style VQA evaluation on every InternVL checkpoint x every task.

Usage (from the InternVL repo root):
    cd ~/UKMP/InternVL
    python -u lavis_eval/run_all.py

Tweak the constants at the top to change GPU, sample count, etc.
The pipeline mirrors the BLIP-2 sweep in LAVIS/scripts/structured_blip2/ukmp_0.2.py
but adapted to InternVL's checkpoint layout and prompt structure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GPU = "0"
BASE_MODEL = "OpenGVLab/InternVL3_5-1B"
INTERNVL_ROOT = Path(__file__).resolve().parent.parent  # ~/UKMP/InternVL
OUTPUT_ROOT = INTERNVL_ROOT / "eval_results" / "lavis_vqa"

# Per-task sample cap (set to None for full eval).
# VQAv2 val has 214k Q's — leave at a few thousand for tractable sweeps.
MAX_SAMPLES = {
    "vqav2": 1000,
    "okvqa": 1000,  # full OK-VQA val is 5046; reduce for speed
    "gqa": 1000,    # full testdev_balanced is ~12.5k
}

NUM_BEAMS = 5
MAX_NEW_TOKENS = 10
MAX_TILES = 12
# Append "Answer the question using a single word or phrase." to every prompt.
# This is the standard InternVL VQA evaluation prompt (used in VLMEvalKit).
# Without it, the chat-tuned model returns verbose sentences and exact-match
# scoring collapses to ~0–3% even for the baseline.
SHORT_HINT = True

# ----------------------------------------------------------------------
# Datasets (mirrors LAVIS BLIP-2 eval configs)
# ----------------------------------------------------------------------
LAVIS_DATA = Path("/home/rishabh/UKMP/LAVIS/data/datasets")

TASKS = {
    "vqav2": {
        "ann": LAVIS_DATA / "coco/annotations/vqa_val_eval.json",
        "img": LAVIS_DATA / "coco/images/",
        "apply_lemmatizer": False,
        # Source URL for the annotation JSON if missing on disk.
        "ann_url": (
            "https://storage.googleapis.com/sfr-vision-language-research/"
            "LAVIS/datasets/vqav2/vqa_val_eval.json"
        ),
    },
    "okvqa": {
        "ann": LAVIS_DATA / "okvqa/annotations/vqa_val_eval.json",
        "img": LAVIS_DATA / "coco/images/",
        # VLMEvalKit does NOT lemmatize OK-VQA predictions; LAVIS BLIP-2 does.
        # Toggle to True to reproduce the LAVIS apply_lemmatizer=True setting.
        "apply_lemmatizer": False,
        "ann_url": (
            "https://storage.googleapis.com/sfr-vision-language-research/"
            "LAVIS/datasets/okvqa/okvqa_val_eval.json"
        ),
    },
    "gqa": {
        "ann": LAVIS_DATA / "gqa/annotations/testdev_balanced_questions.json",
        "img": Path("/home/rishabh/InternVL-Chat-V1-2-SFT-Data/data/gqa/images/"),
        "apply_lemmatizer": False,
        "ann_url": (
            "https://storage.googleapis.com/sfr-vision-language-research/"
            "LAVIS/datasets/gqa/testdev_balanced_questions.json"
        ),
    },
}

# ----------------------------------------------------------------------
# Checkpoints to evaluate — the same 7 methods we benchmarked on OCRBench
# ----------------------------------------------------------------------
PRUNE_ROOT = INTERNVL_ROOT / "pruned_checkpoint" / "ukmp_prune_internvl"
TUNE_ROOT = INTERNVL_ROOT / "tuned_checkpoint" / "internvl_finetune"

P_05_MM = "internvl-1000data-taylor+knowledge-param_first-param_norm-0.05-blockwise-global-select_loss-multimodal"
P_02_MM = "internvl-1000data-taylor+knowledge-param_first-param_norm-0.2-blockwise-global-select_loss-multimodal"
P_02_MMQ = "internvl-1000data-taylor+knowledge-param_first-param_norm-0.2-blockwise-global-select_loss-multimodal-quota"

F_05_MM = P_05_MM + "+finetune-wr_lora-distill-sft-tiled-40pct"
F_02_MM = P_02_MM + "+finetune-wr_lora-distill-sft-tiled-full"
F_02_MMQ = P_02_MMQ + "+finetune-wr_lora-distill-sft-tiled-full"


def _pruned_dir(name):
    return str(PRUNE_ROOT / name / "pruned_model")


def _tuned_dir(name):
    return str(TUNE_ROOT / name / "finetuned_model")


METHODS = [
    # (short_id, ckpt_dir_or_None)
    ("baseline",            None),
    ("pruned_0.05_mm",      _pruned_dir(P_05_MM)),
    ("finetuned_0.05_mm",   _tuned_dir(F_05_MM)),
    ("pruned_0.2_mm",       _pruned_dir(P_02_MM)),
    ("pruned_0.2_mmq",      _pruned_dir(P_02_MMQ)),
    ("finetuned_0.2_mm",    _tuned_dir(F_02_MM)),
    ("finetuned_0.2_mmq",   _tuned_dir(F_02_MMQ)),
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _ensure_annotation(task, cfg):
    ann = cfg["ann"]
    if ann.exists():
        return
    ann.parent.mkdir(parents=True, exist_ok=True)
    url = cfg.get("ann_url")
    if not url:
        raise FileNotFoundError(f"Missing annotation for {task}: {ann}")
    print(f"[download] {task}: {url} -> {ann}")
    urllib.request.urlretrieve(url, ann)


def _run(cmd, desc):
    print("\n" + "=" * 78)
    print(f"[START] {desc}")
    print("=" * 78)
    print(f"CMD: {cmd}\n")
    ret = subprocess.call(cmd, shell=True)
    status = "DONE" if ret == 0 else f"FAILED ({ret})"
    print(f"[{status}] {desc}")
    return ret


def _result_path(task, method_id):
    out_dir = OUTPUT_ROOT / f"{task}_{method_id}"
    return out_dir, out_dir / f"{task}_results_{method_id}.json"


def _summarize():
    print("\n" + "=" * 78)
    print("LAVIS-VQA SUMMARY")
    print("=" * 78)
    header = f"{'Method':<22} " + " ".join(f"{t.upper():>9}" for t in TASKS)
    print(header)
    print("-" * len(header))
    for method_id, _ in METHODS:
        cells = []
        for task in TASKS:
            _, path = _result_path(task, method_id)
            if path.exists():
                try:
                    data = json.load(open(path))
                    cells.append(f"{data['accuracy']:>8.2f}%")
                except Exception:
                    cells.append("   ERR  ")
            else:
                cells.append("    -   ")
        print(f"{method_id:<22} " + " ".join(cells))
    print("=" * 78)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.chdir(INTERNVL_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for task, cfg in TASKS.items():
        _ensure_annotation(task, cfg)

    for method_id, ckpt in METHODS:
        for task, cfg in TASKS.items():
            out_dir, out_file = _result_path(task, method_id)
            if out_file.exists():
                print(f"[skip] {task}/{method_id} already exists: {out_file}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = (
                f"CUDA_VISIBLE_DEVICES={GPU} python -u lavis_eval/evaluate.py"
                f" --task {task}"
                f" --model_name {BASE_MODEL}"
                f" --ann_path {cfg['ann']}"
                f" --image_root {cfg['img']}"
                f" --num_beams {NUM_BEAMS}"
                f" --max_new_tokens {MAX_NEW_TOKENS}"
                f" --max_tiles {MAX_TILES}"
                f" --max_samples {MAX_SAMPLES[task]}"
                f" --job_id {method_id}"
                f" --output_dir {out_dir}"
            )
            cmd += " --short_hint" if SHORT_HINT else " --no_short_hint"
            if cfg["apply_lemmatizer"]:
                cmd += " --apply_lemmatizer"
            if ckpt is not None:
                cmd += f" --pruned_ckpt {ckpt}"

            _run(cmd, f"{task} / {method_id}")

    _summarize()


if __name__ == "__main__":
    main()
