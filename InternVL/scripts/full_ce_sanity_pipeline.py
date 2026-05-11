"""
Sanity / control pipeline: plain CE fine-tuning of FULL (unpruned) InternVL 1B
on the same SFT data used by the pruning pipeline.

Use this to isolate whether the SFT dataset itself is the source of any
downstream regression (vs. pruning + distillation in the main pipeline).

Stages: Fine-tune (multi-GPU, CE only) -> Evaluate on OCRBench
"""
import os
import subprocess
import sys

# ====================================================================
# Configuration
# ====================================================================

# --- Hardware ---
GPU = "0"            # GPU for evaluation (single-GPU)
NGPU = 4             # GPUs for finetuning (DDP)

# --- Model ---
MODEL_NAME = "OpenGVLab/InternVL3_5-1B"

# --- Finetuning ---
MAX_NUM_TILES = 6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
LR = 2e-4
NUM_EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 32
LOG_INTERVAL = 100
FULL_FT = False      # True = train all params (needs much more VRAM)

# --- Data ---
SFT_DATA_CONFIG = "configs/sft_data_config.json"
MAX_SFT_SAMPLES = None    # None = use everything in the config

# --- Evaluation ---
EVAL_DATA_PATH = "configs/ocrbench_full.json"
EVAL_IMAGE_ROOT = os.path.expanduser("~/LMUData/images/OCRBench")

# ====================================================================
# Derived
# ====================================================================
ft_tag = "-fullft" if FULL_FT else f"-lora_r{LORA_R}"
sample_tag = f"-{MAX_SFT_SAMPLES}samp" if MAX_SFT_SAMPLES else "-full"
TUNE_JOB_ID = f"internvl1b-fullmodel-ce{ft_tag}{sample_tag}-{NUM_EPOCHS}ep-t{MAX_NUM_TILES}"
FINETUNED_MODEL_DIR = f"tuned_checkpoint/internvl_full_ce/{TUNE_JOB_ID}/finetuned_model"


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


# ====================================================================
# Stage 1: Fine-tune (multi-GPU DDP)
# ====================================================================
if NGPU > 1:
    gpu_list = ",".join(str(i) for i in range(NGPU))
    launcher = f"CUDA_VISIBLE_DEVICES={gpu_list} torchrun --nproc_per_node={NGPU}"
else:
    launcher = f"CUDA_VISIBLE_DEVICES={GPU} python -u"

finetune_cmd = (
    f"{launcher} finetune_internvl_full_ce.py"
    f" --model_name {MODEL_NAME}"
    f" --save_ckpt_log_name internvl_full_ce"
    f" --job_id {TUNE_JOB_ID}"
    f" --data_config {SFT_DATA_CONFIG}"
    f" --num_epochs {NUM_EPOCHS}"
    f" --batch_size {BATCH_SIZE}"
    f" --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS}"
    f" --lr {LR}"
    f" --max_num_tiles {MAX_NUM_TILES}"
    f" --lora_r {LORA_R}"
    f" --lora_alpha {LORA_ALPHA}"
    f" --log_interval {LOG_INTERVAL}"
)
if MAX_SFT_SAMPLES is not None:
    finetune_cmd += f" --max_samples {MAX_SFT_SAMPLES}"
if FULL_FT:
    finetune_cmd += " --full_ft"

run(finetune_cmd, "Fine-tuning FULL InternVL 1B (CE only)")

# ====================================================================
# Stage 2: Evaluate on OCRBench
# ====================================================================
# The full-model save uses HF format, so we pass it as --model_name
# (not --pruned_ckpt) to the eval script.
eval_cmd = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {FINETUNED_MODEL_DIR}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id eval_{TUNE_JOB_ID}"
    f" --output_dir eval_results/full_ce_{TUNE_JOB_ID}"
)
run(eval_cmd, "Evaluating Fine-tuned Full Model on OCRBench")

# ====================================================================
# Summary
# ====================================================================
print(f"\n{'='*70}")
print("FULL-MODEL CE SANITY PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Fine-tuned model: {FINETUNED_MODEL_DIR}")
print(f"  Eval results:     eval_results/full_ce_{TUNE_JOB_ID}/")
print(f"{'='*70}\n")
