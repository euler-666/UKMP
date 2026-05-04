"""
End-to-end UKMP pipeline for InternVL 3.5 1B with SFT data.
Stages: Generate calibration data -> Prune -> Fine-tune (multi-GPU) -> Evaluate on OCRBench

All key parameters are defined as variables at the top for easy tuning.
"""
import subprocess
import json
import os
import random
import sys

# ====================================================================
# Configuration — edit these variables for your setup
# ====================================================================

# --- Hardware ---
GPU = "0"                          # GPU for pruning and evaluation (single-GPU)
NGPU = 4                           # GPUs for finetuning (DDP)

# --- Model ---
MODEL_NAME = "OpenGVLab/InternVL3_5-1B"

# --- Pruning ---
PRUNING_RATIO = 0.05
MAX_PRUNING_RATIO = 0.5
GRANULARITY = "block"
PRUNER_TYPE = "taylor+knowledge"
TAYLOR_TYPE = "param_first"
NUM_CALIB_EXAMPLES = 1000
CHANNEL_PER_STEP = 200
GLOBAL_PRUNING = True
SELECT_LOSS = True
ENTROPY_IMPORTANCE = True
MULTIMODAL = True

# --- Finetuning ---
MAX_NUM_TILES = 6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
LR = 2e-4
NUM_EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 32
LORA_PRUNED_R = 8
KD_TEMPERATURE = 1.0
BETA1 = 1.0
BETA2 = 1.0
DISTILL_MODE = True
FEATURE_NORM = True
WR_LORA = True
LOG_INTERVAL = 100

# --- Data paths ---
SFT_DATA_CONFIG = "configs/sft_data_config.json"
SFT_SAMPLE_PERCENT = 40           # Use X% of SFT data (e.g. 40 for 40%, 100 for all)

# --- Evaluation ---
EVAL_DATA_PATH = "configs/ocrbench_full.json"
EVAL_IMAGE_ROOT = os.path.expanduser("~/LMUData/images/OCRBench")

# ====================================================================
# Derived variables
# ====================================================================
global_str = "global" if GLOBAL_PRUNING else "local"
select_loss_str = "-select_loss" if SELECT_LOSS else ""
multimodal_str = "-multimodal" if MULTIMODAL else ""

JOB_ID = (
    f"internvl-{NUM_CALIB_EXAMPLES}data-{PRUNER_TYPE}-{TAYLOR_TYPE}-"
    f"param_norm-{PRUNING_RATIO}-{GRANULARITY}wise-{global_str}"
    f"{select_loss_str}{multimodal_str}"
)

PRUNE_DIR = f"pruned_checkpoint/ukmp_prune_internvl/{JOB_ID}"
CALIB_JSON_PATH = f"{PRUNE_DIR}/sft_calibration_data.json"

wr_lora_str = "-wr_lora" if WR_LORA else ""
distill_str = "-distill" if DISTILL_MODE else ""
sample_tag = f"-{SFT_SAMPLE_PERCENT}pct" if SFT_SAMPLE_PERCENT < 100 else "-full"
epoch_tag = f"-{NUM_EPOCHS}ep" if NUM_EPOCHS != 1 else ""
tile_tag = f"-t{MAX_NUM_TILES}" if MAX_NUM_TILES != 6 else ""
TUNE_JOB_ID = f"{JOB_ID}+finetune{wr_lora_str}{distill_str}-sft-tiled{sample_tag}{epoch_tag}{tile_tag}"
FINETUNED_MODEL_DIR = f"tuned_checkpoint/internvl_finetune/{TUNE_JOB_ID}/finetuned_model"

# Compute max_samples from percentage (None means use all)
if SFT_SAMPLE_PERCENT < 100:
    with open(SFT_DATA_CONFIG, "r") as _f:
        _cfg = json.load(_f)
    _total = 0
    for _ds in _cfg:
        _ann = _ds["annotation"]
        with open(_ann) as _af:
            _count = sum(1 for _ in _af)
        _cap = _ds.get("max_samples")
        _total += min(_count, _cap) if _cap else _count
    MAX_SFT_SAMPLES = int(_total * SFT_SAMPLE_PERCENT / 100)
    print(f"SFT sampling: {SFT_SAMPLE_PERCENT}% of ~{_total} = {MAX_SFT_SAMPLES} samples")
else:
    MAX_SFT_SAMPLES = None


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
# Stage 0: Generate calibration JSON from SFT datasets
# ====================================================================
print(f"\n{'='*70}")
print("[START] Generating calibration data from SFT datasets")
print(f"{'='*70}")

os.makedirs(PRUNE_DIR, exist_ok=True)

with open(SFT_DATA_CONFIG, "r") as f:
    sft_config = json.load(f)

calib_items = []
per_dataset = max(1, NUM_CALIB_EXAMPLES // len(sft_config))

for ds in sft_config:
    annotation = ds["annotation"]
    image_root = ds["image_root"]
    items = []
    with open(annotation, "r") as f:
        if annotation.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        else:
            items = json.load(f)

    random.seed(42)
    random.shuffle(items)
    for item in items[:per_dataset]:
        img = item.get("image", "")
        if not os.path.isabs(img):
            img = os.path.join(image_root, img)
        item["image"] = img
        calib_items.append(item)

random.shuffle(calib_items)
calib_items = calib_items[:NUM_CALIB_EXAMPLES]

with open(CALIB_JSON_PATH, "w") as f:
    json.dump(calib_items, f)

print(f"  Saved {len(calib_items)} calibration samples to {CALIB_JSON_PATH}")
print("[DONE] Calibration data generated\n")

# ====================================================================
# Stage 1: Prune
# ====================================================================
prune_cmd = (
    f"CUDA_VISIBLE_DEVICES={GPU} CUDA_LAUNCH_BLOCKING=1 python -u ukmp_prune_internvl.py"
    f" --model_name {MODEL_NAME}"
    f" --device cuda"
    f" --save_ckpt_log_name ukmp_prune_internvl"
    f" --job_id {JOB_ID}"
    f" --pruning_ratio {PRUNING_RATIO}"
    f" --max_pruning_ratio {MAX_PRUNING_RATIO}"
    f" --granularity {GRANULARITY}"
    f" --pruner_type \"{PRUNER_TYPE}\""
    f" --taylor {TAYLOR_TYPE}"
    f" --num_examples {NUM_CALIB_EXAMPLES}"
    f" --channel_per_step {CHANNEL_PER_STEP}"
    f" --calibration_bs 1"
    f" --calib_data_path {CALIB_JSON_PATH}"
    f" --calib_image_root /"
)
if GLOBAL_PRUNING:
    prune_cmd += " --global_pruning"
if SELECT_LOSS:
    prune_cmd += " --select_loss"
if ENTROPY_IMPORTANCE:
    prune_cmd += " --entropy_importance"
if MULTIMODAL:
    prune_cmd += " --multimodal"

run(prune_cmd, "Pruning InternVL Model")

# ====================================================================
# Stage 2: Evaluate Pruned Model (OCRBench — full 1000 samples)
# ====================================================================
pruned_ckpt = f"{PRUNE_DIR}/pruned_model"
if not os.path.isdir(pruned_ckpt):
    pruned_ckpt = f"{PRUNE_DIR}/pytorch_model.bin"

eval_pruned_cmd = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {pruned_ckpt}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id eval_pruned_{JOB_ID}"
    f" --output_dir eval_results/pruned_{JOB_ID}"
)
run(eval_pruned_cmd, "Evaluating Pruned Model on OCRBench (1000 samples)")

# ====================================================================
# Stage 3: Fine-tune (multi-GPU DDP)
# ====================================================================
if NGPU > 1:
    gpu_list = ",".join(str(i) for i in range(NGPU))
    launcher = f"CUDA_VISIBLE_DEVICES={gpu_list} torchrun --nproc_per_node={NGPU}"
else:
    launcher = f"CUDA_VISIBLE_DEVICES={GPU} python -u"

finetune_cmd = (
    f"{launcher} ukmp_finetune_internvl.py"
    f" --model_name {MODEL_NAME}"
    f" --device cuda"
    f" --save_ckpt_log_name internvl_finetune"
    f" --job_id {TUNE_JOB_ID}"
    f" --pruned_ckpt {pruned_ckpt}"
    f" --data_config {SFT_DATA_CONFIG}"
    + (f" --max_samples {MAX_SFT_SAMPLES}" if MAX_SFT_SAMPLES is not None else "")
    + f" --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q,k,v,proj,fc1,fc2"
    f" --pruned_mask {PRUNE_DIR}/prune_masks.json"
    f" --num_epochs {NUM_EPOCHS}"
    f" --batch_size {BATCH_SIZE}"
    f" --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS}"
    f" --lr {LR}"
    f" --max_num_tiles {MAX_NUM_TILES}"
    f" --lora_r {LORA_R}"
    f" --lora_alpha {LORA_ALPHA}"
    f" --lora_pruned_r {LORA_PRUNED_R}"
    f" --log_interval {LOG_INTERVAL}"
    f" --kd_temperature {KD_TEMPERATURE}"
    f" --beta1 {BETA1}"
    f" --beta2 {BETA2}"
)
if DISTILL_MODE:
    finetune_cmd += " --distill_mode"
if FEATURE_NORM:
    finetune_cmd += " --feature_norm"
if WR_LORA:
    finetune_cmd += " --wr_lora"

run(finetune_cmd, "Fine-tuning Pruned Model with SFT Data")

# ====================================================================
# Stage 4: Evaluate Fine-tuned Model (OCRBench)
# ====================================================================
eval_ft_cmd = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {FINETUNED_MODEL_DIR}"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode ocrbench"
    f" --job_id eval_finetuned_{TUNE_JOB_ID}"
    f" --output_dir eval_results/finetuned_{TUNE_JOB_ID}"
)
run(eval_ft_cmd, "Evaluating Fine-tuned Model on OCRBench (1000 samples)")

# ====================================================================
# Summary
# ====================================================================
print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Pruned model:     {pruned_ckpt}")
print(f"  Fine-tuned model: {FINETUNED_MODEL_DIR}")
print(f"  Pruned eval:      eval_results/pruned_{JOB_ID}/")
print(f"  Fine-tuned eval:  eval_results/finetuned_{TUNE_JOB_ID}/")
print(f"{'='*70}\n")
