"""
End-to-end UKMP pipeline for InternVL 3.5 1B at 20% pruning ratio.
Stages: Prune -> Evaluate Pruned -> Fine-tune -> Evaluate Fine-tuned
"""
import subprocess

GPU = "0"
port = 18083

# ---- Pruning configuration ----
pruner_type = "taylor+knowledge"
taylor_type = "param_first"
granularity = "block"
pruning_ratio = 0.2
global_pruning = True
num_examples = 1000
imp_normalizer = "param"
select_loss = True
entropy_importance = True
multimodal = True

# ---- Paths (update these for your setup) ----
MODEL_NAME = "OpenGVLab/InternVL3_5-1B"
CALIB_DATA_PATH = "../LLaVA-CC3M-Pretrain-595K/chat.json"
CALIB_IMAGE_ROOT = "../LLaVA-CC3M-Pretrain-595K/images"
FINETUNE_DATA_PATH = "../LLaVA-CC3M-Pretrain-595K/chat.json"
FINETUNE_IMAGE_ROOT = "../LLaVA-CC3M-Pretrain-595K/images"
EVAL_DATA_PATH = "data/eval/vqa_eval.json"           # evaluation data JSON (update as needed)
EVAL_IMAGE_ROOT = "data/eval/images"                 # evaluation image root (update as needed)

global_str = "global" if global_pruning else "local"
norm_str = imp_normalizer if imp_normalizer is not None else "no"
select_loss_str = "-select_loss" if select_loss else ""
multimodal_str = "-multimodal" if multimodal else ""

job_id = (
    f"internvl-{num_examples}data-{pruner_type}-{taylor_type}-"
    f"{norm_str}_norm-{pruning_ratio}-{granularity}wise-{global_str}"
    f"{select_loss_str}{multimodal_str}"
)

# ================================================================
# Stage 1: Prune
# ================================================================
print("[START] - Start Pruning InternVL Model")
program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u ukmp_prune_internvl.py"
    f" --model_name {MODEL_NAME}"
    f" --device cuda"
    f" --save_ckpt_log_name ukmp_prune_internvl"
    f" --job_id {job_id}"
    f" --pruning_ratio {pruning_ratio}"
    f" --granularity {granularity}"
    f" --pruner_type {pruner_type}"
    f" --taylor {taylor_type}"
    f" --num_examples {num_examples}"
    f" --channel_per_step 1000"
    f" --calib_data_path {CALIB_DATA_PATH}"
    f" --calib_image_root {CALIB_IMAGE_ROOT}"
)
if global_pruning:
    program += " --global_pruning"
if imp_normalizer is not None:
    program += f" --imp_normalizer {imp_normalizer}"
if select_loss:
    program += " --select_loss"
if entropy_importance:
    program += " --entropy_importance"
if multimodal:
    program += " --multimodal"
subprocess.call(program, shell=True)

to_process_ckpt = "pytorch_model"
ckpt_dir = f"pruned_checkpoint/ukmp_prune_internvl/{job_id}"

# ================================================================
# Stage 2: Evaluate Pruned Model
# ================================================================
print("[START] - Start Evaluating Pruned Model")
eval_job_id = "evaluate_pruned"
program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {ckpt_dir}/{to_process_ckpt}.bin"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode vqa"
    f" --job_id {eval_job_id}"
    f" --output_dir eval_results/pruned_0.2"
)
subprocess.call(program, shell=True)

# ================================================================
# Stage 3: Fine-tune
# ================================================================
print("[START] - Start Fine-tuning Model")
wr_lora = True
distill = True
wr_lora_str = "-wr_lora" if wr_lora else ""
distill_str = "-distill" if distill else ""
tune_job_id = f"{job_id}+finetune{wr_lora_str}{distill_str}"

program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u ukmp_finetune_internvl.py"
    f" --model_name {MODEL_NAME}"
    f" --device cuda"
    f" --save_ckpt_log_name ukmp_finetune_internvl"
    f" --job_id {tune_job_id}"
    f" --pruned_ckpt {ckpt_dir}/{to_process_ckpt}.bin"
    f" --data_path {FINETUNE_DATA_PATH}"
    f" --image_root {FINETUNE_IMAGE_ROOT}"
    f" --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q,k,v,proj,fc1,fc2"
    f" --pruned_mask {ckpt_dir}/prune_masks.json"
    f" --num_epochs 3"
    f" --batch_size 4"
    f" --lr 2e-4"
)
if distill:
    program += " --distill_mode"
if wr_lora:
    program += " --wr_lora"
subprocess.call(program, shell=True)

# ================================================================
# Stage 4: Evaluate Fine-tuned Model
# ================================================================
print("[START] - Start Evaluating Fine-tuned Model")
eval_job_id = "evaluate_finetuned"
program = (
    f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_internvl_pruned.py"
    f" --model_name {MODEL_NAME}"
    f" --pruned_ckpt {ckpt_dir}/{to_process_ckpt}.bin"
    f" --peft_ckpt tuned_checkpoint/ukmp_finetune_internvl/{tune_job_id}"
    f" --pruned_mask {ckpt_dir}/prune_masks.json"
    f" --eval_data_path {EVAL_DATA_PATH}"
    f" --eval_image_root {EVAL_IMAGE_ROOT}"
    f" --eval_mode vqa"
    f" --job_id {eval_job_id}"
    f" --output_dir eval_results/finetuned_0.2"
)
subprocess.call(program, shell=True)

print("[DONE] - Full UKMP pipeline for InternVL (0.2 ratio) complete")
