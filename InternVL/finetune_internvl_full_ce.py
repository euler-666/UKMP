"""
Plain CE fine-tuning of the FULL (unpruned) InternVL 3.5 1B.

This script is a sanity / control experiment for the pruning pipeline:
no pruning, no distillation, no progressive stages, no weight-recalling.
Just LoRA (or full-parameter) fine-tuning with cross-entropy loss on the
provided SFT data, so we can isolate whether the dataset itself is the
source of any downstream regression.

The dataset / preprocessing / batch construction are imported from
`ukmp_finetune_internvl` so the data pipeline is byte-for-byte identical
to the pruned-model fine-tune. The only differences are:
  - we load the unpruned base model directly via `AutoModel.from_pretrained`
  - the loss is exclusively `outputs.loss` (CE on label tokens)
  - the model is saved in HF format via `save_pretrained` so it can be
    loaded by evaluation scripts using `--model_name <save_dir>`.

Launch (multi-GPU DDP):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      finetune_internvl_full_ce.py \
      --model_name OpenGVLab/InternVL3_5-1B \
      --data_config configs/sft_data_config.json \
      --job_id full_ce_sanity \
      --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 32 --lr 2e-4
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer

from internvl_lib.common.logger import LoggerWithDepth
from internvl_lib.peft import LoraConfig, get_peft_model

from ukmp_finetune_internvl import (
    FinetuneDataset,
    build_training_batch,
    get_rank,
    get_world_size,
    is_dist,
    is_main_process,
    setup_distributed,
    setup_seeds,
)


def load_full_model(model_name, device, dtype=torch.bfloat16):
    """Load the unpruned InternVL base model in eval-ready state.

    Unlike `load_pruned_model`, we do NOT decouple the fused qkv layers —
    we keep the original fused weights so the model is identical to the
    HuggingFace release, and the saved checkpoint stays HF-compatible.
    """
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return model


def main(args):
    local_rank, device = setup_distributed()
    args.device = str(device)
    setup_seeds(args.seed + get_rank())

    if is_main_process():
        os.makedirs("tuned_checkpoint", exist_ok=True)
        os.makedirs(os.path.join("tuned_checkpoint", args.save_ckpt_log_name),
                    exist_ok=True)
        logger = LoggerWithDepth(
            env_name=args.save_ckpt_log_name,
            config=args.__dict__,
            root_dir="tuned_checkpoint",
            setup_sublogger=True,
            sublogger_name=args.job_id,
        )
    else:
        class _SilentLogger:
            def __getattr__(self, _):
                return lambda *a, **kw: None
            sub_dir = "/tmp"
        logger = _SilentLogger()

    # ------------------------------------------------------------------
    # 1. Tokenizer + full base model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.log(f"Loading full base model: {args.model_name}")
    model = load_full_model(args.model_name, device)

    # Disable dropout for deterministic SFT (matches the pruning-side script).
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    # Disable gradient checkpointing on both ViT and LLM. InternVL's modeling
    # code calls torch.utils.checkpoint with the legacy use_reentrant=True,
    # which is incompatible with DDP + LoRA: the inner reentrant backward
    # fires DDP's autograd hooks a second time on the same params and
    # crashes with "Expected to mark a variable ready only once". With LoRA
    # we only have ~16M trainable params, so the extra activation memory is
    # easily affordable on a 1B model at bs=1.
    def _disable_gc(m):
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass
        for sub in m.modules():
            if hasattr(sub, "gradient_checkpointing"):
                sub.gradient_checkpointing = False

    if hasattr(model, "vision_model"):
        _disable_gc(model.vision_model)
    if hasattr(model, "language_model"):
        _disable_gc(model.language_model)
    logger.log("Disabled gradient checkpointing on ViT and LLM "
               "(required for DDP + LoRA).")

    # ------------------------------------------------------------------
    # 2. LoRA wrap (default) or full-parameter fine-tuning
    # ------------------------------------------------------------------
    if args.full_ft:
        logger.log("Full-parameter fine-tuning (LoRA disabled).")
        for p in model.parameters():
            p.requires_grad = True
    else:
        lora_targets = [t.strip() for t in args.lora_target_modules.split(",")]
        logger.log(f"Setting up LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
                   f"targets={lora_targets}")
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

    # ------------------------------------------------------------------
    # 3. img_context_token_id (required for InternVL forward pass)
    # ------------------------------------------------------------------
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    base_model = model.model if type(model).__name__ == "PeftModel" else model
    base_model.img_context_token_id = img_context_token_id

    # Save reference to the PEFT state_dict before DDP wrapping so we can
    # call merge_and_unload() at the end.
    old_state_dict = model.state_dict if not args.full_ft else None

    # ------------------------------------------------------------------
    # 4. DDP wrap
    # ------------------------------------------------------------------
    if is_dist():
        # find_unused_parameters=True is the safe default: works whether or
        # not every LoRA adapter participates in every forward (handy if the
        # loss is later changed to something modality-specific). DDP will
        # print a benign warning if nothing is actually unused — that's a
        # cosmetic perf hint, not a bug.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        logger.log(f"Wrapped model in DDP across {get_world_size()} GPUs")

    # ------------------------------------------------------------------
    # 5. Dataset + dataloader
    # ------------------------------------------------------------------
    if not args.data_config and not args.data_path:
        raise ValueError("Either --data_config or --data_path must be provided.")

    dataset = FinetuneDataset(
        data_path=args.data_path,
        image_root=args.image_root or "",
        max_samples=args.max_samples,
        max_num_tiles=args.max_num_tiles,
        data_config=args.data_config,
    )
    logger.log(f"Dataset: {len(dataset)} samples, max_num_tiles={args.max_num_tiles}")

    sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        collate_fn=lambda batch: batch,
        drop_last=True,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 6. Optimizer + cosine LR schedule
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.log(f"Trainable parameters: {n_trainable:,}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(
        1, len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )

    def cosine_lr(step, _total=total_steps):
        if step >= _total:
            return 0.01
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * step / _total))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

    # ------------------------------------------------------------------
    # 7. Training loop (CE only)
    # ------------------------------------------------------------------
    logger.log(f"=== Training | {len(dataset)} samples, {args.num_epochs} epoch(s), "
               f"{get_world_size()} GPU(s), bs={args.batch_size}, "
               f"grad_accum={args.gradient_accumulation_steps} ===")

    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_logged = 0

        for batch_idx, batch_samples in enumerate(dataloader):
            inputs = build_training_batch(batch_samples, base_model, tokenizer, device)

            # InternViT uses gradient checkpointing, and with LoRA the base
            # ViT params have requires_grad=False. If pixel_values also have
            # requires_grad=False, torch.utils.checkpoint short-circuits the
            # backward pass for the ViT segment ("None of the inputs have
            # requires_grad=True. Gradients will be None") and the ViT-side
            # LoRA adapters never receive a gradient. Detach + require grad
            # on pixel_values gives autograd a starting point so the
            # checkpointed backward runs all the way through the ViT.
            if inputs["pixel_values"].dtype.is_floating_point:
                inputs["pixel_values"] = (
                    inputs["pixel_values"].detach().requires_grad_(True)
                )

            outputs = model(**inputs)
            loss = outputs.loss

            loss_val = loss.item()
            bad_loss = math.isnan(loss_val) or math.isinf(loss_val)
            if bad_loss:
                # Keep the graph connected (so DDP allreduce still fires
                # on every rank) but contribute zero gradient.
                loss = loss * 0.0
                loss_val = 0.0

            (loss / args.gradient_accumulation_steps).backward()
            if not bad_loss:
                epoch_loss += loss_val
                n_logged += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                has_nan = any(
                    torch.isnan(p.grad).any() for p in trainable_params
                    if p.grad is not None
                )
                if has_nan:
                    optimizer.zero_grad()
                    global_step += 1
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / max(n_logged, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                logger.log(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.4f}, LR: {lr_now:.2e}"
                )

        avg_epoch_loss = epoch_loss / max(n_logged, 1)
        logger.log(f"Epoch {epoch+1} complete. Avg CE loss: {avg_epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # 8. Merge LoRA (if used) and save (rank 0 only)
    # ------------------------------------------------------------------
    if is_dist():
        dist.barrier()
        model = model.module

    if not args.full_ft:
        logger.log("Merging LoRA adapters into base model...")
        model.state_dict = old_state_dict
        model = model.merge_and_unload()
    model.eval()

    if is_main_process():
        save_dir = os.path.join(logger.sub_dir, "finetuned_model")
        os.makedirs(save_dir, exist_ok=True)

        # Save in HF format so the eval script can load via
        #   --model_name <save_dir>   (no --pruned_ckpt needed)
        # safe_serialization=True writes model.safetensors. Required because
        # transformers >=4.43 refuses to load pickle checkpoints unless
        # torch>=2.6 (CVE-2025-32434), and this env pins torch 2.4.
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)

        logger.log(f"Fine-tuned model saved to {save_dir}")
        logger.log("[FINISH] - Full-model CE fine-tune complete")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plain CE SFT for full (unpruned) InternVL 3.5 1B"
    )

    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B")
    parser.add_argument("--job_id", type=str, default="full_ce_sanity")
    parser.add_argument("--save_ckpt_log_name", type=str, default="internvl_full_ce")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, default=None,
                        help="Single training file (JSON or JSONL). "
                             "Ignored when --data_config is set.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Image root for --data_path.")
    parser.add_argument("--data_config", type=str, default=None,
                        help="Multi-dataset SFT config (same format as the "
                             "pruning pipeline).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_num_tiles", type=int, default=6,
                        help="Max image tiles for dynamic preprocessing "
                             "(higher = better OCR but more VRAM).")

    # LoRA
    parser.add_argument("--full_ft", action="store_true",
                        help="Disable LoRA and train all parameters "
                             "(needs much more VRAM).")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
                "qkv,proj,fc1,fc2",
        help="Comma-separated linear module suffixes to attach LoRA to. "
             "Note: full base model uses fused `qkv` for the ViT, not q/k/v.",
    )

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    main(args)
