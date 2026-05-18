"""Recovery / SFT fine-tune of InternVL3.5-1B on FineVision.

Port of `Qwen3.5/pruning/phase3_vlm/vlm_sft.py` (Jonghwa Yim) to InternVL.
Same recipe philosophy: ViT frozen, projector + LM body trained at decoupled
learning rates, embed/lm_head frozen by default. Plain CE on assistant
tokens computed in fp32; bf16 weights/forward.

InternVL-specific mapping
-------------------------
| Qwen3.5 (Jonghwa)           | InternVL3.5-1B (here)
| `model.visual.*` ViT          | `vision_model.*`            (frozen)
| `model.visual.merger.*`       | `mlp1.*`                    (merger group)
| `model.language_model.*` body | `language_model.model.layers.*`
|                               | + `language_model.model.norm.*`  (body group)
| `embed_tokens` (tied w/ head) | `language_model.model.embed_tokens.*`
|                               | AND `language_model.lm_head.*`  (embed group;
|                                                                 untied here,
|                                                                 frozen
|                                                                 together by
|                                                                 default for
|                                                                 symmetry
|                                                                 with the
|                                                                 Qwen recipe)

Key engineering simplifications vs Jonghwa's script
---------------------------------------------------
- **DDP wrap** (with `find_unused_parameters=True`) instead of manual flat
  all-reduce. The user's existing `finetune_internvl_full_ce.py` already
  uses DDP successfully for this model size and the Qwen3.5-specific
  cuDNN regression in `Qwen3_5VisionPatchEmbed.Conv3d` doesn't apply to
  InternVL (InternViT uses Conv2d patch embedding).
- **No `mm_token_type_ids`**. InternVL identifies image positions via
  `image_flags` + `<IMG_CONTEXT>` placeholder tokens, both produced by the
  existing `build_training_batch` helper.
- **Gradient checkpointing disabled** on both ViT and LLM. InternVL's
  `modeling_internvl_chat.py` uses `torch.utils.checkpoint(..., use_reentrant=True)`
  which collides with DDP autograd hooks (same bug the user already fixed
  in `finetune_internvl_full_ce.py`).

Saved checkpoint is a standard HF directory (safetensors), directly
loadable by the OCRBench eval pipeline via `--model_name <save_dir>`.

Launch (multi-GPU DDP):

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      finetune_internvl_finevision.py \
      --job_id internvl1b-finevision-cell5 \
      --body_lr 3e-5 --tail_prob 0.1 \
      --max_samples 100000 --micro_batch 1 --grad_accum 32
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from finevision_dataset_internvl import (
    FineVisionIterableDataset,
    MixedSFTIterableDataset,
    PenguinIterableDataset,
    identity_collate,
)
from internvl_lib.common.logger import LoggerWithDepth
from ukmp_finetune_internvl import (
    build_training_batch,
    get_rank,
    get_world_size,
    is_dist,
    is_main_process,
    setup_distributed,
    setup_seeds,
)


IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Safety net: ensure NCCL process group is destroyed on any exit path.
# Borrowed from Jonghwa's vlm_sft.py - sidesteps the CG-zombie pattern when
# training raises (OOM, NaN, ckpt failure, etc.).
# ---------------------------------------------------------------------------


def _destroy_dist_on_exit() -> None:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


atexit.register(_destroy_dist_on_exit)


# ---------------------------------------------------------------------------
# Param groups (InternVL-mapped)
# ---------------------------------------------------------------------------


def setup_param_groups(
    model: nn.Module,
    body_lr: float,
    merger_lr: float,
    embed_lr: float,
    unfreeze_embed: bool,
    weight_decay: float,
) -> tuple[list[dict], dict]:
    """Freeze ViT and (by default) embed_tokens + lm_head; build per-group
    optimizer entries for merger / body (+ embed when unfrozen).

    InternVL3.5 module naming (verified):
        vision_model.*                          - InternViT (frozen)
        mlp1.*                                  - merger (trainable)
        language_model.model.embed_tokens.*     - embed (frozen by default)
        language_model.lm_head.*                - lm_head (frozen by default)
        language_model.model.layers.*           - LM body (trainable)
        language_model.model.norm.*             - final norm (trainable, body)

    InternVL3.5-1B has `tie_word_embeddings=False` so embed_tokens and
    lm_head are SEPARATE tensors. We freeze both together by default and
    unfreeze both together under `--unfreeze_embed` - symmetric with
    Jonghwa's Qwen3.5 recipe where the tied pair is treated as one knob.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    merger_params: list[torch.nn.Parameter] = []
    body_params: list[torch.nn.Parameter] = []
    embed_params: list[torch.nn.Parameter] = []
    vit_frozen_tensors = 0

    for name, param in model.named_parameters():
        if name.startswith("vision_model."):
            vit_frozen_tensors += 1
        elif name.startswith("mlp1."):
            param.requires_grad_(True)
            merger_params.append(param)
        elif name.startswith("language_model.model.embed_tokens.") or name.startswith(
            "language_model.lm_head."
        ):
            if unfreeze_embed:
                param.requires_grad_(True)
                embed_params.append(param)
        elif name.startswith("language_model."):
            param.requires_grad_(True)
            body_params.append(param)
        else:
            # Defensive: unknown top-level. Leave frozen and surface in logs.
            pass

    groups: list[dict] = [
        {"params": merger_params, "lr": merger_lr, "weight_decay": weight_decay,
         "name": "merger"},
        {"params": body_params, "lr": body_lr, "weight_decay": weight_decay,
         "name": "body"},
    ]
    if unfreeze_embed and embed_params:
        groups.append({"params": embed_params, "lr": embed_lr,
                       "weight_decay": weight_decay, "name": "embed"})

    counts = {
        "merger": sum(p.numel() for p in merger_params),
        "body": sum(p.numel() for p in body_params),
        "embed": sum(p.numel() for p in embed_params),
        "vit_frozen_tensors": vit_frozen_tensors,
    }
    return groups, counts


# ---------------------------------------------------------------------------
# Loss: bf16 forward + fp32 CE on assistant tokens
# ---------------------------------------------------------------------------


def sft_forward(
    model: nn.Module, inputs: dict, ignore_index: int = IGNORE_INDEX
) -> tuple[torch.Tensor, dict]:
    """One forward pass + CE in fp32. Labels are pre-masked with -100 for
    non-assistant positions by `build_training_batch`.

    Returns (loss, stats). `stats["n_tokens"] == 0` means the batch had no
    labeled tokens (degenerate) and the caller should skip the backward
    step to avoid a zero-grad-but-with-graph optimizer update.
    """
    outputs = model(**inputs)
    logits = outputs.logits  # [B, T, V] bf16

    V = logits.size(-1)
    shift_logits = logits[:, :-1, :].reshape(-1, V).float()  # fp32 CE
    shift_labels = inputs["labels"][:, 1:].reshape(-1)
    valid = shift_labels != ignore_index
    n_tokens = int(valid.sum())
    if n_tokens == 0:
        zero = logits.sum() * 0.0
        return zero, {"nll": 0.0, "n_tokens": 0}
    nll = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="mean")
    return nll, {"nll": float(nll.detach()), "n_tokens": n_tokens}


# ---------------------------------------------------------------------------
# Train state + SIGTERM-safe checkpoint with `latest` symlink
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    samples_seen: int = 0
    best_eval_nll: float = float("inf")
    eval_history: list = field(default_factory=list)


def save_checkpoint(
    out_dir: str,
    tag: str,
    model: nn.Module,
    tokenizer,
    optimizer,
    scheduler,
    state: TrainState,
    args,
) -> str:
    """Save the full VLM (safetensors) + tokenizer + train state. Atomically
    swing the `latest` symlink so resume always sees a complete dir."""
    ckpt_dir = os.path.join(out_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    inner = model.module if hasattr(model, "module") else model
    inner.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)

    train_blob = {
        "step": state.step,
        "samples_seen": state.samples_seen,
        "best_eval_nll": state.best_eval_nll,
        "eval_history": state.eval_history,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "args": vars(args),
    }
    torch.save(train_blob, os.path.join(ckpt_dir, "train_state.pt"))

    latest = os.path.join(out_dir, "latest")
    tmp = latest + ".tmp"
    if os.path.islink(tmp) or os.path.exists(tmp):
        os.unlink(tmp)
    os.symlink(tag, tmp)
    os.replace(tmp, latest)
    return ckpt_dir


def maybe_load_train_state(out_dir: str, optimizer, scheduler, state: TrainState) -> bool:
    latest = os.path.join(out_dir, "latest")
    if not (os.path.islink(latest) or os.path.isdir(latest)):
        return False
    target = os.path.realpath(latest)
    blob_path = os.path.join(target, "train_state.pt")
    if not os.path.isfile(blob_path):
        return False
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    state.step = blob["step"]
    state.samples_seen = blob["samples_seen"]
    state.best_eval_nll = blob.get("best_eval_nll", float("inf"))
    state.eval_history = blob.get("eval_history", [])
    rng = blob.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch_cpu" in rng:
        torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return True


def resolve_resume_path(out_dir: str, default_model: str) -> str:
    latest = os.path.join(out_dir, "latest")
    if os.path.islink(latest) or os.path.isdir(latest):
        target = os.path.realpath(latest)
        if os.path.isfile(os.path.join(target, "train_state.pt")):
            return target
    return default_model


# ---------------------------------------------------------------------------
# Held-out NLL probe (the regression detector; OCRBench is the real eval)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_nll(
    model: nn.Module, eval_batches: list[dict], tokenizer, device: str, base_model
) -> float:
    """Mean assistant-token NLL on a per-rank held-out batch list. Distributed
    via all-reduce on (sum_nll, n_tokens). The eval pool is built once before
    training and stable within a run."""
    inner = model.module if hasattr(model, "module") else model
    was_training = inner.training
    inner.eval()

    local_nll = 0.0
    local_tokens = 0
    for batch_samples in eval_batches:
        inputs = build_training_batch(batch_samples, base_model, tokenizer, device)
        outputs = inner(**inputs)
        logits = outputs.logits
        V = logits.size(-1)
        shift_logits = logits[:, :-1, :].reshape(-1, V).float()
        shift_labels = inputs["labels"][:, 1:].reshape(-1)
        valid = shift_labels != IGNORE_INDEX
        if valid.any():
            nll_sum = F.cross_entropy(
                shift_logits[valid], shift_labels[valid], reduction="sum"
            )
            local_nll += float(nll_sum)
            local_tokens += int(valid.sum())

    if was_training:
        inner.train()

    if is_dist():
        stats = torch.tensor(
            [local_nll, float(local_tokens)], device=device, dtype=torch.float64
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_nll, total_tokens = stats.tolist()
    else:
        total_nll, total_tokens = local_nll, float(local_tokens)
    if total_tokens <= 0:
        return float("nan")
    return total_nll / total_tokens


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, device, dtype=torch.bfloat16) -> nn.Module:
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return model


def _disable_gradient_checkpointing(model: nn.Module) -> None:
    """InternVL's modeling_internvl_chat calls torch.utils.checkpoint with
    use_reentrant=True. Combined with DDP autograd hooks this causes
    "Expected to mark a variable ready only once" because the inner reentrant
    backward fires DDP hooks twice on the same param. The user already had to
    apply this fix in finetune_internvl_full_ce.py; we carry it forward.

    InternVL 1B at mb=1 fits comfortably without gradient checkpointing on
    a 40GB+ GPU.
    """
    def _walk(m):
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass
        for sub in m.modules():
            if hasattr(sub, "gradient_checkpointing"):
                sub.gradient_checkpointing = False

    if hasattr(model, "vision_model"):
        _walk(model.vision_model)
    if hasattr(model, "language_model"):
        _walk(model.language_model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    if args.merger_lr is None:
        args.merger_lr = args.body_lr / 5.0
    if args.embed_lr is None:
        args.embed_lr = args.body_lr / 10.0

    local_rank, device = setup_distributed()
    setup_seeds(args.seed + get_rank())
    rank = get_rank()
    world_size = get_world_size()

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

    out_dir = logger.sub_dir if is_main_process() else "/tmp"
    if is_dist():
        # Broadcast out_dir from rank 0 so all ranks save/resume to same place.
        out_dir_list = [out_dir]
        dist.broadcast_object_list(out_dir_list, src=0)
        out_dir = out_dir_list[0]

    # ------------------------------------------------------------------
    # Tokenizer + model (resume-aware)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resume_dir = (
        resolve_resume_path(out_dir, args.model_name)
        if args.resume == "auto"
        else args.model_name
    )
    logger.log(f"Loading model from {resume_dir} (resume={args.resume})")
    model = load_model(resume_dir, device)

    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    _disable_gradient_checkpointing(model)
    logger.log("Disabled gradient checkpointing on ViT and LLM "
               "(required for DDP + ViT bf16 reentrant backward).")

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    # ------------------------------------------------------------------
    # Param groups (merger / body / embed)
    # ------------------------------------------------------------------
    groups, counts = setup_param_groups(
        model,
        body_lr=args.body_lr,
        merger_lr=args.merger_lr,
        embed_lr=args.embed_lr,
        unfreeze_embed=args.unfreeze_embed,
        weight_decay=args.weight_decay,
    )
    n_trainable = counts["merger"] + counts["body"] + counts["embed"]
    logger.log(
        f"Trainable: merger={counts['merger']:,} ({counts['merger']/1e6:.1f}M)  "
        f"body={counts['body']:,} ({counts['body']/1e6:.1f}M)  "
        f"embed={counts['embed']:,} ({counts['embed']/1e6:.1f}M)  "
        f"total={n_trainable:,} ({n_trainable/1e6:.1f}M)"
    )
    logger.log(f"Frozen ViT param tensors: {counts['vit_frozen_tensors']}")
    logger.log(
        f"LR: merger={args.merger_lr:.2e}  body={args.body_lr:.2e}  "
        f"embed={args.embed_lr:.2e} (unfrozen={args.unfreeze_embed})"
    )

    # Flat trainable list for grad clipping.
    trainable_params: list[torch.nn.Parameter] = []
    for g in groups:
        trainable_params.extend(g["params"])
    if not trainable_params:
        raise RuntimeError("no trainable params - check freeze policy")

    # ------------------------------------------------------------------
    # DDP wrap
    # ------------------------------------------------------------------
    if is_dist():
        # find_unused_parameters=False: every trainable param participates in
        # every forward (DDP confirmed this with the runtime warning when we
        # ran with True). With the zero-grad-backward fix above guaranteeing
        # symmetric backward() calls across ranks, no rank should ever have
        # unused params - it's safe to disable the extra graph traversal.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        logger.log(f"Wrapped model in DDP across {world_size} GPUs "
                   f"(find_unused_parameters=False)")
    base_model = model.module if hasattr(model, "module") else model

    # ------------------------------------------------------------------
    # Optimizer + cosine LR with 2% warmup
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    samples_per_step = world_size * args.micro_batch * args.grad_accum
    total_steps = max(1, math.ceil(args.max_samples / samples_per_step))
    warmup_steps = max(1, int(round(total_steps * args.warmup_ratio)))
    logger.log(
        f"samples_per_step = {samples_per_step} "
        f"(world={world_size} x mb={args.micro_batch} x accum={args.grad_accum})"
    )
    logger.log(
        f"total_steps = {total_steps:,}  warmup_steps = {warmup_steps}  "
        f"max_samples = {int(args.max_samples):,}"
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    save_every = max(1, int(round(total_steps * args.save_frac)))
    eval_every = max(1, int(round(total_steps * args.eval_every_frac)))
    logger.log(f"save_every = {save_every} steps  eval_every = {eval_every} steps")

    # ------------------------------------------------------------------
    # Restore train state (resume)
    # ------------------------------------------------------------------
    state = TrainState()
    if args.resume == "auto" and is_main_process():
        loaded = maybe_load_train_state(out_dir, optimizer, scheduler, state)
        if loaded:
            logger.log(
                f"[resume] loaded train state: step={state.step}, "
                f"samples_seen={state.samples_seen:,}"
            )
    if is_dist():
        # Broadcast restored state from rank 0 so all ranks agree.
        scalars = torch.tensor([state.step, state.samples_seen],
                               device=device, dtype=torch.long)
        dist.broadcast(scalars, src=0)
        state.step = int(scalars[0])
        state.samples_seen = int(scalars[1])

    # ------------------------------------------------------------------
    # Data loader (FineVision primary + optional Penguin tail)
    # ------------------------------------------------------------------
    primary = FineVisionIterableDataset(
        root=args.finevision_root,
        image_size=448,
        max_num_tiles=args.max_num_tiles,
        base_seed=args.seed,
        vd_min=args.vd_min,
        ic_min=args.ic_min,
        rl_min=args.rl_min,
        fmt_min=args.fmt_min,
    )
    if args.tail_prob > 0:
        tail = PenguinIterableDataset(
            root_dir=args.penguin_root,
            image_size=448,
            max_num_tiles=args.max_num_tiles,
            base_seed=args.seed + 9999,
        )
        dataset = MixedSFTIterableDataset(
            primary=primary, tail=tail, tail_prob=args.tail_prob, seed=args.seed
        )
        logger.log(f"Data: FineVision + Penguin tail @ p={args.tail_prob}")
    else:
        dataset = primary
        logger.log("Data: FineVision only (no Penguin tail)")

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = iter(loader)

    # ------------------------------------------------------------------
    # Held-out eval pool (built once, stable within run)
    # ------------------------------------------------------------------
    eval_pool: list[list[dict]] = []
    target_eval_batches = max(1, args.eval_samples // args.micro_batch)
    while len(eval_pool) < target_eval_batches:
        b = next(data_iter)
        if b:
            eval_pool.append(b)
    logger.log(f"Eval pool: {len(eval_pool)} batches per rank")

    # ------------------------------------------------------------------
    # SIGTERM-safe save
    # ------------------------------------------------------------------
    interrupted = {"flag": False}

    def handle_term(signum, _frame):
        if not interrupted["flag"]:
            logger.log(f"[signal] caught {signum}; will checkpoint and exit")
            interrupted["flag"] = True
        else:
            logger.log(f"[signal] caught {signum} again; hard exit")
            sys.exit(130)

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.log("=== training ===")
    t0 = time.time()
    win_loss = 0.0
    win_count = 0
    step_t0 = time.time()
    total_samples = int(args.max_samples)
    bad_loss_count = 0
    zero_token_count = 0

    try:
        while (
            state.samples_seen < total_samples
            and state.step < total_steps
            and not interrupted["flag"]
        ):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            valid_micros = 0

            for _ in range(args.grad_accum):
                batch_samples = next(data_iter)
                if not batch_samples:
                    continue
                inputs = build_training_batch(
                    batch_samples, base_model, tokenizer, device
                )
                # InternViT path bug: with ViT frozen, pixel_values must
                # require_grad so torch.utils.checkpoint doesn't short-
                # circuit the backward (same fix as full_ce). Even with
                # gradient checkpointing disabled this is a no-op cost
                # and keeps the path bulletproof.
                if inputs["pixel_values"].dtype.is_floating_point:
                    inputs["pixel_values"] = (
                        inputs["pixel_values"].detach().requires_grad_(True)
                    )

                loss, stats = sft_forward(model, inputs)
                loss_val = float(loss.detach())

                # DDP-safe handling: NEVER skip backward() on one rank while
                # other ranks might run it. The autograd-hook count must
                # match across ranks or the per-bucket allreduces drift and
                # NCCL hits a 600s watchdog timeout (this is the bug that
                # killed cell 5's first launch ~10 min in - search the
                # trace for "Watchdog caught collective operation timeout").
                #
                # Two degenerate paths converge to "backward zero":
                #  (a) n_tokens == 0   - build_training_batch truncated all
                #                        labels to -100 (likely max_num_tiles
                #                        too high → too few tokens left for
                #                        the answer span). `loss` is already
                #                        `logits.sum() * 0.0`, in-graph.
                #  (b) loss is NaN/inf - rare numerical blowup; zero out via
                #                        `loss * 0.0` to keep the graph
                #                        connected but contribute nothing.
                # Either way we backward through a graph that touches every
                # trainable param, so DDP buckets fire symmetrically.
                bad = math.isnan(loss_val) or math.isinf(loss_val)
                if stats["n_tokens"] == 0 or bad:
                    if bad:
                        bad_loss_count += 1
                    else:
                        zero_token_count += 1
                    (loss * 0.0 / args.grad_accum).backward()
                    del loss
                    continue
                (loss / args.grad_accum).backward()
                accum_loss += loss_val
                valid_micros += 1
                del loss

            local_valid = torch.tensor(
                [valid_micros], device=device, dtype=torch.long
            )
            if is_dist():
                dist.all_reduce(local_valid, op=dist.ReduceOp.SUM)
            global_valid = int(local_valid.item())

            if global_valid == 0:
                scheduler.step()
                state.step += 1
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            state.step += 1
            state.samples_seen += global_valid * args.micro_batch

            if valid_micros > 0:
                win_loss += accum_loss / valid_micros
                win_count += 1

            if state.step % args.log_every == 0 or state.step == 1:
                local_count = max(win_count, 1)
                if is_dist():
                    stats_t = torch.tensor(
                        [win_loss], device=device, dtype=torch.float64
                    )
                    dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
                    avg_loss = float(stats_t[0]) / world_size / local_count
                else:
                    avg_loss = win_loss / local_count

                now = time.time()
                dt = now - step_t0
                samps_per_sec = (
                    (args.log_every * samples_per_step) / max(dt, 1e-3)
                    if state.step > 1
                    else 0.0
                )
                lr_now = [g["lr"] for g in optimizer.param_groups]
                logger.log(
                    f"step {state.step:6d}/{total_steps}  "
                    f"samp {state.samples_seen/1e3:7.1f}K  "
                    f"loss {avg_loss:7.4f}  "
                    f"lr {[f'{x:.2e}' for x in lr_now]}  "
                    f"samp/s {samps_per_sec:6.1f}  "
                    f"elapsed {(now - t0)/60:6.1f}min  "
                    f"zero_tok={zero_token_count} bad={bad_loss_count}"
                )
                win_loss = 0.0
                win_count = 0
                step_t0 = now

            if state.step % eval_every == 0:
                ev_t0 = time.time()
                nll = eval_nll(model, eval_pool, tokenizer, device, base_model)
                if is_main_process():
                    ppl = math.exp(nll) if nll == nll and nll < 50 else float("inf")
                    logger.log(
                        f"[eval] step {state.step}  held-out-nll {nll:.4f}  "
                        f"ppl {ppl:.2f}  ({time.time() - ev_t0:.1f}s)"
                    )
                    state.eval_history.append(
                        {"step": state.step, "nll": nll, "ppl": ppl}
                    )
                    if nll < state.best_eval_nll:
                        state.best_eval_nll = nll

            if args.save_frac > 0 and state.step % save_every == 0:
                if is_main_process():
                    ck = save_checkpoint(
                        out_dir,
                        f"step-{state.step:07d}",
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        state,
                        args,
                    )
                    logger.log(f"[ckpt] saved {ck}")
                if is_dist():
                    dist.barrier()
    finally:
        try:
            del data_iter
        except Exception:
            pass
        try:
            del loader
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final save (always, even on interrupt - rank 0 only)
    # ------------------------------------------------------------------
    if is_main_process():
        tag = (
            "finetuned_model"
            if state.step >= total_steps
            else f"step-{state.step:07d}-interrupted"
        )
        ck = save_checkpoint(
            out_dir, tag, model, tokenizer, optimizer, scheduler, state, args
        )
        logger.log(f"[ckpt] final saved {ck}")
        logger.log(f"total wall = {(time.time() - t0)/60:.1f} min  "
                   f"bad_loss_count={bad_loss_count}")
        # Ensure there's always a `finetuned_model` symlink for the eval
        # script to point at, regardless of whether we hit total_steps.
        finetuned_link = os.path.join(out_dir, "finetuned_model")
        if not os.path.exists(finetuned_link) and tag != "finetuned_model":
            os.symlink(tag, finetuned_link + ".tmp")
            os.replace(finetuned_link + ".tmp", finetuned_link)
        logger.log("[FINISH] - FineVision recovery fine-tune complete")
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
    if interrupted["flag"]:
        sys.exit(130)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="FineVision recovery SFT of InternVL3.5-1B"
    )
    p.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B")
    p.add_argument("--job_id", type=str, default="finevision_recovery")
    p.add_argument("--save_ckpt_log_name", type=str, default="internvl_finevision")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--finevision_root", type=str,
                   default="/home/pengfei/datasets/FineVision")
    p.add_argument("--penguin_root", type=str,
                   default="/home/khashmi/data/qvac-vlm/penguin_recap_i")
    p.add_argument("--tail_prob", type=float, default=0.0,
                   help="Bernoulli mix probability for Penguin tail "
                        "(0.0 = pure FineVision)")

    # FineVision quality filter (thr0 = (4, 4, 3, 2))
    p.add_argument("--vd_min", type=int, default=4)
    p.add_argument("--ic_min", type=int, default=4)
    p.add_argument("--rl_min", type=int, default=3)
    p.add_argument("--fmt_min", type=int, default=2)

    # Budget + batch
    p.add_argument("--max_samples", type=int, default=100_000)
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_num_tiles", type=int, default=6)

    # Per-group LR
    p.add_argument("--body_lr", type=float, default=3e-5)
    p.add_argument("--merger_lr", type=float, default=None,
                   help="default = body_lr / 5")
    p.add_argument("--embed_lr", type=float, default=None,
                   help="default = body_lr / 10 (only used with --unfreeze_embed)")
    p.add_argument("--unfreeze_embed", action="store_true",
                   help="Unfreeze embed_tokens AND lm_head together "
                        "(InternVL has untied head)")

    # Optimizer / schedule
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # System
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_frac", type=float, default=0.0,
                   help="0 = final ckpt only; 0.15 = save 7 mid + final")
    p.add_argument("--eval_every_frac", type=float, default=0.25)
    p.add_argument("--eval_samples", type=int, default=32)
    p.add_argument("--resume", choices=["auto", "no"], default="auto")

    args = p.parse_args()
    main(args)
