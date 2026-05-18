"""FineVision SFT streaming dataset for InternVL recovery fine-tuning.

Port of `Qwen3.5/pruning/phase3_vlm/finevision_dataset.py` (Jonghwa Yim) to
InternVL's sample format. Same quality filter (thr0), same per-shard
streaming with `(rank * n_workers + worker_id)`-strided sharding, same drop
conditions. The OUTPUT format is `{pixel_values, question, answer}` so it
plugs directly into the existing `build_training_batch` in
`ukmp_finetune_internvl.py` (no changes to the trainer's batch path).

Differences from Jonghwa's loader (intentional, documented here so future
readers don't have to diff):

- **Single-turn only.** For each FineVision row we take the first
  (user, assistant) turn and ignore the rest. FineVision's per-image
  multi-turn supervision is sacrificed for simplicity; most rated subsets
  are single-turn anyway, and reusing the existing `build_training_batch`
  saves ~150 lines of brittle label-masking code. Multi-turn is a future
  extension if recovery numbers warrant the cost.
- **No HF processor.** Vision side uses InternVL's tile-based
  `dynamic_preprocess` (imported from `ukmp_finetune_internvl`) instead of
  Qwen3.5's `AutoProcessor`. Tiles are 448x448 each, up to `max_num_tiles`
  per sample. The trainer's `build_training_batch` will inject
  `<img>...<IMG_CONTEXT> * (256 * n_tiles)...</img>` markers at the right
  spot in the prompt and produce `image_flags` of all-ones.
- **No `mm_token_type_ids` / `image_grid_thw`.** Those are Qwen3.5 M-RoPE
  specifics; InternVL uses `image_flags`, which `build_training_batch`
  generates from `pixel_values.shape[0]`.
- **Configurable subset filter.** Jonghwa hardcodes a 16-name text-only
  exclusion list; we keep that same list (with the same source comment).

Penguin-Recap-I tail
--------------------
`PenguinIterableDataset` and `MixedSFTIterableDataset` mirror Jonghwa's
counterparts but yield the same single-turn `{pixel_values, question,
answer}` schema. Penguin is single-turn caption data (`{"from": "human",
"value": "...<image>..."}, {"from": "gpt", "value": "<caption>"}`), so
single-turn here costs nothing.

Drop conditions (silent) - same as Jonghwa's:
    - record's image count != 1 (multi-image not in scope)
    - image bytes corrupt
    - any rating's min-over-turns below threshold
    - empty assistant text on the first turn
    - PIL conversion / dynamic_preprocess failure
"""

from __future__ import annotations

import glob
import json
import os
import random
from io import BytesIO
from typing import Iterator, Optional

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from ukmp_finetune_internvl import build_transform, dynamic_preprocess


FINEVISION_ROOT = "/home/pengfei/datasets/FineVision"

# Subsets without rating columns. Same list as Jonghwa's loader
# (`phase3_vlm/finevision_dataset.py::TEXT_ONLY_SUBSETS`); confirmed by the
# quality-histogram run (`phase3_vlm/results/finevision_quality_hist.json`
# :: `no_rating_subsets`). Out of scope for VLM SFT - would skew supervision
# toward LM-only signal.
TEXT_ONLY_SUBSETS = frozenset(
    {
        "text_OpenMathInstruct-2",
        "text_code_feedback",
        "text_codefeedback_filtered_instruction",
        "text_infinitymath",
        "text_mathinstruct",
        "text_mathqa",
        "text_mathstepdpo10k",
        "text_numinamath_cot",
        "text_openhermes_2_5",
        "text_openorca",
        "text_orcamath",
        "text_pythoncode25k",
        "text_pythoncodealpaca",
        "text_ruozhiba",
        "text_theoremqa",
        "text_wizardlm_evol",
    }
)

RATING_COLS = (
    "visual_dependency_ratings",
    "image_correspondence_ratings",
    "relevance_ratings",
    "formatting_ratings",
)


# ---------------------------------------------------------------------------
# FineVision
# ---------------------------------------------------------------------------


def list_finevision_shards(root: str = FINEVISION_ROOT) -> list[tuple[str, str]]:
    """Return `[(subset_name, parquet_path), ...]` for all qualifying shards.

    Excludes the 16 text-only subsets and any subset with no
    `train-*.parquet`. Stable order: sorted by subset, then by shard filename.
    """
    if not os.path.isdir(root):
        raise RuntimeError(f"FineVision root not found: {root}")
    out: list[tuple[str, str]] = []
    for subset in sorted(os.listdir(root)):
        if subset in TEXT_ONLY_SUBSETS:
            continue
        d = os.path.join(root, subset)
        if not os.path.isdir(d):
            continue
        shards = sorted(glob.glob(os.path.join(d, "train-*.parquet")))
        for s in shards:
            out.append((subset, s))
    if not out:
        raise RuntimeError(f"no FineVision parquet shards under {root}")
    return out


def _min_over_turns(ratings) -> int:
    """Min across per-turn ratings; 0 sentinel on missing/empty/bad."""
    if ratings is None:
        return 0
    try:
        if len(ratings) == 0:
            return 0
        return int(min(int(x) for x in ratings))
    except Exception:
        return 0


def _passes_quality(row, vd_min: int, ic_min: int, rl_min: int, fmt_min: int) -> bool:
    return (
        _min_over_turns(row.get("visual_dependency_ratings")) >= vd_min
        and _min_over_turns(row.get("image_correspondence_ratings")) >= ic_min
        and _min_over_turns(row.get("relevance_ratings")) >= rl_min
        and _min_over_turns(row.get("formatting_ratings")) >= fmt_min
    )


def _decode_single_image(images_field) -> Optional[Image.Image]:
    """FineVision stores images as a list of `{bytes: <raw>}`. Returns the
    single PIL image or None if missing / corrupt / multi-image."""
    if images_field is None:
        return None
    try:
        n = len(images_field)
    except TypeError:
        return None
    if n != 1:
        return None
    item = images_field[0]
    raw = item.get("bytes") if isinstance(item, dict) else item
    if not raw:
        return None
    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        return None


class FineVisionIterableDataset(IterableDataset):
    """Streaming FineVision SFT loader for InternVL (single-turn).

    Each yielded sample is:

        pixel_values:  [N_tiles, 3, 448, 448]  bf16/fp32 (un-typed; trainer
                                               casts to model dtype)
        question:      str   (first user turn, with <image> tags stripped)
        answer:        str   (first assistant turn)

    which is the exact schema `FinetuneDataset.__getitem__` returns in
    `ukmp_finetune_internvl.py`. So `build_training_batch` consumes us as-is.

    Worker/rank striding mirrors Jonghwa's: each
    `global_id = rank * n_workers + worker_id` takes
    `shards[global_id::total_strides]`. Per-worker seeded shuffle each epoch.
    On worker exhaustion we restart the epoch loop (the trainer drives
    termination on `samples_seen`).
    """

    def __init__(
        self,
        root: str = FINEVISION_ROOT,
        image_size: int = 448,
        max_num_tiles: int = 6,
        base_seed: int = 1234,
        vd_min: int = 4,
        ic_min: int = 4,
        rl_min: int = 3,
        fmt_min: int = 2,
        rg_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.base_seed = base_seed
        self.vd_min = vd_min
        self.ic_min = ic_min
        self.rl_min = rl_min
        self.fmt_min = fmt_min
        self.rg_batch_size = rg_batch_size
        # Build once; pickle-safe so DataLoader workers inherit it cheaply.
        self._transform = build_transform(image_size)
        # Pre-glob in main proc - fail fast on missing data.
        self.shards = list_finevision_shards(root)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        n_workers = info.num_workers if info is not None else 1
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        global_id = rank * n_workers + worker_id
        total_strides = world * n_workers
        my_shards = self.shards[global_id::total_strides]
        if not my_shards:
            raise RuntimeError(
                f"rank {rank} worker {worker_id} has no shards "
                f"(n_shards={len(self.shards)}, total_strides={total_strides})"
            )

        epoch = 0
        epoch_yields = 0
        while True:
            order = list(my_shards)
            rng = random.Random(self.base_seed + 1000 * global_id + epoch * 31337)
            rng.shuffle(order)
            for subset_name, shard_path in order:
                try:
                    pf = pq.ParquetFile(shard_path)
                except Exception as e:
                    print(
                        f"[finevision] rank {rank} worker {worker_id} "
                        f"open fail {shard_path}: {e}",
                        flush=True,
                    )
                    continue

                try:
                    batch_iter = pf.iter_batches(
                        batch_size=self.rg_batch_size, columns=None
                    )
                except Exception as e:
                    print(
                        f"[finevision] rank {rank} worker {worker_id} "
                        f"iter_batches fail {shard_path}: {e}",
                        flush=True,
                    )
                    continue

                # Guard the inner loop - pyarrow can raise on row-group
                # decompression (e.g. corrupt snappy); without this a single
                # bad shard takes down the worker and cascades to NCCL
                # timeout. Jonghwa hit this on job 47424; same fix here.
                shard_aborted = False
                while True:
                    try:
                        arrow_batch = next(batch_iter)
                        df = arrow_batch.to_pandas()
                    except StopIteration:
                        break
                    except Exception as e:
                        print(
                            f"[finevision] rank {rank} worker {worker_id} "
                            f"corrupt-batch in {shard_path}: {e}; skipping shard",
                            flush=True,
                        )
                        shard_aborted = True
                        break
                    if any(c not in df.columns for c in RATING_COLS):
                        print(
                            f"[finevision] missing rating cols in {shard_path}; "
                            "skipping batch",
                            flush=True,
                        )
                        continue
                    for _, row in df.iterrows():
                        if not _passes_quality(
                            row, self.vd_min, self.ic_min, self.rl_min, self.fmt_min
                        ):
                            continue
                        sample = self._build_sample(row)
                        if sample is not None:
                            epoch_yields += 1
                            yield sample
                if shard_aborted:
                    continue
            if epoch_yields == 0:
                print(
                    f"[finevision] rank {rank} worker {worker_id} epoch {epoch} "
                    f"yielded 0 samples (n_shards={len(my_shards)}); will retry, "
                    "but this likely means the quality filter is too strict for "
                    "this worker's slice",
                    flush=True,
                )
            epoch += 1
            epoch_yields = 0

    def _build_sample(self, row) -> Optional[dict]:
        image = _decode_single_image(row.get("images"))
        if image is None:
            return None

        texts_field = row.get("texts")
        if texts_field is None:
            return None
        try:
            n_turns = len(texts_field)
        except TypeError:
            return None
        if n_turns == 0:
            return None

        first = texts_field[0]
        if not isinstance(first, dict):
            return None
        user_text = (first.get("user") or "").replace("<image>", "").strip()
        asst_text = (first.get("assistant") or "").strip()
        if not asst_text:
            return None
        if not user_text:
            user_text = "Describe the image."

        try:
            tiles = dynamic_preprocess(
                image,
                image_size=self.image_size,
                use_thumbnail=True,
                max_num=self.max_num_tiles,
            )
            pixel_values = torch.stack([self._transform(t) for t in tiles])
        except Exception:
            return None

        return {
            "pixel_values": pixel_values,
            "question": user_text,
            "answer": asst_text,
        }


# ---------------------------------------------------------------------------
# Penguin tail
# ---------------------------------------------------------------------------


PENGUIN_ROOT = "/home/khashmi/data/qvac-vlm/penguin_recap_i"
PENGUIN_SUBSETS = ("datacomp_coyo_penguin", "sa1b_penguin", "openimages_penguin")
PENGUIN_ANNOTATION_GLOB = "processed/llava/train/annotations_*.jsonl"


def list_penguin_annotations(
    root_dir: str = PENGUIN_ROOT, subsets: tuple[str, ...] = PENGUIN_SUBSETS
) -> list[tuple[str, str]]:
    """Return `[(subset_name, abs_jsonl_path), ...]` across the requested
    Penguin subsets. Missing subsets are skipped with a stderr note (same
    behavior as Jonghwa's loader) - only an all-empty result is fatal."""
    files: list[tuple[str, str]] = []
    for subset in subsets:
        d = os.path.join(root_dir, subset)
        if not os.path.isdir(d):
            print(f"[penguin] subset missing on disk, skipping: {d}", flush=True)
            continue
        pattern = os.path.join(d, PENGUIN_ANNOTATION_GLOB)
        sub_files = sorted(glob.glob(pattern))
        if not sub_files:
            print(f"[penguin] no annotations under {pattern}", flush=True)
            continue
        for f in sub_files:
            files.append((subset, f))
    if not files:
        raise RuntimeError(
            f"no Penguin annotations found under {root_dir} "
            f"(subsets={list(subsets)}, glob={PENGUIN_ANNOTATION_GLOB!r})"
        )
    return files


class PenguinIterableDataset(IterableDataset):
    """Streaming Penguin-Recap-I caption-style loader for InternVL.

    Per-record LLaVA schema:
        {"id": str,
         "image": ["images/<shard>/<id>.jpg"],
         "conversations": [{"from": "human", "value": "...<image>..."},
                           {"from": "gpt",   "value": "<long-form caption>"}]}

    Image paths are relative to `<root>/<subset>/processed/`. Same worker/
    rank striding as the FineVision loader; same epoch-restart on
    exhaustion.

    Single-turn by construction - Penguin is caption data with exactly one
    user/assistant pair.
    """

    def __init__(
        self,
        root_dir: str = PENGUIN_ROOT,
        subsets: tuple[str, ...] = PENGUIN_SUBSETS,
        image_size: int = 448,
        max_num_tiles: int = 6,
        base_seed: int = 1234,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.subsets = subsets
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.base_seed = base_seed
        self._transform = build_transform(image_size)
        self.files = list_penguin_annotations(root_dir, subsets)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        n_workers = info.num_workers if info is not None else 1
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        global_id = rank * n_workers + worker_id
        total_strides = world * n_workers
        my_files = self.files[global_id::total_strides]
        if not my_files:
            raise RuntimeError(
                f"rank {rank} worker {worker_id} has no penguin annotation files "
                f"(n_files={len(self.files)}, total_strides={total_strides})"
            )

        epoch = 0
        while True:
            order = list(my_files)
            rng = random.Random(self.base_seed + 7919 * global_id + epoch * 13)
            rng.shuffle(order)
            for subset_name, jsonl_path in order:
                subset_processed = os.path.join(
                    self.root_dir, subset_name, "processed"
                )
                try:
                    fh = open(jsonl_path, "r")
                except Exception as e:
                    print(
                        f"[penguin] rank {rank} worker {worker_id} "
                        f"open fail {jsonl_path}: {e}",
                        flush=True,
                    )
                    continue
                with fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        sample = self._build_sample(rec, subset_processed)
                        if sample is not None:
                            yield sample
            epoch += 1

    def _build_sample(self, rec: dict, subset_processed: str) -> Optional[dict]:
        img_field = rec.get("image")
        if isinstance(img_field, list):
            if not img_field:
                return None
            img_rel = img_field[0]
        elif isinstance(img_field, str):
            img_rel = img_field
        else:
            return None
        img_path = os.path.join(subset_processed, img_rel)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        convs = rec.get("conversations") or []
        user_text = ""
        asst_text = ""
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            val = (turn.get("value", turn.get("content", "")) or "")
            val = val.replace("<image>", "").strip()
            if role in ("human", "user") and not user_text:
                user_text = val
            elif role in ("gpt", "assistant") and not asst_text:
                asst_text = val
        if not asst_text:
            return None
        if not user_text:
            user_text = "Describe the image."

        try:
            tiles = dynamic_preprocess(
                image,
                image_size=self.image_size,
                use_thumbnail=True,
                max_num=self.max_num_tiles,
            )
            pixel_values = torch.stack([self._transform(t) for t in tiles])
        except Exception:
            return None

        return {
            "pixel_values": pixel_values,
            "question": user_text,
            "answer": asst_text,
        }


# ---------------------------------------------------------------------------
# Mixer (FineVision primary + Penguin tail @ tail_prob)
# ---------------------------------------------------------------------------


class MixedSFTIterableDataset(IterableDataset):
    """Bernoulli mixer over two infinite iterables.

    Each draw picks `tail` with probability `tail_prob`, else `primary`.
    Both component iterables are expected to restart epochs internally; if
    either raises StopIteration we restart it locally as a safety net.
    """

    def __init__(
        self,
        primary: IterableDataset,
        tail: IterableDataset,
        tail_prob: float = 0.1,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if not 0.0 <= tail_prob <= 1.0:
            raise ValueError(f"tail_prob must be in [0, 1], got {tail_prob}")
        self.primary = primary
        self.tail = tail
        self.tail_prob = tail_prob
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        rng = random.Random(self.seed + 31 * rank + worker_id)
        it_primary = iter(self.primary)
        it_tail = iter(self.tail)
        while True:
            if rng.random() < self.tail_prob:
                try:
                    yield next(it_tail)
                except StopIteration:
                    it_tail = iter(self.tail)
                    yield next(it_tail)
            else:
                try:
                    yield next(it_primary)
                except StopIteration:
                    it_primary = iter(self.primary)
                    yield next(it_primary)


def identity_collate(batch: list[dict]) -> list[dict]:
    """No-op collate: hand the list of sample dicts to `build_training_batch`
    verbatim, which already does the right thing (variable tile counts +
    right-padded text)."""
    return batch
