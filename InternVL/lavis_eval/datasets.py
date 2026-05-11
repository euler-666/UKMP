"""Dataset wrappers that read the LAVIS-format VQA annotation JSONs."""
import json
import os
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset


class _BaseVQADataset(Dataset):
    """Shared image-loading logic."""

    def __init__(self, image_root: str, max_samples: Optional[int] = None):
        self.image_root = image_root
        self.max_samples = max_samples
        self.records: List[dict] = []

    def _truncate(self):
        if self.max_samples is not None:
            self.records = self.records[: self.max_samples]

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (448, 448))


class VQAv2EvalDataset(_BaseVQADataset):
    """Reads LAVIS coco_vqa / vqa_val_eval.json (also works for OK-VQA val).

    Each row has fields: ``question_id``, ``question``, ``answer`` (list of 10),
    ``image`` (e.g. ``val2014/COCO_val2014_000000262148.jpg``), ``dataset``.
    """

    def __init__(self, ann_path: str, image_root: str, max_samples: Optional[int] = None):
        super().__init__(image_root, max_samples)
        with open(ann_path, "r") as f:
            self.records = json.load(f)
        self._truncate()

    def __getitem__(self, idx):
        item = self.records[idx]
        img_path = os.path.join(self.image_root, item["image"])
        return {
            "image": self._load_image(img_path),
            "question": item["question"],
            "question_id": item["question_id"],
            "answers": list(item["answer"]),  # 10 GT answers
            "image_path": img_path,
        }


class GQAEvalDataset(_BaseVQADataset):
    """Reads GQA testdev_balanced_questions.json.

    Two schemas are supported:

    1. **LAVIS-hosted** (list of dicts, e.g. the file at
       ``storage.googleapis.com/.../LAVIS/datasets/gqa/testdev_balanced_questions.json``):
       each entry has fields ``image`` (filename like ``"n161313.jpg"``),
       ``question_id``, ``question``, ``answer``.

    2. **Original GQA** (dict ``qid -> {imageId, question, answer, ...}``):
       ``imageId`` is the image stem (e.g. ``"2354795"`` or ``"n12345"``).

    The reader tries the LAVIS ``image`` field first and falls back to
    ``imageId`` with extensions / ``n``-prefix variants.
    """

    def __init__(self, ann_path: str, image_root: str, max_samples: Optional[int] = None):
        super().__init__(image_root, max_samples)
        with open(ann_path, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            self.records = [
                {"question_id": qid, **entry} for qid, entry in raw.items()
            ]
        else:
            self.records = list(raw)
        self._truncate()

    def _resolve_image_path(self, item):
        # Schema 1: LAVIS-style, item["image"] = filename
        img = item.get("image")
        if isinstance(img, str):
            stem = img.rsplit("/", 1)[-1]
            cand = os.path.join(self.image_root, stem)
            if os.path.exists(cand):
                return cand
            # Fall through to try the imageId-style lookup as a backup.

        # Schema 2: official GQA, item["imageId"] = stem
        image_id = item.get("imageId")
        if image_id is not None:
            image_id = str(image_id)
            for candidate in (f"{image_id}.jpg", f"n{image_id}.jpg"):
                full = os.path.join(self.image_root, candidate)
                if os.path.exists(full):
                    return full
            return os.path.join(self.image_root, f"{image_id}.jpg")

        # Last resort: best-effort using whatever ``image`` had.
        if isinstance(img, str):
            return os.path.join(self.image_root, img.rsplit("/", 1)[-1])
        raise KeyError(
            f"GQA entry has neither 'image' nor 'imageId'; keys={list(item.keys())}"
        )

    def __getitem__(self, idx):
        item = self.records[idx]
        img_path = self._resolve_image_path(item)
        return {
            "image": self._load_image(img_path),
            "question": item["question"],
            "question_id": str(item["question_id"]),
            "answer": item["answer"],
            "image_path": img_path,
        }
