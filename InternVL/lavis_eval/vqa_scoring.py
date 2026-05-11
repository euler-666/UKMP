"""LAVIS-compatible VQA scoring.

We reuse the official ``VQAEval`` class shipped with LAVIS rather than
re-vendoring its (very large) normalization tables. The class is fully
self-contained (it only imports ``sys`` and ``re``), so we can load it
by injecting the LAVIS path into ``sys.path``.

Exposed helpers:
    - normalize_answer(text)
    - vqa_score(pred, gt_answers)    # VQAv2 / OK-VQA: 10-GT-answer average
    - exact_match(pred, gt)          # GQA: 1.0 / 0.0 after normalization
    - Lemmatizer()                   # spaCy lemmatizer used for OK-VQA
"""
import importlib
import importlib.util
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_LAVIS_VQA_EVAL = (
    _THIS_DIR.parent.parent / "LAVIS" / "lavis" / "common" / "vqa_tools" / "vqa_eval.py"
)


def _load_lavis_vqaeval():
    """Load the LAVIS VQAEval class directly from its source file."""
    if not _LAVIS_VQA_EVAL.exists():
        raise FileNotFoundError(
            f"Could not find LAVIS vqa_eval.py at {_LAVIS_VQA_EVAL}. "
            "Adjust the path in vqa_scoring.py if your LAVIS lives elsewhere."
        )
    spec = importlib.util.spec_from_file_location(
        "_lavis_vqa_eval_vendor", str(_LAVIS_VQA_EVAL)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VQAEval


VQAEval = _load_lavis_vqaeval()
_evaluator = VQAEval()  # instance used purely for its normalization tables / methods


def normalize_answer(text):
    """Apply VQAEval's punctuation + digit/article normalization."""
    if text is None:
        return ""
    out = str(text).replace("\n", " ").replace("\t", " ").strip()
    out = _evaluator.processPunctuation(out)
    out = _evaluator.processDigitArticle(out)
    return out


def vqa_score(pred, gt_answers):
    """Standard VQAv2 / OK-VQA accuracy for one sample.

    For each of the 10 ground-truth answers, hold it out and check how
    many of the remaining 9 match the (normalized) prediction. Per-leave-one-out
    score = min(matches / 3, 1). The sample score is the average of those 10.

    Returns a float in [0, 1].
    """
    pred_norm = normalize_answer(pred)
    gts_norm = [normalize_answer(a) for a in gt_answers]
    if not gts_norm:
        return 0.0
    accs = []
    for i in range(len(gts_norm)):
        others = gts_norm[:i] + gts_norm[i + 1 :]
        matching = sum(1 for a in others if a == pred_norm)
        accs.append(min(1.0, matching / 3.0))
    return sum(accs) / len(accs)


def exact_match(pred, gt):
    """GQA-style single-answer exact match after VQAEval normalization."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


class Lemmatizer:
    """spaCy lemmatizer used by LAVIS for OK-VQA (apply_lemmatizer=True)."""

    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, text):
        doc = self.nlp(str(text))
        words = []
        for tok in doc:
            if tok.pos_ in ("NOUN", "VERB"):
                words.append(tok.lemma_)
            else:
                words.append(tok.text)
        return " ".join(words)
