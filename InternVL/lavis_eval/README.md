# VLMEvalKit-style VQA evaluation for InternVL

Self-contained module that reproduces the **VLMEvalKit** evaluation
pipeline (VQAv2, OK-VQA, GQA) for **InternVL 3.5 1B** — baseline, pruned,
or finetuned variants. The dataset JSONs are read from the LAVIS data
folder (same annotations BLIP-2 was evaluated on), so the comparison
against the LAVIS BLIP-2 numbers is apples-to-apples.

## Evaluation protocol (matches VLMEvalKit)

| Aspect | Behaviour |
|---|---|
| Prompt | `<image>\n{question}\nAnswer the question using a single word or phrase.` — the InternVL VQA prompt used in VLMEvalKit (`--no_short_hint` falls back to the bare prompt). |
| Image preprocessing | InternVL dynamic tiling (`max_num=12`, `448` per tile, ImageNet normalization). |
| Generation | `num_beams=5`, `max_new_tokens=10`, greedy. |
| Normalization | `processPunctuation` + `processDigitArticle` from the official `VQAEval`. Imported directly from `LAVIS/lavis/common/vqa_tools/vqa_eval.py` (no vendored copy). |
| VQAv2 / OK-VQA score | Official metric: `min(matches/3, 1)` averaged over 10 leave-one-out subsets. **No lemmatization** (matches VLMEvalKit). To reproduce the LAVIS BLIP-2 `apply_lemmatizer=True` setting, flip `TASKS["okvqa"]["apply_lemmatizer"]` to `True` in `run_all.py` or pass `--apply_lemmatizer` to `evaluate.py`. |
| GQA score | Exact match after normalization, single GT. |
| Checkpoint loading | Same logic as `evaluate_internvl_pruned.py` (decouple ViT QKV, reshape MLP/QKV per `pruned_shapes.json`, load `model.pt`). |

## Files

- `vqa_scoring.py` — thin wrapper around LAVIS `VQAEval`.
- `datasets.py` — readers for VQAv2/OK-VQA `vqa_val_eval.json` and GQA `testdev_balanced_questions.json`.
- `generation.py` — InternVL image preprocessing + `model.chat()` wrapper.
- `model_loader.py` — pruned/finetuned-aware InternVL loader.
- `evaluate.py` — single-task / single-checkpoint CLI.
- `run_all.py` — full sweep (7 checkpoints × 3 tasks). Auto-downloads GQA testdev JSON if missing.
- `summarize.py` — print the result table from saved JSONs.

## Run

```bash
cd ~/UKMP/InternVL
conda activate ukmp
python -u lavis_eval/run_all.py
```

Adjust `MAX_SAMPLES` / `NUM_BEAMS` at the top of `run_all.py` if needed.
Per-(task,method) result JSONs land in `eval_results/lavis_vqa/<task>_<method>/`.
Re-run `python -u lavis_eval/summarize.py` any time to reprint the table.
