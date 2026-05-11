"""Pretty-print predictions vs ground truths from a saved LAVIS-VQA result JSON.

Usage:
    python -u lavis_eval/inspect_results.py <path_to_results.json> [--n 20] [--only-wrong] [--only-right]

Examples:
    python -u lavis_eval/inspect_results.py /tmp/lavis_eval_smoke/vqav2_results_smoke.json
    python -u lavis_eval/inspect_results.py eval_results/lavis_vqa/okvqa_baseline/okvqa_results_baseline.json --n 30 --only-wrong
"""
from __future__ import annotations

import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a *_results_*.json file produced by evaluate.py")
    parser.add_argument("--n", type=int, default=20, help="How many rows to print (default 20)")
    parser.add_argument("--only-wrong", action="store_true", help="Only show samples with score < 1")
    parser.add_argument("--only-right", action="store_true", help="Only show samples with score == 1")
    args = parser.parse_args()

    with open(args.path) as f:
        d = json.load(f)

    print("=" * 80)
    print(f"file              : {args.path}")
    print(f"task              : {d.get('task')}")
    print(f"model_name        : {d.get('model_name')}")
    print(f"pruned_ckpt       : {d.get('pruned_ckpt')}")
    print(f"job_id            : {d.get('job_id')}")
    print(f"num_samples       : {d.get('num_samples')}")
    print(f"accuracy          : {d.get('accuracy'):.2f}%")
    print(f"short_hint        : {d.get('short_hint')}")
    print(f"apply_lemmatizer  : {d.get('apply_lemmatizer')}")
    print(f"num_beams         : {d.get('num_beams')}")
    print(f"max_new_tokens    : {d.get('max_new_tokens')}")

    # Prediction-style sanity stats
    results = d["results"]
    pred_lens = [len(r["prediction"].split()) for r in results]
    print()
    print("prediction length (words):"
          f"  min={min(pred_lens)} median={statistics.median(pred_lens):.0f}"
          f" mean={statistics.mean(pred_lens):.1f} max={max(pred_lens)}")
    verbose = sum(1 for L in pred_lens if L > 10)
    print(f"predictions >10 words: {verbose}/{len(results)}"
          f"  (if non-trivial, short_hint may be off or generation isn't truncating)")

    rows = results
    if args.only_wrong:
        rows = [r for r in rows if r["score"] < 1.0]
    elif args.only_right:
        rows = [r for r in rows if r["score"] >= 1.0]

    print()
    print(f"Showing {min(args.n, len(rows))} of {len(rows)} rows:")
    print("-" * 80)
    for r in rows[: args.n]:
        ans = r["answer"]
        if isinstance(ans, list):
            ans_str = "[" + ", ".join(repr(a) for a in ans) + "]"
        else:
            ans_str = repr(ans)
        print(f"qid={r['question_id']}  score={r['score']:.2f}")
        print(f"  Q : {r['question']}")
        print(f"  A : {ans_str}")
        print(f"  P : {r['prediction']!r}")
        if r.get("prediction_for_score") and r["prediction_for_score"] != r["prediction"]:
            print(f"  P*: {r['prediction_for_score']!r}    (after lemmatizer)")
        print()


if __name__ == "__main__":
    main()
