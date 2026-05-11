"""Print a summary table from existing LAVIS-VQA result JSONs.

Usage:
    cd ~/UKMP/InternVL
    python -u lavis_eval/summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

from run_all import METHODS, OUTPUT_ROOT, TASKS, _result_path  # noqa: F401


def main():
    print("=" * 78)
    print("LAVIS-VQA SUMMARY")
    print("=" * 78)
    header = f"{'Method':<22} " + " ".join(f"{t.upper():>9}" for t in TASKS)
    print(header)
    print("-" * len(header))
    for method_id, _ in METHODS:
        cells = []
        for task in TASKS:
            _, path = _result_path(task, method_id)
            if path.exists():
                try:
                    data = json.load(open(path))
                    n = data.get("num_samples", "?")
                    cells.append(f"{data['accuracy']:>7.2f}% ")
                except Exception:
                    cells.append("   ERR    ")
            else:
                cells.append("    -     ")
        print(f"{method_id:<22} " + " ".join(cells))
    print("=" * 78)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
