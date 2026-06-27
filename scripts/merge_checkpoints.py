#!/usr/bin/env python3
"""Merge per-method checkpoint files into a combined baseline_results.json.

Usage:
    python scripts/merge_checkpoints.py

Reads checkpoints from data/results/checkpoints/ and produces
data/results/baseline_results.json with all 8 methods.
"""

import sys
from pathlib import Path

# Add src/ to path so we can import baseline_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
from baseline_runner import (
    _load_checkpoint,
    save_results,
    GOLD_FILE,
    CHECKPOINT_DIR,
)

METHODS = [
    "oracle",
    "bm25",
    "hybrid_rag",
    "dense_rag",
    "reranker_rag",
    "rag_cot",
    "rag_judge",
    "tree_traversal",
]


def main():
    # Load gold questions
    with open(GOLD_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} gold questions from {GOLD_FILE.name}")

    # Load all checkpoints
    all_results: dict[str, list] = {}
    for method in METHODS:
        ckpt = _load_checkpoint(method)
        if not ckpt:
            print(f"  WARNING: no checkpoint for {method}, skipping")
            continue
        results = list(ckpt.values())
        all_results[method] = results
        complete = "COMPLETE" if len(results) == len(questions) else "PARTIAL"
        print(f"  {method}: {len(results)}/{len(questions)} questions ({complete})")

    print(f"\nMerging {len(all_results)} methods...")
    save_results(all_results, questions)
    print("Done.")


if __name__ == "__main__":
    main()
