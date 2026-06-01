"""Run all TreeBench baselines — all methods in parallel.

Usage:
  python src/run_baselines.py                    # all methods, parallel
  python src/run_baselines.py --methods bm25 oracle  # specific methods
  python src/run_baselines.py --limit 50         # first 50 questions only
  python src/run_baselines.py --sequential       # one method at a time
"""

from __future__ import annotations
import sys, argparse, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Flush stdout for real-time output
sys.stdout.reconfigure(line_buffering=True)

from baseline_runner import load_gold, run_method, save_results, load_store

METHODS_REGISTRY = {
    "bm25": "retrieval_baselines:bm25_retrieve",
    "dense_rag": "retrieval_baselines:dense_retrieve",
    "hybrid_rag": "retrieval_baselines:hybrid_retrieve",
    "reranker_rag": "retrieval_baselines:reranker_retrieve",
    "rag_cot": "reasoning_baselines:rag_cot",
    "rag_judge": "reasoning_baselines:rag_judge",
    "tree_traversal": "reasoning_baselines:tree_traversal",
    "oracle": "retrieval_baselines:oracle_retrieve",
}

# Group methods by resource type for parallel scheduling:
# Phase 1: no-API methods (can run truly parallel, CPU/memory bound)
# Phase 2: LLM methods (parallel but rate-limited)
PHASE_1 = ["bm25", "oracle", "dense_rag", "hybrid_rag", "reranker_rag"]
PHASE_2 = ["rag_cot", "rag_judge", "tree_traversal"]


def _import_method(spec: str):
    """Import a method function from 'module:function' spec."""
    module_name, fn_name = spec.split(":")
    module = __import__(module_name)
    return getattr(module, fn_name)


def _run_single_method(args):
    """Run one method end-to-end (used by parallel executor)."""
    method_name, questions, limit = args
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    spec = METHODS_REGISTRY[method_name]
    method_fn = _import_method(spec)

    # Pre-load stores
    titles = set(q["source_title"] for q in (questions[:limit] if limit else questions))
    for title in titles:
        try:
            load_store(title)
        except Exception:
            pass

    results = run_method(method_name, method_fn, questions, limit=limit)
    return method_name, results


def main():
    parser = argparse.ArgumentParser(description="Run TreeBench baselines")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--sequential", action="store_true",
                        help="Run methods one at a time instead of parallel")
    args = parser.parse_args()

    questions = load_gold()
    print(f"Loaded {len(questions)} gold questions", flush=True)

    requested = args.methods or (PHASE_1 + PHASE_2)
    requested = [m for m in requested if m in METHODS_REGISTRY]

    all_results = {}
    t_total = time.time()

    if args.sequential:
        for method_name in requested:
            method_fn = _import_method(METHODS_REGISTRY[method_name])
            results = run_method(method_name, method_fn, questions, limit=args.limit)
            all_results[method_name] = results
    else:
        # Pre-load all stores once (shared across threads)
        print("Pre-loading tree stores...", flush=True)
        titles = set(q["source_title"] for q in questions)
        for title in sorted(titles):
            try:
                t0 = time.time()
                load_store(title)
                print(f"  Loaded {title} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  WARN: {title}: {e}", flush=True)

        # Pre-build embeddings sequentially (avoids OOM from parallel embedding)
        needs_dense = any(m in requested for m in ["dense_rag", "hybrid_rag", "reranker_rag", "rag_cot", "rag_judge"])
        if needs_dense:
            print("\nPre-building embeddings (sequential, one title at a time)...", flush=True)
            from retrieval_baselines import _get_dense_index
            for title in sorted(titles):
                try:
                    store = load_store(title)
                    t0 = time.time()
                    _get_dense_index(store)
                    print(f"  Indexed {title} ({time.time()-t0:.1f}s)", flush=True)
                except Exception as e:
                    print(f"  WARN embedding {title}: {e}", flush=True)

        # Phase 1: non-LLM methods in parallel threads
        phase1 = [m for m in requested if m in PHASE_1]
        phase2 = [m for m in requested if m in PHASE_2]

        if phase1:
            print(f"\n>>> PHASE 1: {len(phase1)} retrieval methods in parallel", flush=True)
            with ThreadPoolExecutor(max_workers=len(phase1)) as pool:
                futures = {
                    pool.submit(_run_single_method, (m, questions, args.limit)): m
                    for m in phase1
                }
                for future in futures:
                    method_name, results = future.result()
                    all_results[method_name] = results

        if phase2:
            print(f"\n>>> PHASE 2: {len(phase2)} LLM methods in parallel", flush=True)
            with ThreadPoolExecutor(max_workers=len(phase2)) as pool:
                futures = {
                    pool.submit(_run_single_method, (m, questions, args.limit)): m
                    for m in phase2
                }
                for future in futures:
                    method_name, results = future.result()
                    all_results[method_name] = results

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    save_results(all_results, questions)


if __name__ == "__main__":
    main()
