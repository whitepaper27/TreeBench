"""TreeBench Baseline Runner — harness + metrics + results logging.

Architecture:
  retrieve(question, store) → list[node_id]
  answer(question, retrieved_nodes, store) → str
  Each method returns (retrieved_ids, answer_text, cost, latency)
  Harness computes all 7 metrics from that.

Metrics per method:
  answer_accuracy, path_accuracy, required_node_recall,
  distractor_hit_rate, abstention_rate, cost_per_query, latency_per_query

Breakdowns: domain, failure_type, difficulty, tree_depth
"""

from __future__ import annotations
import json, time, re, sys, os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_node import TreeNode, TreeStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_gold_v1 = DATA_DIR / "pilot" / "treebench_v1_861_gold.json"
_gold_legacy = DATA_DIR / "pilot" / "treebench_1000_gold.json"
GOLD_FILE = _gold_v1 if _gold_v1.exists() else _gold_legacy
PARSED_DIR = DATA_DIR / "parsed"
RESULTS_DIR = DATA_DIR / "results"

# Map source_title to parsed tree file
TITLE_TO_FILE = {
    "eCFR Title 26 — Internal Revenue": "ECFR-title26_tree.json",
    "eCFR Title 12 — Banks and Banking": "ECFR-title12_tree.json",
    "eCFR Title 17 — Securities": "ECFR-title17_tree.json",
    "eCFR Title 21 — Food and Drugs": "ECFR-title21_tree.json",
    "eCFR Title 42 — Public Health": "ECFR-title42_tree.json",
    "eCFR Title 29 — Labor": "ECFR-title29_tree.json",
    "eCFR Title 15 — Commerce": "ECFR-title15_tree.json",
    "eCFR Title 40 — Environment": "ECFR-title40_tree.json",
    "eCFR Title 45 — HHS/HIPAA": "ECFR-title45_tree.json",
    "eCFR Title 31 — Treasury/AML": "ECFR-title31_tree.json",
}

_store_cache: dict[str, TreeStore] = {}


def load_store(source_title: str) -> TreeStore:
    """Load and cache a parsed tree."""
    if source_title in _store_cache:
        return _store_cache[source_title]
    fname = TITLE_TO_FILE.get(source_title)
    if not fname:
        raise ValueError(f"Unknown source_title: {source_title}")
    path = PARSED_DIR / fname
    store = TreeStore.load(str(path))
    _store_cache[source_title] = store
    return store


def load_gold() -> list[dict]:
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """Result from one method on one question."""
    question_id: str
    method: str
    retrieved_ids: list[str]
    answer_text: str
    cost: float       # USD
    latency: float    # seconds

    # Computed by harness
    answer_correct: Optional[bool] = None
    path_correct: Optional[bool] = None
    required_recall: float = 0.0
    distractor_hits: int = 0
    abstained: bool = False


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a method across questions."""
    method: str
    n_questions: int
    answer_accuracy: float = 0.0
    path_accuracy: float = 0.0
    required_node_recall: float = 0.0
    distractor_hit_rate: float = 0.0
    abstention_rate: float = 0.0
    cost_per_query: float = 0.0
    latency_per_query: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(result: MethodResult, question: dict) -> MethodResult:
    """Compute all metrics for one method-question pair."""
    retrieved = set(result.retrieved_ids)
    required = set(question.get("required_node_ids", []))
    distractors = set(question.get("distractor_node_ids", []))
    gold_path = question.get("gold_path", [])

    # Abstention
    result.abstained = not result.answer_text.strip()

    # Required node recall: what fraction of required nodes were retrieved?
    # Skip synthetic __ref_ IDs that don't exist in the tree
    real_required = {nid for nid in required if "__ref_" not in nid}
    if real_required:
        result.required_recall = len(retrieved & real_required) / len(real_required)
    else:
        result.required_recall = 1.0 if retrieved else 0.0

    # Distractor hit rate: how many distractors were in the retrieved set?
    result.distractor_hits = len(retrieved & distractors)

    # Path accuracy: did we retrieve at least one required node?
    # Strict check — only direct node ID match counts.
    result.path_correct = bool(retrieved & real_required) if real_required else False

    # Answer accuracy: does the retrieved evidence contain the key information
    # needed to answer correctly?
    #
    # We check TWO things:
    # 1. Evidence overlap: does the answer/retrieved text contain words from
    #    the gold_evidence (the actual regulatory text)?
    # 2. Required node hit: did we retrieve at least one required node?
    #
    # A method "answers correctly" if it retrieves the right evidence,
    # NOT if it generates matching prose.
    result.answer_correct = False

    if result.answer_text.strip():
        answer_lower = result.answer_text.lower()

        # Check against gold_evidence text (the actual regulatory content)
        evidence_words = set()
        for ev in question.get("gold_evidence", []):
            et = ev.get("evidence_text", "").lower()
            evidence_words.update(re.findall(r"[a-z]{4,}", et))

        if evidence_words:
            answer_words = set(re.findall(r"[a-z]{4,}", answer_lower))
            # How much of the gold evidence appears in the answer?
            evidence_recall = len(evidence_words & answer_words) / len(evidence_words)
            # Also check: did we hit a required node?
            hit_required = bool(retrieved & real_required) if real_required else False

            # Correct if: evidence recall > 20% AND hit at least one required node
            # OR evidence recall > 50% (strong content match even without exact node ID)
            result.answer_correct = (
                (evidence_recall > 0.2 and hit_required) or
                evidence_recall > 0.5
            )

    return result


def aggregate(results: list[MethodResult], method: str) -> AggregateMetrics:
    """Aggregate metrics across all questions for one method."""
    method_results = [r for r in results if r.method == method]
    n = len(method_results)
    if n == 0:
        return AggregateMetrics(method=method, n_questions=0)

    return AggregateMetrics(
        method=method,
        n_questions=n,
        answer_accuracy=sum(1 for r in method_results if r.answer_correct) / n,
        path_accuracy=sum(1 for r in method_results if r.path_correct) / n,
        required_node_recall=sum(r.required_recall for r in method_results) / n,
        distractor_hit_rate=sum(r.distractor_hits for r in method_results) / sum(
            len(set(q.get("distractor_node_ids", [])))
            for q, r in zip(load_gold(), method_results)
        ) if method_results else 0.0,
        abstention_rate=sum(1 for r in method_results if r.abstained) / n,
        cost_per_query=sum(r.cost for r in method_results) / n,
        latency_per_query=sum(r.latency for r in method_results) / n,
    )


def breakdown(results: list[MethodResult], questions: list[dict],
              method: str, group_key: str) -> dict[str, AggregateMetrics]:
    """Break down metrics by a question attribute (domain, failure_type, etc)."""
    method_results = [(r, q) for r, q in zip(results, questions) if r.method == method]
    groups: dict[str, list[MethodResult]] = defaultdict(list)
    for r, q in method_results:
        key_val = str(q.get(group_key, "unknown"))
        groups[key_val].append(r)

    return {
        k: AggregateMetrics(
            method=method,
            n_questions=len(rs),
            answer_accuracy=sum(1 for r in rs if r.answer_correct) / len(rs),
            path_accuracy=sum(1 for r in rs if r.path_correct) / len(rs),
            required_node_recall=sum(r.required_recall for r in rs) / len(rs),
            distractor_hit_rate=sum(r.distractor_hits for r in rs) / max(len(rs), 1),
            abstention_rate=sum(1 for r in rs if r.abstained) / len(rs),
            cost_per_query=sum(r.cost for r in rs) / len(rs),
            latency_per_query=sum(r.latency for r in rs) / len(rs),
        )
        for k, rs in groups.items()
    }


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

MethodFn = Callable[[dict, TreeStore], MethodResult]

# Thread config: LLM methods get 8, non-LLM get 12
LLM_METHODS = {"rag_cot", "rag_judge", "tree_traversal"}
THREADS_NON_LLM = 12
THREADS_LLM = 8

# Checkpoint config
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "results" / "checkpoints"


def _process_one(args: tuple[dict, MethodFn, str]) -> tuple[MethodResult, dict]:
    """Process one question — used by thread pool."""
    q, method_fn, method_name = args
    try:
        store = load_store(q["source_title"])
    except Exception as e:
        return MethodResult(
            question_id=q["question_id"], method=method_name,
            retrieved_ids=[], answer_text="", cost=0.0, latency=0.0,
        ), q

    t0 = time.time()
    try:
        result = method_fn(q, store)
    except Exception as e:
        result = MethodResult(
            question_id=q["question_id"], method=method_name,
            retrieved_ids=[], answer_text="", cost=0.0, latency=time.time() - t0,
        )
    return result, q


def _load_checkpoint(method_name: str) -> dict[str, MethodResult]:
    """Load checkpoint for a method. Returns {question_id: MethodResult}."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{method_name}.json"
    if not ckpt_path.exists():
        return {}
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both old format (list) and new format (dict with metadata)
    rows = data.get("results", data) if isinstance(data, dict) else data
    results = {}
    for row in rows:
        r = MethodResult(
            question_id=row["question_id"],
            method=method_name,
            retrieved_ids=row.get("retrieved_ids", []),
            answer_text=row.get("answer_text", ""),
            cost=row.get("cost", 0.0),
            latency=row.get("latency", 0.0),
            answer_correct=row.get("answer_correct"),
            path_correct=row.get("path_correct"),
            required_recall=row.get("required_recall", 0.0),
            distractor_hits=row.get("distractor_hits", 0),
            abstained=row.get("abstained", False),
        )
        results[r.question_id] = r
    return results


def _save_checkpoint(method_name: str, results: list[MethodResult]):
    """Save checkpoint for a method. Named with method + timestamp."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{method_name}.json"
    rows = []
    for r in results:
        rows.append({
            "question_id": r.question_id,
            "method": method_name,
            "retrieved_ids": r.retrieved_ids,
            "answer_text": r.answer_text[:500],
            "cost": r.cost,
            "latency": r.latency,
            "answer_correct": r.answer_correct,
            "path_correct": r.path_correct,
            "required_recall": r.required_recall,
            "distractor_hits": r.distractor_hits,
            "abstained": r.abstained,
        })
    meta = {
        "method": method_name,
        "n_results": len(rows),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": rows,
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    # Also save timestamped backup so we never lose data
    backup_path = CHECKPOINT_DIR / f"{method_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def run_method(method_name: str, method_fn: MethodFn,
               questions: list[dict], limit: int | None = None) -> list[MethodResult]:
    """Run a method on all questions with thread pool.

    Checkpoints every 50 questions. On resume, skips already-completed questions.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    qs = questions[:limit] if limit else questions
    n_threads = THREADS_LLM if method_name in LLM_METHODS else THREADS_NON_LLM

    # Load checkpoint — skip already-done questions
    completed = _load_checkpoint(method_name)
    remaining = [q for q in qs if q["question_id"] not in completed]
    results: list[MethodResult] = list(completed.values())

    print(f"\n{'='*60}", flush=True)
    print(f"Running: {method_name} ({len(remaining)} remaining, "
          f"{len(completed)} cached, {n_threads} threads)", flush=True)
    print(f"{'='*60}", flush=True)

    if not remaining:
        acc = sum(1 for r in results if r.answer_correct) / max(len(results), 1)
        recall = sum(r.required_recall for r in results) / max(len(results), 1)
        print(f"  DONE (from checkpoint): {len(results)} questions, "
              f"acc={acc:.2f}, recall={recall:.2f}", flush=True)
        return results

    # Pre-load stores
    titles = set(q["source_title"] for q in remaining)
    for title in titles:
        try:
            load_store(title)
        except Exception:
            pass

    args_list = [(q, method_fn, method_name) for q in remaining]
    done_new = 0
    total_cost = sum(r.cost for r in results)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_process_one, args): i for i, args in enumerate(args_list)}
        for future in as_completed(futures):
            result, q = future.result()
            result = compute_metrics(result, q)
            results.append(result)
            done_new += 1
            total_cost += result.cost

            # Checkpoint every 50 new results
            if done_new % 50 == 0:
                _save_checkpoint(method_name, results)
                acc = sum(1 for r in results if r.answer_correct) / len(results)
                recall = sum(r.required_recall for r in results) / len(results)
                print(f"  [{done_new}/{len(remaining)}] acc={acc:.2f} "
                      f"recall={recall:.2f} cost=${total_cost:.4f} "
                      f"(checkpointed)", flush=True)

    # Final checkpoint
    _save_checkpoint(method_name, results)

    # Sort by original question order
    id_order = {q["question_id"]: i for i, q in enumerate(qs)}
    results.sort(key=lambda r: id_order.get(r.question_id, 0))

    acc = sum(1 for r in results if r.answer_correct) / max(len(results), 1)
    recall = sum(r.required_recall for r in results) / max(len(results), 1)
    print(f"  DONE: {len(results)} questions, acc={acc:.2f}, "
          f"recall={recall:.2f}, total_cost=${total_cost:.4f}", flush=True)

    return results


def save_results(all_results: dict[str, list[MethodResult]], questions: list[dict]):
    """Save results to JSON with all breakdowns."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "metadata": {
            "n_questions": len(questions),
            "methods": list(all_results.keys()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {},
        "breakdowns": {},
        "per_question": {},
    }

    for method, results in all_results.items():
        # Summary
        agg = aggregate(results, method)
        output["summary"][method] = asdict(agg)

        # Breakdowns
        output["breakdowns"][method] = {}
        for key in ["domain", "failure_type", "difficulty", "tree_depth"]:
            bd = breakdown(results, questions, method, key)
            output["breakdowns"][method][key] = {k: asdict(v) for k, v in bd.items()}

        # Per-question
        output["per_question"][method] = [
            {
                "question_id": r.question_id,
                "answer_correct": r.answer_correct,
                "path_correct": r.path_correct,
                "required_recall": r.required_recall,
                "distractor_hits": r.distractor_hits,
                "abstained": r.abstained,
                "cost": r.cost,
                "latency": r.latency,
                "n_retrieved": len(r.retrieved_ids),
            }
            for r in results
        ]

    # Save timestamped results (never overwrite)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"baseline_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Also save as latest (for easy access)
    latest_path = RESULTS_DIR / "baseline_results_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")
    print(f"Latest copy:   {latest_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Method':<25s} {'Acc':>6s} {'Path':>6s} {'Recall':>8s} {'Distr':>6s} {'Abst':>6s} {'Cost':>8s} {'Lat':>6s}")
    print(f"{'-'*90}")
    for method in all_results:
        agg = output["summary"][method]
        print(f"{method:<25s} "
              f"{agg['answer_accuracy']:>5.1%} "
              f"{agg['path_accuracy']:>5.1%} "
              f"{agg['required_node_recall']:>7.1%} "
              f"{agg['distractor_hit_rate']:>5.2f} "
              f"{agg['abstention_rate']:>5.1%} "
              f"${agg['cost_per_query']:>6.4f} "
              f"{agg['latency_per_query']:>5.2f}s")

    return output
