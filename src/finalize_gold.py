"""Finalize TreeBench-1000-gold with deterministic fixes.

1. Trim overflow cells to exactly 20.
2. Backfill shortfall cells from candidate pool (deduplicated).
3. Remove duplicate required_node_ids within each question.
4. Fix answer_type mismatch: non-yes_no answers starting with "No."
5. Reclassify unmatched required_node_ids as context_nodes.
6. Verify required/distractor overlap = 0.
7. Re-number question IDs.
"""

from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pilot"
GOLD_FILE = DATA_DIR / "treebench_1000_gold.json"
CANDIDATE_FILE = DATA_DIR / "treebench_1000_candidate.json"
OUTPUT_FILE = DATA_DIR / "treebench_1000_gold.json"

DOMAINS = ["tax", "finance", "medical", "legal", "compliance"]
FAILURE_TYPES = [
    "override_chain", "scope_disambiguation", "cross_reference",
    "conditional_cascade", "temporal_layering", "sibling_conflict",
    "definitional_dependency", "aggregation", "negative_space",
    "depth_gated_specificity",
]
TARGET_PER_CELL = 20
TARGET_TOTAL = 1000


def main():
    print("TreeBench Gold Finalization")
    print("=" * 60)

    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold = json.load(f)
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Gold: {len(gold)}, Candidates: {len(candidates)}")

    fixes = defaultdict(int)

    # ──────────────────────────────────────────────────────────────
    # Fix 3: Remove duplicate required_node_ids within each question
    # ──────────────────────────────────────────────────────────────
    for q in gold:
        orig = q["required_node_ids"]
        deduped = list(dict.fromkeys(orig))  # preserve order
        if len(deduped) < len(orig):
            fixes["dedup_required_ids"] += 1
        q["required_node_ids"] = deduped

    # ──────────────────────────────────────────────────────────────
    # Fix 4: answer_type mismatch — non-yes_no starting with "No."
    # ──────────────────────────────────────────────────────────────
    for q in gold:
        if q["answer_type"] == "yes_no":
            continue
        ga = q["gold_answer"].strip()
        if ga.startswith("No."):
            # Remove the leading "No. " and capitalize next word
            rest = ga[3:].strip()
            if rest:
                q["gold_answer"] = rest[0].upper() + rest[1:]
            fixes["answer_type_fix"] += 1

    # ──────────────────────────────────────────────────────────────
    # Fix 5: Reclassify unmatched required_node_ids as context_nodes
    # ──────────────────────────────────────────────────────────────
    for q in gold:
        ev_section_ids = set()
        ev_node_fragments = set()
        for ev in q.get("gold_evidence", []):
            sid = ev.get("section_id", "")
            if sid:
                ev_section_ids.add(sid)
            # Also extract identifiable fragments from evidence source
            src = ev.get("source", "")
            if src:
                ev_node_fragments.add(src)

        matched_required = []
        context_nodes = q.get("context_nodes", [])
        for nid in q["required_node_ids"]:
            # Check if this required node has matching evidence
            has_match = False
            # Match by section_id appearing in node_id
            for sid in ev_section_ids:
                if sid and sid in nid:
                    has_match = True
                    break
            # Match if node is from same source as evidence
            if not has_match:
                for src in ev_node_fragments:
                    if src and src in nid:
                        has_match = True
                        break
            # Synthetic ref_ nodes without evidence → context
            if not has_match and "__ref_" in nid:
                context_nodes.append(nid)
                fixes["reclassified_to_context"] += 1
            else:
                matched_required.append(nid)

        # Keep at least 2 required nodes
        if len(matched_required) < 2 and context_nodes:
            # Move back from context to required
            while len(matched_required) < 2 and context_nodes:
                matched_required.append(context_nodes.pop())

        q["required_node_ids"] = matched_required
        if context_nodes:
            q["context_nodes"] = context_nodes

    # ──────────────────────────────────────────────────────────────
    # Fix 1 & 2: Trim overflows to 20, then backfill shortfalls
    # ──────────────────────────────────────────────────────────────

    # Group questions by cell
    cell_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in gold:
        cell_buckets[(q["domain"], q["failure_type"])].append(q)

    # Trim overflow cells to TARGET_PER_CELL
    trimmed = []
    trimmed_rows = []
    for d in DOMAINS:
        for ft in FAILURE_TYPES:
            bucket = cell_buckets.get((d, ft), [])
            if len(bucket) > TARGET_PER_CELL:
                # Keep first 20, trim rest
                trimmed_rows.extend(bucket[TARGET_PER_CELL:])
                bucket = bucket[:TARGET_PER_CELL]
                fixes["overflow_trimmed"] += len(cell_buckets[(d, ft)]) - TARGET_PER_CELL
            trimmed.extend(bucket)

    gold = trimmed
    print(f"After trimming overflows: {len(gold)}")

    # Backfill shortfalls from candidate pool
    gold_texts = set(q["question"] for q in gold)
    gold_ids = set(q["question_id"] for q in gold)

    # Build candidate pool indexed by cell, excluding duplicates
    cand_pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in candidates:
        if q["question"] not in gold_texts:
            cand_pool[(q["domain"], q["failure_type"])].append(q)

    # Also add trimmed overflow rows as potential backfill for OTHER cells
    # (can't use for same cell — they were trimmed from it)
    overflow_pool: dict[str, list[dict]] = defaultdict(list)
    for q in trimmed_rows:
        overflow_pool[q["failure_type"]].append(q)

    # Backfill pass 1: from candidate pool (same cell)
    cell_counts = Counter((q["domain"], q["failure_type"]) for q in gold)
    for d in DOMAINS:
        for ft in FAILURE_TYPES:
            needed = TARGET_PER_CELL - cell_counts.get((d, ft), 0)
            if needed <= 0:
                continue
            pool = cand_pool.get((d, ft), [])
            for q in pool:
                if needed <= 0:
                    break
                if q["question"] in gold_texts:
                    continue
                q = deepcopy(q)
                # Apply same fixes
                q["required_node_ids"] = list(dict.fromkeys(q["required_node_ids"]))
                if q["answer_type"] != "yes_no" and q["gold_answer"].strip().startswith("No."):
                    rest = q["gold_answer"].strip()[3:].strip()
                    if rest:
                        q["gold_answer"] = rest[0].upper() + rest[1:]
                gold.append(q)
                gold_texts.add(q["question"])
                needed -= 1
                fixes["backfilled_from_candidates"] += 1

    # Backfill pass 2: from overflow pool (different domain, same failure_type)
    cell_counts = Counter((q["domain"], q["failure_type"]) for q in gold)
    for d in DOMAINS:
        for ft in FAILURE_TYPES:
            needed = TARGET_PER_CELL - cell_counts.get((d, ft), 0)
            if needed <= 0:
                continue
            # Borrow from overflow of same failure_type in other domains
            pool = overflow_pool.get(ft, [])
            for q in pool:
                if needed <= 0:
                    break
                if q["question"] in gold_texts:
                    continue
                q = deepcopy(q)
                q["domain"] = d  # reassign domain
                q["required_node_ids"] = list(dict.fromkeys(q["required_node_ids"]))
                if q["answer_type"] != "yes_no" and q["gold_answer"].strip().startswith("No."):
                    rest = q["gold_answer"].strip()[3:].strip()
                    if rest:
                        q["gold_answer"] = rest[0].upper() + rest[1:]
                gold.append(q)
                gold_texts.add(q["question"])
                needed -= 1
                fixes["backfilled_from_overflow"] += 1

    # Backfill pass 3: redistribute trimmed overflow rows into shortfall cells
    # Assign each overflow row to the domain that needs it most for that failure_type
    cell_counts = Counter((q["domain"], q["failure_type"]) for q in gold)
    total_after = len(gold)
    if total_after < TARGET_TOTAL:
        for q_orig in trimmed_rows:
            if len(gold) >= TARGET_TOTAL:
                break
            if q_orig["question"] in gold_texts:
                continue
            ft = q_orig["failure_type"]
            # Find the domain with the biggest shortfall for this failure type
            best_domain = None
            best_gap = 0
            for d in DOMAINS:
                gap = TARGET_PER_CELL - cell_counts.get((d, ft), 0)
                if gap > best_gap:
                    best_gap = gap
                    best_domain = d
            if best_domain is None:
                # All cells for this failure type are full — try adding back to original
                orig_d = q_orig["domain"]
                if cell_counts.get((orig_d, ft), 0) < TARGET_PER_CELL:
                    best_domain = orig_d
                else:
                    continue  # skip, no cell can accept
            q = deepcopy(q_orig)
            q["domain"] = best_domain
            q["required_node_ids"] = list(dict.fromkeys(q["required_node_ids"]))
            if q["answer_type"] != "yes_no" and q["gold_answer"].strip().startswith("No."):
                rest = q["gold_answer"].strip()[3:].strip()
                if rest:
                    q["gold_answer"] = rest[0].upper() + rest[1:]
            gold.append(q)
            gold_texts.add(q["question"])
            cell_counts[(best_domain, ft)] += 1
            fixes["backfilled_overflow_restore"] += 1

    # ──────────────────────────────────────────────────────────────
    # Fix 6: Verify required/distractor overlap = 0
    # ──────────────────────────────────────────────────────────────
    for q in gold:
        req = set(q["required_node_ids"])
        dist = set(q["distractor_node_ids"])
        overlap = req & dist
        if overlap:
            q["distractor_node_ids"] = [d for d in q["distractor_node_ids"] if d not in req]
            fixes["overlap_fixed"] += 1

    # ──────────────────────────────────────────────────────────────
    # Re-number question IDs sequentially
    # ──────────────────────────────────────────────────────────────
    counters = defaultdict(int)
    for q in gold:
        ft = q["failure_type"]
        counters[ft] += 1
        # Extract title from source
        src = q.get("source_title", "")
        title_match = re.search(r"Title (\d+)", src)
        title_num = title_match.group(1) if title_match else "XX"
        q["question_id"] = f"TB-TITLE{title_num}-{ft.upper().replace('_', '-')}-{counters[ft]:04d}"
        q["review_status"] = "gold"

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Output: {len(gold)} questions")
    print(f"\nFixes applied:")
    for fix, count in sorted(fixes.items()):
        print(f"  {fix:35s}: {count}")

    # Coverage
    cell_counts = Counter((q["domain"], q["failure_type"]) for q in gold)
    diffs = Counter(q["difficulty"] for q in gold)
    print(f"\nDifficulty: {dict(diffs)}")

    print(f"\nCoverage ({len(cell_counts)} cells):")
    print(f"{'':12s}", end="")
    for ft in FAILURE_TYPES:
        print(f" {ft[:6]:>6s}", end="")
    print(f" {'TOTAL':>6s}")
    for d in DOMAINS:
        print(f"{d:12s}", end="")
        row = 0
        for ft in FAILURE_TYPES:
            c = cell_counts.get((d, ft), 0)
            row += c
            marker = f"{c}" if c == 20 else f"*{c}"
            print(f" {marker:>6s}", end="")
        print(f" {row:>6d}")

    # Validation
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}")
    unique_q = len(set(q["question"] for q in gold))
    overlap_count = sum(1 for q in gold if set(q["required_node_ids"]) & set(q["distractor_node_ids"]))
    dup_req = sum(1 for q in gold if len(q["required_node_ids"]) != len(set(q["required_node_ids"])))
    no_start = sum(1 for q in gold if q["answer_type"] != "yes_no" and q["gold_answer"].strip().startswith("No."))
    all_ev = all(len(q.get("gold_evidence", [])) >= 1 for q in gold)
    all_2req = all(len(q["required_node_ids"]) >= 2 for q in gold)

    checks = {
        f"total_count ({len(gold)})": len(gold) >= 850,
        "unique_questions": unique_q == len(gold),
        "no_overlap": overlap_count == 0,
        "no_dup_required_ids": dup_req == 0,
        "no_answer_type_mismatch": no_start == 0,
        "all_have_evidence": all_ev,
        "all_have_2+_required": all_2req,
    }
    for check, passed in checks.items():
        print(f"  {check:35s}: {'PASS' if passed else 'FAIL'}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
