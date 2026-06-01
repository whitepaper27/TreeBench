"""Candidate Validator — quality-rank candidates before question generation.

Filters:
1. Depth >= 4 (skip shallow structural-only nodes)
2. Target node text >= 50 chars (substantive content, not just headings)
3. Confounder strength: parent-child vocabulary overlap for override/depth-gated
4. Sibling density: >= 2 siblings with text for scope/sibling types
5. Cross-ref resolution: referenced section exists in tree

Outputs ranked candidates per (domain, failure_type) cell.
"""

from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_node import TreeNode, TreeStore
from pattern_hunters import CandidateMatch
from question_schema import (
    CONFOUNDER_MAP, SOURCE_META, DOMAINS, FAILURE_TYPES, ContextPackage,
)


MIN_DEPTH = 4
MIN_TEXT_LEN = 50
TOP_PER_CELL = 30  # top candidates per (domain, failure_type) for 1000-scale


def _word_set(text: str) -> set[str]:
    """Extract content words (length >= 3) from text."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)}


def _vocabulary_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity of word sets. Higher = more confounding for RAG."""
    words_a = _word_set(text_a)
    words_b = _word_set(text_b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def validate_candidate(candidate: CandidateMatch, store: TreeStore) -> tuple[bool, float, str]:
    """Validate a candidate and return (is_valid, quality_score, rejection_reason)."""
    target = store.get(candidate.target_node_id)
    if not target:
        return False, 0.0, "target_node_not_found"

    # Filter 1: Depth
    if candidate.tree_depth < MIN_DEPTH:
        return False, 0.0, f"depth_too_shallow ({candidate.tree_depth} < {MIN_DEPTH})"

    # Filter 2: Text quality
    if len(target.text.strip()) < MIN_TEXT_LEN:
        return False, 0.0, f"text_too_short ({len(target.text)} < {MIN_TEXT_LEN})"

    # Score: start with base from confidence
    score = candidate.confidence * 20

    # Bonus for depth (deeper = more structurally interesting)
    score += min(candidate.tree_depth / 8.0, 1.0) * 25

    # Bonus for text length (more substance)
    score += min(len(target.text) / 500, 1.0) * 15

    # Filter 3 & Bonus: Confounder strength
    ancestors = store.ancestors(candidate.target_node_id)
    if ancestors:
        parent = ancestors[-1]
        if parent.text:
            overlap = _vocabulary_overlap(parent.text, target.text)
            score += overlap * 30  # Higher overlap = better trap for RAG

            # For override/depth-gated, require some overlap (it's the whole point)
            if candidate.failure_type in ("override_chain", "depth_gated_specificity"):
                if overlap < 0.05:
                    return False, 0.0, "no_confounder_overlap"

    # Filter 4: Sibling density for scope/sibling types
    if candidate.failure_type in ("scope_disambiguation", "sibling_conflict"):
        siblings = store.siblings(candidate.target_node_id)
        siblings_with_text = [s for s in siblings if len(s.text.strip()) >= 30]
        if len(siblings_with_text) < 2:
            return False, 0.0, f"insufficient_siblings ({len(siblings_with_text)} < 2)"
        score += min(len(siblings_with_text) / 5, 1.0) * 10

    # Filter 5: Cross-ref resolution
    if candidate.failure_type == "cross_reference":
        if not target.cross_refs:
            return False, 0.0, "no_cross_refs_found"
        # Check if at least one referenced section exists
        found_ref = False
        for ref in target.cross_refs:
            ref_num = re.search(r"[\d]+(?:\.[\d\w\-]+)*", ref)
            if ref_num:
                for node in store.nodes.values():
                    if node.number == ref_num.group(0) and node.id != target.id:
                        found_ref = True
                        break
            if found_ref:
                break
        if not found_ref:
            score -= 10  # Penalize but don't reject (ref might be in another title)

    # Bonus for cross-references (structural complexity)
    if target.cross_refs:
        score += min(len(target.cross_refs) / 3, 1.0) * 10

    # Bonus for having the override signal deeper in the text (not just the first word)
    signal_pos = target.text.lower().find(candidate.signal_text.lower())
    if signal_pos > 20:
        score += 5  # Signal is embedded, not at start

    return True, score, ""


def build_context_package(candidate: CandidateMatch, store: TreeStore) -> ContextPackage:
    """Build a full context package for a validated candidate."""
    target = store.get(candidate.target_node_id)
    ancestors = store.ancestors(candidate.target_node_id)
    siblings = store.siblings(candidate.target_node_id)

    # Domain and source
    domain, source_title = SOURCE_META.get(
        candidate.source_id, ("unknown", candidate.source_id)
    )

    # Ancestral context
    anc_ctx = []
    for a in ancestors:
        anc_ctx.append({
            "node_type": a.node_type,
            "number": a.number,
            "heading": a.heading,
            "text_preview": a.text[:300].strip(),
        })

    # Parent rule text (what RAG would likely retrieve)
    parent_rule = ""
    if ancestors:
        parent = ancestors[-1]
        parent_rule = parent.text[:500].strip()

    # Sibling context
    sib_ctx = []
    for s in siblings[:5]:
        sib_ctx.append({
            "node_id": s.id,
            "heading": s.heading or f"{s.node_type} {s.number}",
            "text_preview": s.text[:200].strip(),
        })

    # Gold path as array
    path_parts = target.path.split(" > ")
    gold_path = [p.strip() for p in path_parts if p.strip()]

    # Confounder score
    confounder_score = 0.0
    if ancestors and ancestors[-1].text:
        confounder_score = _vocabulary_overlap(ancestors[-1].text, target.text)

    return ContextPackage(
        candidate_source_id=candidate.source_id,
        failure_type=candidate.failure_type,
        structural_confounder_type=CONFOUNDER_MAP.get(candidate.failure_type, "parent_child"),
        domain=domain,
        source_title=source_title,
        target_node_id=target.id,
        target_node_type=target.node_type,
        target_node_number=target.number,
        target_node_heading=target.heading or f"{target.node_type} {target.number}",
        target_node_text=target.text[:1000].strip(),
        target_tree_path=target.path,
        target_tree_depth=candidate.tree_depth,
        ancestral_context=anc_ctx,
        parent_rule_text=parent_rule,
        sibling_context=sib_ctx,
        cross_refs=target.cross_refs[:5],
        gold_path=gold_path,
        confounder_score=confounder_score,
        signal_text=candidate.signal_text,
    )


def validate_all(parsed_dir: str, candidates_dir: str, output_dir: str) -> dict:
    """Validate all candidates across all titles. Return ranked candidates per cell."""
    os.makedirs(output_dir, exist_ok=True)

    # Map source_id -> (tree_path, candidates_path)
    title_files = {
        "ECFR_TITLE12_XML": ("ECFR-title12", "ECFR-title12"),
        "ECFR_TITLE15_XML": ("ECFR-title15", "ECFR-title15"),
        "ECFR_TITLE17_XML": ("ECFR-title17", "ECFR-title17"),
        "ECFR_TITLE21_XML": ("ECFR-title21", "ECFR-title21"),
        "ECFR_TITLE26_XML": ("ECFR-title26", "ECFR-title26"),
        "ECFR_TITLE29_XML": ("ECFR-title29", "ECFR-title29"),
        "ECFR_TITLE31_XML": ("ECFR-title31", "ECFR-title31"),
        "ECFR_TITLE40_XML": ("ECFR-title40", "ECFR-title40"),
        "ECFR_TITLE42_XML": ("ECFR-title42", "ECFR-title42"),
        "ECFR_TITLE45_XML": ("ECFR-title45", "ECFR-title45"),
    }

    # Collect validated candidates grouped by (domain, failure_type)
    cells: dict[tuple[str, str], list[tuple[float, ContextPackage]]] = defaultdict(list)

    total_checked = 0
    total_valid = 0
    rejection_counts: dict[str, int] = defaultdict(int)

    for source_id, (tree_name, cand_name) in title_files.items():
        tree_path = os.path.join(parsed_dir, f"{tree_name}_tree.json")
        cand_path = os.path.join(candidates_dir, f"{cand_name}_candidates.json")

        if not os.path.exists(tree_path) or not os.path.exists(cand_path):
            print(f"  SKIP {source_id} — files not found")
            continue

        domain = SOURCE_META.get(source_id, ("unknown",))[0]
        print(f"  Validating {source_id} ({domain})...", flush=True)

        store = TreeStore.load(tree_path)

        with open(cand_path, "r", encoding="utf-8") as f:
            candidates_by_type = json.load(f)

        title_valid = 0
        for failure_type, cand_dicts in candidates_by_type.items():
            for cd in cand_dicts:
                candidate = CandidateMatch(**cd)
                total_checked += 1

                is_valid, score, reason = validate_candidate(candidate, store)
                if not is_valid:
                    rejection_counts[reason] += 1
                    continue

                total_valid += 1
                title_valid += 1

                ctx = build_context_package(candidate, store)
                cell_key = (domain, failure_type)
                cells[cell_key].append((score, ctx))

        print(f"    Valid: {title_valid}/{sum(len(v) for v in candidates_by_type.values())}")

    # Sort each cell by score (descending) and keep top N
    ranked_cells: dict[str, list[dict]] = {}
    for (domain, ftype), scored_ctxs in cells.items():
        scored_ctxs.sort(key=lambda x: x[0], reverse=True)
        top = scored_ctxs[:TOP_PER_CELL]
        cell_key = f"{domain}/{ftype}"
        ranked_cells[cell_key] = [
            {"score": round(s, 2), "context": ctx.to_dict()} for s, ctx in top
        ]

    # Save ranked candidates
    out_path = os.path.join(output_dir, "validated_candidates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranked_cells, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total checked:  {total_checked}")
    print(f"Total valid:    {total_valid} ({100*total_valid/max(total_checked,1):.0f}%)")
    print(f"Total rejected: {total_checked - total_valid}")
    print(f"\nRejection reasons:")
    for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:40s}: {count}")
    print(f"\nCells populated: {len(ranked_cells)} / {len(DOMAINS)*len(FAILURE_TYPES)} (target: 50)")
    print(f"Candidates per cell: {TOP_PER_CELL}")

    # Show coverage matrix
    print(f"\nCoverage matrix (candidates per cell):")
    print(f"{'Domain':<15s}", end="")
    for ft in FAILURE_TYPES:
        print(f" {ft[:8]:>8s}", end="")
    print()
    for domain in DOMAINS:
        print(f"{domain:<15s}", end="")
        for ft in FAILURE_TYPES:
            key = f"{domain}/{ft}"
            count = len(ranked_cells.get(key, []))
            marker = f"{count}" if count > 0 else "-"
            print(f" {marker:>8s}", end="")
        print()

    return {
        "total_checked": total_checked,
        "total_valid": total_valid,
        "cells_populated": len(ranked_cells),
        "output_path": out_path,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parsed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "parsed")
    candidates_dir = os.path.join(os.path.dirname(__file__), "..", "data", "questions")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "validated")

    print("TreeBench Candidate Validator")
    print("="*60)
    stats = validate_all(parsed_dir, candidates_dir, output_dir)
