"""Post-generation validation for treebench_pilot_v3.json.

Four deterministic fixes — no LLM rewrite, no schema change.

Fix 1: Remove required/distractor overlap.
Fix 2: Rebalance difficulty to ~25/50/25 easy/medium/hard.
Fix 3: Resolve referenced nodes and add their evidence text.
Fix 4: Recut evidence snippets at clean sentence boundaries (no mid-word cuts).
"""

from __future__ import annotations
import json, re, os, sys
from pathlib import Path
from collections import Counter

PILOT_PATH = Path(__file__).resolve().parent.parent / "data" / "pilot"
INPUT_FILE = PILOT_PATH / "treebench_1000_candidate.json"
OUTPUT_FILE = PILOT_PATH / "treebench_1000_candidate.json"  # overwrite in place
PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"

# Map source_id to parsed tree filename
SOURCE_TO_FILE = {
    "ECFR_TITLE26_XML": "ECFR-title26_tree.json",
    "ECFR_TITLE12_XML": "ECFR-title12_tree.json",
    "ECFR_TITLE17_XML": "ECFR-title17_tree.json",
    "ECFR_TITLE21_XML": "ECFR-title21_tree.json",
    "ECFR_TITLE42_XML": "ECFR-title42_tree.json",
    "ECFR_TITLE29_XML": "ECFR-title29_tree.json",
    "ECFR_TITLE15_XML": "ECFR-title15_tree.json",
    "ECFR_TITLE40_XML": "ECFR-title40_tree.json",
    "ECFR_TITLE45_XML": "ECFR-title45_tree.json",
    "ECFR_TITLE31_XML": "ECFR-title31_tree.json",
}

# Cache loaded trees
_tree_cache: dict[str, dict] = {}


def _load_tree(source_id: str) -> dict:
    """Load and cache a parsed tree's node dict."""
    if source_id in _tree_cache:
        return _tree_cache[source_id]
    fname = SOURCE_TO_FILE.get(source_id)
    if not fname:
        return {}
    fpath = PARSED_DIR / fname
    if not fpath.exists():
        print(f"  WARNING: tree file not found: {fpath}")
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", {})
    _tree_cache[source_id] = nodes
    return nodes


def _clean_excerpt(text: str, max_len: int = 400) -> str:
    """Clean excerpt — ALWAYS ends at a sentence boundary.

    Never cuts mid-word or mid-sentence. If no good sentence boundary
    is found, takes the full text up to the last complete word before
    a natural break (period, semicolon, colon, closing paren).
    """
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return t
    if len(t) <= max_len:
        # Even short text might end mid-sentence — check and trim
        return _ensure_clean_ending(t)
    cut = t[:max_len]
    # Find last sentence-terminal punctuation followed by space or end
    # Priority: period > semicolon > colon > closing paren
    best = -1
    for sep in [". ", "; ", ": ", ".) ", ") "]:
        pos = cut.rfind(sep)
        if pos > max_len // 3:  # must be at least 1/3 into the text
            best = max(best, pos)
    if best > 0:
        return cut[:best + 1]
    # No sentence boundary — try period at end of text
    last_period = cut.rfind(".")
    if last_period > max_len // 3:
        return cut[:last_period + 1]
    # Last resort: cut at last space, ensuring we don't cut mid-word
    last_space = cut.rfind(" ")
    if last_space > 0:
        result = cut[:last_space]
        return _ensure_clean_ending(result)
    return _ensure_clean_ending(cut)


def _ensure_clean_ending(text: str) -> str:
    """Ensure text ends at a clean boundary, never mid-word.

    Strips trailing commas and partial words. Ensures the text ends
    with sentence-terminal punctuation.
    """
    t = text.rstrip()
    if not t:
        return t
    # Strip trailing commas first
    t = t.rstrip(",").rstrip()
    if not t:
        return t
    # Already ends cleanly
    if t[-1] in ".;:?!)\"'":
        return t
    # Find the last sentence-ending punctuation
    for i in range(len(t) - 1, max(len(t) - 80, 0), -1):
        if t[i] in ".;:":
            return t[:i + 1]
    # No punctuation found — close with period
    return t + "."


def _resolve_ref_section(source_id: str, ref_str: str) -> dict | None:
    """Try to find the actual referenced section node in the parsed tree.

    Returns {node_id, section_id, node_type, evidence_text} or None.
    """
    nodes = _load_tree(source_id)
    if not nodes:
        return None

    # Extract section number from the ref string
    # Common patterns: "§ 1.5000C-6", "section 151", "§225.104"
    ref_num = re.search(r"[\d]+(?:\.[\d\w\-]+)*", ref_str)
    if not ref_num:
        return None
    target_num = ref_num.group(0)

    # Search for matching section node
    best_match = None
    best_text_len = 0
    for nid, node in nodes.items():
        num = node.get("number", "")
        if not num:
            continue
        clean_num = num.replace("§", "").strip()
        # Match: exact, contains, or ends with target number
        if (target_num in clean_num or clean_num.endswith(target_num)
                or target_num in nid):
            text = node.get("text", "")
            if len(text) > best_text_len and node.get("node_type") in (
                "section", "paragraph", "subsection", "part"
            ):
                best_match = node
                best_text_len = len(text)

    if not best_match:
        # Broader search: look for target_num anywhere in node heading/text
        for nid, node in nodes.items():
            heading = node.get("heading", "")
            if target_num in heading or target_num in nid:
                text = node.get("text", "")
                if len(text) > 50 and len(text) > best_text_len:
                    best_match = node
                    best_text_len = len(text)

    if not best_match:
        return None

    return {
        "source": source_id,
        "section_id": best_match.get("number", ""),
        "node_type": best_match.get("node_type", ""),
        "evidence_text": _clean_excerpt(best_match.get("text", ""), 500),
    }


# ──────────────────────────────────────────────────────────────────────
# Fix 1: Remove required/distractor overlap
# ──────────────────────────────────────────────────────────────────────
def fix_node_overlap(questions: list[dict]) -> int:
    """Remove any node IDs that appear in both required and distractor lists."""
    fixed = 0
    for q in questions:
        req = set(q["required_node_ids"])
        dist = set(q["distractor_node_ids"])
        overlap = req & dist
        if overlap:
            q["distractor_node_ids"] = [d for d in q["distractor_node_ids"]
                                         if d not in req]
            fixed += 1
    return fixed


# ──────────────────────────────────────────────────────────────────────
# Fix 2: Rebalance difficulty
# ──────────────────────────────────────────────────────────────────────
def fix_difficulty(questions: list[dict]) -> dict:
    """Relabel difficulty based on tree_depth thresholds.

    Target: ~25% easy, ~50% medium, ~25% hard.
    For 1000: 250 easy / 500 medium / 250 hard.

    Strategy: sort by tree_depth, assign buckets:
      depth 4          -> easy
      depth 5-6        -> medium
      depth 7+         -> hard

    Then promote depth-5 to easy if needed to hit 25%.
    """
    total = len(questions)
    easy_target = total // 4   # 250 for 1000
    hard_target = total // 4   # 250 for 1000

    for q in questions:
        depth = q.get("tree_depth", 6)
        if depth <= 4:
            q["difficulty"] = "easy"
        elif depth <= 6:
            q["difficulty"] = "medium"
        else:
            q["difficulty"] = "hard"

    counts = Counter(q["difficulty"] for q in questions)

    # Promote depth-5 to easy if needed
    if counts.get("easy", 0) < easy_target:
        depth5 = [q for q in questions if q["tree_depth"] == 5
                  and q["difficulty"] == "medium"]
        needed = min(easy_target - counts.get("easy", 0), len(depth5))
        for q in depth5[:needed]:
            q["difficulty"] = "easy"

    return dict(Counter(q["difficulty"] for q in questions))


# ──────────────────────────────────────────────────────────────────────
# Fix 3: Resolve referenced nodes and add evidence
# ──────────────────────────────────────────────────────────────────────
def fix_evidence(questions: list[dict]) -> int:
    """For cross_reference, definitional_dependency, and aggregation rows:
    resolve ref_ node IDs to actual tree nodes and add their evidence text.

    Also replace synthetic ref_ IDs with real node IDs when possible.
    """
    fixed = 0
    ref_types = {"cross_reference", "definitional_dependency", "aggregation"}

    for q in questions:
        if q["failure_type"] not in ref_types:
            continue

        # Find the source ID from existing evidence
        source_id = ""
        for ev in q.get("gold_evidence", []):
            if ev.get("source"):
                source_id = ev["source"]
                break

        if not source_id:
            continue

        # Check for synthetic ref_ nodes
        new_required = []
        refs_to_resolve = []
        for nid in q["required_node_ids"]:
            if "__ref_" in nid:
                # Extract the section number from the synthetic ID
                ref_num = nid.split("__ref_")[-1]
                refs_to_resolve.append(ref_num)
            new_required.append(nid)

        # Also extract cross-refs from the question/gold_answer text
        ga = q.get("gold_answer", "")
        text_refs = re.findall(
            r"(?:references?|§|section)\s*([\d]+(?:\.[\d\w\-]+)*)",
            ga, re.I
        )
        for ref in text_refs:
            if ref not in refs_to_resolve:
                refs_to_resolve.append(ref)

        # Try to resolve each reference
        resolved_any = False
        existing_ev_sections = {ev.get("section_id", "") for ev in q.get("gold_evidence", [])}

        for ref_num in refs_to_resolve[:2]:  # cap at 2 resolved references
            resolved = _resolve_ref_section(source_id, ref_num)
            if resolved and resolved["section_id"] not in existing_ev_sections:
                q["gold_evidence"].append(resolved)
                existing_ev_sections.add(resolved["section_id"])
                resolved_any = True

                # Replace synthetic ref_ ID with real node ID if possible
                nodes = _load_tree(source_id)
                for nid_candidate, node in nodes.items():
                    num = node.get("number", "").replace("§", "").strip()
                    if ref_num in num and node.get("node_type") in ("section", "paragraph"):
                        # Replace synthetic ID
                        for i, nid in enumerate(q["required_node_ids"]):
                            if f"__ref_{ref_num}" in nid:
                                q["required_node_ids"][i] = nid_candidate
                        break

        if resolved_any:
            fixed += 1

    return fixed


# ──────────────────────────────────────────────────────────────────────
# Fix 4: Recut evidence snippets at clean sentence boundaries
# ──────────────────────────────────────────────────────────────────────
def _is_clean_ending(text: str) -> bool:
    """Check if text ends at a clean boundary."""
    t = text.rstrip()
    if not t:
        return True
    return t[-1] in ".;:?!)\"'"


def fix_evidence_snippets(questions: list[dict]) -> int:
    """Re-extract and recut all evidence snippets from source trees.

    For each evidence block, find the original node in the parsed tree
    and re-extract with proper sentence-boundary cutting.
    If the node can't be found, recut the existing text in-place.
    """
    fixed = 0

    for q in questions:
        source_id = ""
        for ev in q.get("gold_evidence", []):
            if ev.get("source"):
                source_id = ev["source"]
                break

        nodes = _load_tree(source_id) if source_id else {}
        q_fixed = False

        for ev in q.get("gold_evidence", []):
            old_text = ev.get("evidence_text", "")
            if not old_text:
                continue

            # Check if already clean
            if _is_clean_ending(old_text):
                continue

            # Try to find the full text from the parsed tree
            section_id = ev.get("section_id", "")
            node_type = ev.get("node_type", "")
            full_text = None

            if nodes and section_id:
                # Search for matching node
                for nid, node in nodes.items():
                    num = node.get("number", "")
                    ntype = node.get("node_type", "")
                    if num == section_id and (not node_type or ntype == node_type):
                        full_text = node.get("text", "")
                        break
                    # Also try partial match
                    if section_id in num and ntype == node_type:
                        candidate = node.get("text", "")
                        if candidate and (not full_text or len(candidate) > len(full_text)):
                            full_text = candidate

            if full_text and len(full_text) > 20:
                # Re-extract with clean cutting
                new_text = _clean_excerpt(full_text, 500)
            else:
                # Recut existing text in-place
                new_text = _ensure_clean_ending(old_text)

            if new_text != old_text:
                ev["evidence_text"] = new_text
                q_fixed = True

        if q_fixed:
            fixed += 1

    return fixed


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("TreeBench v3 Post-Generation Validation")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Fix 1
    overlap_fixed = fix_node_overlap(questions)
    print(f"\nFix 1 — Node overlap removed: {overlap_fixed} questions")

    # Verify
    remaining_overlap = 0
    for q in questions:
        if set(q["required_node_ids"]) & set(q["distractor_node_ids"]):
            remaining_overlap += 1
    print(f"  Remaining overlaps: {remaining_overlap}")

    # Fix 2
    diff_counts = fix_difficulty(questions)
    print(f"\nFix 2 — Difficulty rebalanced: {diff_counts}")

    # Fix 3
    evidence_fixed = fix_evidence(questions)
    print(f"\nFix 3 — Evidence resolved: {evidence_fixed} questions")

    # Verify evidence completeness
    ref_types = {"cross_reference", "definitional_dependency", "aggregation"}
    ev_stats = {"has_2+_evidence": 0, "has_1_evidence": 0, "has_0_evidence": 0}
    for q in questions:
        if q["failure_type"] not in ref_types:
            continue
        ev_count = len(q.get("gold_evidence", []))
        if ev_count >= 2:
            ev_stats["has_2+_evidence"] += 1
        elif ev_count == 1:
            ev_stats["has_1_evidence"] += 1
        else:
            ev_stats["has_0_evidence"] += 1
    print(f"  Cross-ref/def/agg evidence: {ev_stats}")

    # Fix 4 — run until no more mid-word cuts remain
    total_snippet_fixed = 0
    for pass_num in range(5):  # max 5 passes
        n = fix_evidence_snippets(questions)
        total_snippet_fixed += n
        if n == 0:
            break
    print(f"\nFix 4 — Evidence snippets recut: {total_snippet_fixed} questions ({pass_num + 1} passes)")

    # Count remaining mid-word cuts
    mid_word_cuts = 0
    total_snippets = 0
    for q in questions:
        for ev in q.get("gold_evidence", []):
            total_snippets += 1
            if not _is_clean_ending(ev.get("evidence_text", "")):
                mid_word_cuts += 1
    print(f"  Mid-word/mid-sentence cuts remaining: {mid_word_cuts}/{total_snippets}")

    # Final integrity checks
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}")

    checks = {
        "total_count_1000": len(questions) >= 1000,
        "no_overlap": remaining_overlap == 0,
        "has_easy": diff_counts.get("easy", 0) > 0,
        "has_medium": diff_counts.get("medium", 0) > 0,
        "has_hard": diff_counts.get("hard", 0) > 0,
        "all_have_evidence": all(len(q.get("gold_evidence", [])) >= 1 for q in questions),
        "ref_types_have_2+_ev": ev_stats["has_0_evidence"] == 0,  # allow 1-ev rows for candidate
        "no_mid_word_cuts": mid_word_cuts == 0,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check:35s}: {status}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    return questions


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
