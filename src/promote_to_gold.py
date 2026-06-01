"""Promote TreeBench-1000-candidate to TreeBench-1000-gold.

Deterministic cleanup — no regeneration, no LLM calls.

1. Remove duplicate question texts (keep first occurrence).
2. Fix signal hallucinations: if gold_answer cites a signal not in evidence,
   replace with actual text from evidence.
3. Drop evidence blocks < 50 chars.
4. Strip any remaining structural giveaway words from questions.
5. Replace synthetic __ref_ IDs where possible.
6. Re-number question IDs sequentially.
7. Export failed/dropped rows separately.
"""

from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy

PILOT_PATH = Path(__file__).resolve().parent.parent / "data" / "pilot"
INPUT_FILE = PILOT_PATH / "treebench_1000_candidate.json"
GOLD_FILE = PILOT_PATH / "treebench_1000_gold.json"
REJECTED_FILE = PILOT_PATH / "treebench_1000_rejected.json"

BANNED_WORDS = [
    "subordinate", "hierarchy", "hierarchical", "parent rule",
    "child exception", "child provision", "sibling node",
    "tree position", "subtree", "traversal", "tree depth",
    "ancestor", "descendant", "parent node", "child node",
]
BANNED_RE = re.compile("|".join(re.escape(w) for w in BANNED_WORDS), re.I)


def main():
    print("TreeBench Gold Promotion")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidates")

    gold = []
    rejected = []
    seen_texts = set()
    fixes = defaultdict(int)

    for q in candidates:
        q = deepcopy(q)
        reject_reason = None

        # ── 1. Deduplicate by question text ──
        qt = q["question"]
        if qt in seen_texts:
            reject_reason = "duplicate_question_text"
            rejected.append({"question_id": q["question_id"], "reason": reject_reason})
            fixes["dedup"] += 1
            continue
        seen_texts.add(qt)

        # ── 2. Fix signal hallucinations ──
        ga = q["gold_answer"]
        signal_matches = list(re.finditer(
            r'(?:contains the (?:signal|language)|This provision contains the language) "([^"]+)"'
            r'(?: which [^.]*\.)?',
            ga
        ))
        for m in signal_matches:
            sig = m.group(1)
            full_match = m.group(0)
            # Check if signal is in any evidence text
            found = any(
                sig.lower() in ev.get("evidence_text", "").lower()
                for ev in q.get("gold_evidence", [])
            )
            if not found:
                # Try to find a real excerpt from evidence
                replacement = ""
                for ev in q.get("gold_evidence", []):
                    et = ev.get("evidence_text", "").strip()
                    if len(et) > 30:
                        clean = re.sub(r"^\([a-zA-Z0-9]+\)\s*", "", et)
                        clean = re.sub(r"^§\s*[\d]+(?:\.[\d\w\-]+)*\s+\S+\s*[.]\s*", "", clean)
                        cm = re.match(r"(.{15,60}?)[,;.]", clean.strip())
                        if cm:
                            replacement = cm.group(1).strip()
                            break
                if replacement:
                    new_phrase = f'states "{replacement}"'
                else:
                    # No good excerpt — remove the claim entirely
                    new_phrase = "qualifies the general rule"
                ga = ga.replace(full_match, new_phrase)
                fixes["signal_fix"] += 1
        # Catch any remaining unmatched patterns
        ga = re.sub(
            r'contains the (?:signal|language) "([^"]+)"[^.]*\.',
            lambda m: (
                m.group(0)
                if any(m.group(1).lower() in ev.get("evidence_text", "").lower()
                       for ev in q.get("gold_evidence", []))
                else "qualifies the general rule."
            ),
            ga,
        )
        q["gold_answer"] = ga

        # ── 3. Drop short evidence blocks ──
        good_evidence = []
        for ev in q.get("gold_evidence", []):
            if len(ev.get("evidence_text", "")) >= 50:
                good_evidence.append(ev)
            else:
                fixes["short_evidence_dropped"] += 1
        q["gold_evidence"] = good_evidence

        # If no evidence remains, reject
        if not q["gold_evidence"]:
            reject_reason = "no_evidence_after_cleanup"
            rejected.append({"question_id": q["question_id"], "reason": reject_reason})
            fixes["no_evidence"] += 1
            continue

        # ── 4. Strip structural giveaways from question ──
        if BANNED_RE.search(q["question"]):
            q["question"] = BANNED_RE.sub("", q["question"])
            q["question"] = re.sub(r"\s+", " ", q["question"]).strip()
            fixes["giveaway_stripped"] += 1

        # ── 5. Mark synthetic __ref_ IDs (can't resolve without tree, just flag) ──
        has_synthetic = any("__ref_" in nid for nid in q["required_node_ids"])
        if has_synthetic:
            fixes["synthetic_ref_remaining"] += 1

        gold.append(q)

    # ── 6. Re-number question IDs sequentially ──
    counters = defaultdict(int)
    for q in gold:
        ft = q["failure_type"]
        counters[ft] += 1
        title_short = q["question_id"].split("-")[1]  # e.g., TITLE26
        q["question_id"] = f"TB-{title_short}-{ft.upper().replace('_', '-')}-{counters[ft]:04d}"

    # ── 7. Set review status ──
    for q in gold:
        q["review_status"] = "gold_candidate"

    # Save gold
    with open(GOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    # Save rejected
    with open(REJECTED_FILE, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"GOLD PROMOTION SUMMARY")
    print(f"{'='*60}")
    print(f"Input:    {len(candidates)} candidates")
    print(f"Output:   {len(gold)} gold")
    print(f"Rejected: {len(rejected)} rows")
    print(f"\nFixes applied:")
    for fix, count in sorted(fixes.items()):
        print(f"  {fix:30s}: {count}")

    # Difficulty
    diffs = Counter(q["difficulty"] for q in gold)
    print(f"\nDifficulty: {dict(diffs)}")

    # Coverage
    domains = ["tax", "finance", "medical", "legal", "compliance"]
    ftypes = sorted(set(q["failure_type"] for q in gold))
    cells = Counter((q["domain"], q["failure_type"]) for q in gold)
    print(f"\nCoverage ({len(cells)} cells):")
    print(f"{'Domain':<12s}", end="")
    for ft in ftypes:
        print(f" {ft[:7]:>7s}", end="")
    print(f" {'TOTAL':>7s}")
    for d in domains:
        print(f"{d:<12s}", end="")
        row = 0
        for ft in ftypes:
            c = cells.get((d, ft), 0)
            row += c
            print(f" {c:>7d}", end="")
        print(f" {row:>7d}")

    # Final validation
    print(f"\n{'='*60}")
    print("GOLD VALIDATION")
    print(f"{'='*60}")
    dup_check = len(set(q["question"] for q in gold))
    overlap_check = sum(1 for q in gold if set(q["required_node_ids"]) & set(q["distractor_node_ids"]))
    signal_check = 0
    for q in gold:
        for m in re.finditer(r'contains the (?:signal|language) "([^"]+)"', q["gold_answer"]):
            sig = m.group(1)
            if not any(sig.lower() in ev.get("evidence_text", "").lower() for ev in q.get("gold_evidence", [])):
                signal_check += 1
                break
    giveaway_check = sum(1 for q in gold if BANNED_RE.search(q["question"]))
    short_ev_check = sum(1 for q in gold for ev in q["gold_evidence"] if len(ev.get("evidence_text", "")) < 50)
    mid_word_check = sum(1 for q in gold for ev in q["gold_evidence"]
                         if ev.get("evidence_text", "").rstrip() and ev["evidence_text"].rstrip()[-1] not in '.;:?!)"\'')

    checks = {
        "unique_questions": dup_check == len(gold),
        "no_overlap": overlap_check == 0,
        "no_signal_hallucination": signal_check == 0,
        "no_giveaways": giveaway_check == 0,
        "no_short_evidence": short_ev_check == 0,
        "no_mid_word_cuts": mid_word_check == 0,
        "all_have_evidence": all(len(q["gold_evidence"]) >= 1 for q in gold),
        "all_have_2+_required": all(len(q["required_node_ids"]) >= 2 for q in gold),
    }
    for check, passed in checks.items():
        print(f"  {check:35s}: {'PASS' if passed else 'FAIL'}")

    print(f"\nSaved: {GOLD_FILE}")
    print(f"Rejected: {REJECTED_FILE}")
    return gold


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
