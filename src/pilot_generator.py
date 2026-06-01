"""Pilot Generator — produce 100 high-quality TreeBench questions.

5 domains x 10 failure types x 2 questions = 100

Uses validated candidates with full context packages.
Generates scenario-based questions grounded in actual regulation text.
Synthesizes rag_likely_answer from parent/sibling text.
"""

from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_schema import (
    TreeBenchQuestion, ContextPackage, CONFOUNDER_MAP,
    SOURCE_META, DOMAINS, FAILURE_TYPES,
)


PILOT_PER_CELL = 2  # 5 domains x 10 types x 2 = 100


def _truncate(text: str, n: int = 200) -> str:
    t = text.replace("\n", " ").strip()
    return t[:n] + "..." if len(t) > n else t


def _classify_difficulty(ctx: ContextPackage) -> str:
    if ctx.target_tree_depth <= 4 and len(ctx.ancestral_context) <= 3:
        return "easy"
    elif ctx.target_tree_depth <= 6 and len(ctx.ancestral_context) <= 5:
        return "medium"
    return "hard"


def _classify_answer_type(failure_type: str) -> str:
    if failure_type in ("override_chain", "negative_space"):
        return "yes_no"
    elif failure_type in ("depth_gated_specificity", "aggregation"):
        return "numeric"
    elif failure_type in ("scope_disambiguation", "sibling_conflict"):
        return "classification"
    elif failure_type in ("cross_reference", "definitional_dependency"):
        return "section_reference"
    else:
        return "multi_part"


# ──────────────────────────────────────────────────────────────────────
# Scenario-based question generation per failure type
# ──────────────────────────────────────────────────────────────────────

def _gen_override_chain(ctx: ContextPackage) -> dict:
    parent_text = _truncate(ctx.parent_rule_text, 150)
    target_text = _truncate(ctx.target_node_text, 300)
    heading = ctx.target_node_heading

    question = (
        f"A compliance officer reviewing {ctx.source_title} encounters the general rule "
        f'under {ctx.ancestral_context[-1]["heading"] if ctx.ancestral_context else "the parent provision"}, '
        f'which states: "{parent_text}". '
        f"Is this general rule sufficient to determine the outcome, or does {heading} "
        f"introduce an exception that changes the analysis?"
    )

    gold_answer = (
        f"The general rule alone is insufficient. The provision at {ctx.target_tree_path} states: "
        f'"{target_text}". This contains the signal "{ctx.signal_text}" which indicates an override '
        f"of the parent rule. The correct analysis requires walking the full path: "
        f"{' > '.join(ctx.gold_path)} to reach the controlling exception."
    )

    rag_likely = (
        f"A similarity-based retrieval system would return the parent provision: "
        f'"{parent_text}". This appears directly responsive to the query and would lead to '
        f"an answer based solely on the general rule, missing the exception at depth {ctx.target_tree_depth}."
    )

    why_fails = (
        f"The parent node and the exception child share high vocabulary overlap "
        f"(confounder score: {ctx.confounder_score:.2f}), making them near-identical "
        f"in embedding space. Cosine similarity cannot distinguish the general rule from "
        f"its exception — only the tree position (parent vs. child) resolves the correct answer."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_scope_disambiguation(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)
    sibling_text = _truncate(ctx.sibling_context[0]["text_preview"], 150) if ctx.sibling_context else ""

    question = (
        f"Under {ctx.source_title}, the term referenced in {ctx.target_node_heading} "
        f"appears in multiple regulatory subtrees. A practitioner working within the scope "
        f"of {ctx.gold_path[-2] if len(ctx.gold_path) > 1 else 'this section'} needs "
        f"to determine the applicable definition. Which definition controls in this context?"
    )

    gold_answer = (
        f"The controlling definition is found at {ctx.target_tree_path}: "
        f'"{target_text}". This definition is scoped to the current subtree. '
        f"Other definitions of the same term exist in sibling branches but carry "
        f"different meanings in their respective contexts."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the most frequently occurring or "
        f"longest definition of the term, likely from a sibling subtree: "
        f'"{sibling_text}". This definition may be semantically similar but is '
        f"structurally scoped to a different regulatory context."
    )

    why_fails = (
        f"Both definitions use nearly identical vocabulary (the same defined term), "
        f"producing high cosine similarity. The correct definition depends entirely "
        f"on which subtree the query falls within — a positional distinction that "
        f"embedding similarity cannot encode."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_cross_reference(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)
    refs = ", ".join(ctx.cross_refs[:3]) if ctx.cross_refs else ctx.signal_text

    question = (
        f"A regulatory analyst reviewing {ctx.target_node_heading} under "
        f"{ctx.source_title} notices it references {refs}. What does the "
        f"cross-referenced provision provide, and how does it modify the "
        f"rule at {ctx.gold_path[-1] if ctx.gold_path else 'this section'}?"
    )

    gold_answer = (
        f"The provision at {ctx.target_tree_path} states: \"{target_text}\". "
        f"It explicitly cross-references {refs}, which means the answer "
        f"requires traversing to a different subtree to retrieve the referenced "
        f"provision's content. The full resolution path is: {' > '.join(ctx.gold_path)}."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the source section containing "
        f"the cross-reference text, but would NOT follow the link to {refs}. "
        f'The retrieved context would be: "{_truncate(ctx.parent_rule_text, 150)}", '
        f"which mentions but does not include the referenced provision's content."
    )

    why_fails = (
        f"The cross-reference ({refs}) is a structural pointer — it links two "
        f"nodes in different subtrees. Cosine similarity treats the reference "
        f"text as content rather than as a traversal instruction. The retrieval "
        f"system fetches the node containing the reference, not the node being referenced."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_conditional_cascade(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)
    ancestor_chain = " > ".join(
        a["heading"] or f'{a["node_type"]} {a["number"]}' for a in ctx.ancestral_context[-3:]
    )

    question = (
        f"To determine whether the rule at {ctx.target_node_heading} applies "
        f"under {ctx.source_title}, what conditions must be satisfied? "
        f"Consider the full chain of conditions from parent provisions through "
        f"{ancestor_chain}."
    )

    gold_answer = (
        f"Application requires satisfying {ctx.signal_text}. The condition chain "
        f"traces through: {' > '.join(ctx.gold_path)}. The target provision states: "
        f'"{target_text}". Each ancestor level imposes an additional gating condition '
        f"that must be met before the leaf rule applies."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the leaf-level rule text at "
        f"depth {ctx.target_tree_depth}, which appears to state the rule directly. "
        f'The retrieved text: "{_truncate(ctx.parent_rule_text, 150)}" omits the '
        f"cascading conditions imposed by ancestor nodes at depths "
        f"{', '.join(str(a.get('node_type', '')) for a in ctx.ancestral_context[-3:])}."
    )

    why_fails = (
        f"The leaf node's text is semantically most relevant to the query, but "
        f"the answer is incomplete without the gating conditions from "
        f"{len(ctx.ancestral_context)} ancestor levels. Flat retrieval cannot "
        f"reconstruct the condition chain that the tree structure encodes."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_temporal_layering(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)

    question = (
        f"Under {ctx.source_title}, the provision at {ctx.target_node_heading} "
        f"includes a temporal qualifier. For a transaction occurring in the current "
        f"tax/regulatory year, does the general rule apply, or has a time-specific "
        f"amendment modified the outcome?"
    )

    gold_answer = (
        f"The provision at {ctx.target_tree_path} contains the temporal condition: "
        f'"{ctx.signal_text}". The full text states: "{target_text}". This means '
        f"the rule's application depends on the relevant time period. The correct "
        f"answer requires identifying this temporal qualifier at depth {ctx.target_tree_depth}."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the substantive rule text, which "
        f"discusses the same topic. However, it would surface the text without "
        f'highlighting the temporal qualifier ("{ctx.signal_text}"), leading to '
        f"an answer that assumes the rule applies uniformly across all time periods."
    )

    why_fails = (
        f"The temporal qualifier is embedded within text that is otherwise "
        f"semantically identical to the general rule. The date condition "
        f'("{ctx.signal_text}") does not change the embedding significantly '
        f"because dates carry low semantic weight relative to substantive legal terms."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_sibling_conflict(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)
    sibling_text = _truncate(ctx.sibling_context[0]["text_preview"], 200) if ctx.sibling_context else ""

    question = (
        f"Under {ctx.source_title}, two parallel provisions appear under the "
        f"same parent section. A practitioner finds that {ctx.target_node_heading} "
        f'states one rule, while a sibling provision states: "{sibling_text}". '
        f"Which provision controls, and how is the apparent conflict resolved?"
    )

    gold_answer = (
        f"The provision at {ctx.target_tree_path} states: \"{target_text}\". "
        f"This provision contains the signal \"{ctx.signal_text}\" indicating it "
        f"qualifies or restricts the sibling rule. Resolution requires examining "
        f"both siblings at the same tree level under their common parent."
    )

    rag_likely = (
        f"Similarity-based retrieval would return whichever sibling section "
        f"has higher keyword overlap with the query. With sibling provisions "
        f"sharing the same parent context, the wrong sibling may rank higher. "
        f'The returned text: "{sibling_text}" may state the opposite conclusion.'
    )

    why_fails = (
        f"Sibling nodes share their entire ancestral context, making their "
        f"embeddings nearly identical. The distinguishing information — which "
        f"sibling contains the exception vs. the general rule — is encoded only "
        f"in their relative tree position, not in their semantic content."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_definitional_dependency(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)
    refs = ", ".join(ctx.cross_refs[:2]) if ctx.cross_refs else ctx.signal_text

    question = (
        f"The rule at {ctx.target_node_heading} in {ctx.source_title} uses a "
        f"term that is defined elsewhere ({ctx.signal_text}). What is the controlling "
        f"definition, and how does it affect the application of this rule?"
    )

    gold_answer = (
        f"The provision at {ctx.target_tree_path} states: \"{target_text}\". "
        f"It contains a definitional dependency ({ctx.signal_text}), meaning "
        f"the term's meaning must be imported from the referenced definition section. "
        f"Without the correct definition, the rule's scope is ambiguous."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the rule text that USES the "
        f"defined term, but would not follow the definitional chain to where "
        f'the term is actually defined. The retrieved text: "{_truncate(ctx.parent_rule_text, 150)}" '
        f"appears self-contained but relies on an external definition."
    )

    why_fails = (
        f"The rule text and the definition text are in different subtrees. "
        f"The query matches the rule (where the term is used) with higher similarity "
        f"than the definition (where the term is defined). Flat retrieval returns "
        f"usage, not definition — the structural dependency is invisible."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_aggregation(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)

    question = (
        f"The provision at {ctx.target_node_heading} in {ctx.source_title} "
        f"requires combining information from multiple sections ({ctx.signal_text}). "
        f"What values or provisions must be aggregated to determine the correct outcome?"
    )

    gold_answer = (
        f"The provision at {ctx.target_tree_path} states: \"{target_text}\". "
        f"The aggregation signal \"{ctx.signal_text}\" indicates that the correct "
        f"answer requires data from multiple branches of the regulatory tree. "
        f"A single retrieval cannot produce the complete answer."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the node containing the "
        f"aggregation instruction, but would NOT retrieve the separate branches "
        f'whose values must be combined. The retrieved text mentions "{ctx.signal_text}" '
        f"but omits the referenced values from other subtrees."
    )

    why_fails = (
        f"Aggregation requires information from 2+ separate subtrees. Cosine "
        f"similarity retrieves the single most relevant chunk, which is the "
        f"aggregation node itself. The component values exist in structurally "
        f"distant nodes that don't individually match the query semantics."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_negative_space(ctx: ContextPackage) -> dict:
    question = (
        f"Under {ctx.source_title}, does the regulatory framework at "
        f"{ctx.target_node_heading} address the scenario described by "
        f'"{ctx.signal_text}"? If not, what is the regulatory implication?'
    )

    gold_answer = (
        f"The regulatory subtree at {ctx.target_tree_path} does NOT explicitly "
        f"address this scenario. Analysis of the child nodes reveals: {ctx.signal_text}. "
        f"The absence of coverage means this scenario is either unregulated under "
        f"this section or falls under a different regulatory framework entirely."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the NEAREST semantically "
        f"similar section, producing a confident but incorrect answer. The "
        f"retrieved text would discuss related topics under the same parent, "
        f"leading to a false-positive response that assumes coverage exists."
    )

    why_fails = (
        f"Negative space — the absence of a matching node — cannot be detected "
        f"by cosine similarity. Similarity search always returns SOMETHING; it "
        f"cannot return 'not found'. Only tree traversal can confirm that an "
        f"exhaustive walk through the relevant subtree yields no matching child."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


def _gen_depth_gated(ctx: ContextPackage) -> dict:
    target_text = _truncate(ctx.target_node_text, 300)

    question = (
        f"Under {ctx.source_title}, what is the specific value, rate, or threshold "
        f"provided at {ctx.target_node_heading}? The parent provision gives a general "
        f"rule — does the specific value differ from the general statement?"
    )

    gold_answer = (
        f"The specific provision at {ctx.target_tree_path} provides: "
        f"\"{ctx.signal_text}\". The full text states: \"{target_text}\". "
        f"This specific value is ONLY available at tree depth {ctx.target_tree_depth} "
        f"and is NOT stated in the parent-level summary."
    )

    rag_likely = (
        f"Similarity-based retrieval would return the parent-level summary "
        f'"{_truncate(ctx.parent_rule_text, 150)}" which describes the general rule '
        f"but omits the specific value ({ctx.signal_text}). The parent text is "
        f"longer, more general, and ranks higher by semantic similarity to the query."
    )

    why_fails = (
        f"The parent summary and the leaf-level specific value discuss the same "
        f"topic with high vocabulary overlap (confounder score: {ctx.confounder_score:.2f}). "
        f"The parent node's text is typically longer and more keyword-rich, "
        f"causing it to rank above the leaf in cosine similarity despite lacking "
        f"the precise value the question asks for."
    )

    return {"question": question, "gold_answer": gold_answer,
            "rag_likely_answer": rag_likely, "why_similarity_fails": why_fails}


GENERATORS = {
    "override_chain": _gen_override_chain,
    "scope_disambiguation": _gen_scope_disambiguation,
    "cross_reference": _gen_cross_reference,
    "conditional_cascade": _gen_conditional_cascade,
    "temporal_layering": _gen_temporal_layering,
    "sibling_conflict": _gen_sibling_conflict,
    "definitional_dependency": _gen_definitional_dependency,
    "aggregation": _gen_aggregation,
    "negative_space": _gen_negative_space,
    "depth_gated_specificity": _gen_depth_gated,
}


def generate_pilot(validated_path: str, output_dir: str) -> list[TreeBenchQuestion]:
    """Generate 100-question pilot from validated candidates."""
    with open(validated_path, "r", encoding="utf-8") as f:
        ranked_cells = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    questions: list[TreeBenchQuestion] = []
    counters: dict[str, int] = defaultdict(int)
    coverage_matrix: dict[tuple[str, str], int] = {}

    for domain in DOMAINS:
        for ftype in FAILURE_TYPES:
            cell_key = f"{domain}/{ftype}"
            candidates = ranked_cells.get(cell_key, [])

            if not candidates:
                print(f"  WARNING: No candidates for {cell_key}")
                coverage_matrix[(domain, ftype)] = 0
                continue

            generated = 0
            for entry in candidates[:PILOT_PER_CELL + 2]:  # try extra in case of failures
                if generated >= PILOT_PER_CELL:
                    break

                ctx = ContextPackage(**entry["context"])
                gen_fn = GENERATORS.get(ftype)
                if not gen_fn:
                    continue

                try:
                    result = gen_fn(ctx)
                except Exception as e:
                    print(f"  ERROR generating {cell_key}: {e}")
                    continue

                if not result.get("question") or len(result["question"]) < 30:
                    continue

                counters[ftype] += 1
                q_id = f"TB-{ctx.candidate_source_id.replace('ECFR_', '').replace('_XML', '')}-{ftype.upper().replace('_', '-')}-{counters[ftype]:04d}"

                q = TreeBenchQuestion(
                    question_id=q_id,
                    domain=domain,
                    source_title=ctx.source_title,
                    failure_type=ftype,
                    structural_confounder_type=ctx.structural_confounder_type,
                    question=result["question"],
                    gold_answer=result["gold_answer"],
                    rag_likely_answer=result["rag_likely_answer"],
                    answer_type=_classify_answer_type(ftype),
                    required_node_ids=[ctx.target_node_id],
                    distractor_node_ids=[s["node_id"] for s in ctx.sibling_context[:3]],
                    gold_path=ctx.gold_path,
                    gold_evidence=[{
                        "source": ctx.candidate_source_id,
                        "section_id": ctx.target_node_number,
                        "evidence_text": ctx.target_node_text[:300],
                    }],
                    why_similarity_fails=result["why_similarity_fails"],
                    tree_depth=ctx.target_tree_depth,
                    difficulty=_classify_difficulty(ctx),
                    review_status="draft_needs_validation",
                )
                questions.append(q)
                generated += 1

            coverage_matrix[(domain, ftype)] = generated

    # Save pilot dataset
    pilot_path = os.path.join(output_dir, "treebench_pilot_100.json")
    with open(pilot_path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2, ensure_ascii=False)

    # Save review manifest
    manifest = []
    for q in questions:
        manifest.append({
            "question_id": q.question_id,
            "domain": q.domain,
            "failure_type": q.failure_type,
            "confounder_type": q.structural_confounder_type,
            "question_preview": q.question[:150],
            "gold_answer_preview": q.gold_answer[:150],
            "rag_likely_answer_preview": q.rag_likely_answer[:150],
            "tree_depth": q.tree_depth,
            "difficulty": q.difficulty,
            "review_status": q.review_status,
            "reviewer_notes": "",
        })
    manifest_path = os.path.join(output_dir, "review_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PILOT GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(questions)}")
    print(f"Target: {len(DOMAINS) * len(FAILURE_TYPES) * PILOT_PER_CELL}")
    print(f"\nCoverage matrix:")
    print(f"{'Domain':<15s}", end="")
    for ft in FAILURE_TYPES:
        print(f" {ft[:8]:>8s}", end="")
    print(f" {'TOTAL':>8s}")
    for domain in DOMAINS:
        print(f"{domain:<15s}", end="")
        row_total = 0
        for ft in FAILURE_TYPES:
            count = coverage_matrix.get((domain, ft), 0)
            row_total += count
            print(f" {count:>8d}", end="")
        print(f" {row_total:>8d}")

    print(f"\nSaved: {pilot_path}")
    print(f"Manifest: {manifest_path}")

    return questions


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    validated_path = os.path.join(os.path.dirname(__file__), "..", "data", "validated", "validated_candidates.json")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pilot")

    if not os.path.exists(validated_path):
        print("Run candidate_validator.py first!")
        sys.exit(1)

    questions = generate_pilot(validated_path, output_dir)
