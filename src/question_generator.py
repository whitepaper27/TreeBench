"""TreeBench Question Generator — Phase 4 of the pipeline.

Takes CandidateMatch objects from pattern hunters + the parsed TreeStore,
and generates fully-formed questions matching the v0 workbook 24-column schema.

Two modes:
1. Template-based: deterministic, no API needed, uses node text directly
2. LLM-powered: feeds subgraph context to Claude/GPT for natural-language questions

The template mode generates real questions grounded in actual regulation text.
"""

from __future__ import annotations
import json, re, hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from tree_node import TreeNode, TreeStore
from pattern_hunters import CandidateMatch


# ──────────────────────────────────────────────────────────────────────
# Domain mapping from source_id
# ──────────────────────────────────────────────────────────────────────
SOURCE_DOMAIN_MAP = {
    "ECFR_TITLE26_XML": ("tax", "Federal tax regulation"),
    "ECFR_TITLE12_XML": ("finance", "Banking regulation"),
    "ECFR_TITLE17_XML": ("finance", "Securities regulation"),
    "ECFR_TITLE21_XML": ("medical", "FDA regulation"),
    "ECFR_TITLE42_XML": ("medical", "Public health regulation"),
    "ECFR_TITLE29_XML": ("legal", "Labor regulation"),
    "ECFR_TITLE15_XML": ("legal", "Commerce regulation"),
    "ECFR_TITLE40_XML": ("compliance", "Environmental regulation"),
    "ECFR_TITLE45_XML": ("compliance", "HHS/HIPAA regulation"),
    "ECFR_TITLE31_XML": ("compliance", "Treasury/AML regulation"),
}

# ──────────────────────────────────────────────────────────────────────
# Question templates per failure type
# ──────────────────────────────────────────────────────────────────────
QUESTION_TEMPLATES = {
    "override_chain": {
        "template": 'Under {parent_path}, the general rule states: "{parent_text_short}". However, does a specific exception or override apply when considering {target_heading}?',
        "answer_template": 'Yes. While the general rule under {parent_path} provides that "{parent_text_short}", the specific provision at {target_path} states: "{target_text_short}". This exception overrides the parent rule.',
        "rag_fail": "RAG retrieves the broad parent rule ({parent_path}) based on cosine similarity to the query, but misses the child exception at {target_path} that actually controls the outcome.",
        "answer_type": "yes_no",
    },
    "scope_disambiguation": {
        "template": 'The term referenced in {target_heading} appears in multiple sections of the regulation. What is the specific meaning of this term as used in the context of {target_path}?',
        "answer_template": 'As used in {target_path}, the term is specifically defined as: "{target_text_short}". This differs from the same term used elsewhere in the regulation, which may carry a different scope or definition.',
        "rag_fail": "RAG retrieves the most common or general definition of the term, but the correct answer depends on the specific subtree context at {target_path}.",
        "answer_type": "extractive",
    },
    "cross_reference": {
        "template": 'Section {target_heading} references another provision. What does the cross-referenced section provide, and how does it affect the rule at {target_path}?',
        "answer_template": 'The provision at {target_path} cross-references {signal_text}. The referenced section provides: "{target_text_short}". This cross-reference modifies or qualifies the application of the original rule.',
        "rag_fail": "RAG retrieves the originating section but does not follow the cross-reference to {signal_text}. The answer requires traversing to a different subtree.",
        "answer_type": "extractive",
    },
    "conditional_cascade": {
        "template": 'What conditions must be satisfied to apply the rule at {target_path}? Consider all cascading conditions from parent provisions.',
        "answer_template": 'Application of the rule at {target_path} requires satisfying {signal_text}. The full condition chain traces from the parent provisions through: {required_path}. Each level adds an additional condition.',
        "rag_fail": "RAG retrieves the leaf-level rule text but misses the cascading conditions imposed by ancestor nodes. The answer requires walking the full condition chain.",
        "answer_type": "extractive",
    },
    "temporal_layering": {
        "template": 'What is the effective date or temporal condition governing the rule at {target_path}? Does a specific time-based provision modify the general rule?',
        "answer_template": 'The rule at {target_path} includes a temporal condition: {signal_text}. This means the rule applies differently depending on the relevant time period. The text states: "{target_text_short}".',
        "rag_fail": "RAG retrieves the current text of the rule but does not surface the temporal qualifier ({signal_text}) that determines whether the rule applies to a given time period.",
        "answer_type": "extractive",
    },
    "sibling_conflict": {
        "template": 'Within {parent_path}, parallel provisions appear to provide conflicting rules. Which provision controls, and how is the conflict resolved?',
        "answer_template": 'At {target_path}, the provision states: "{target_text_short}". This appears to conflict with a sibling provision under the same parent. The conflict is resolved by the more specific or later-enacted provision.',
        "rag_fail": "RAG retrieves the most semantically similar sibling section, which may state the opposite rule. Correct resolution requires understanding the structural relationship between parallel siblings.",
        "answer_type": "extractive",
    },
    "definitional_dependency": {
        "template": 'The rule at {target_path} uses a term that is defined elsewhere. What is the controlling definition, and where is it located?',
        "answer_template": 'The provision at {target_path} states: "{target_text_short}". This references a definition ({signal_text}). The controlling definition must be located in the referenced section to properly interpret the rule.',
        "rag_fail": "RAG retrieves the rule text containing the defined term but does not follow the definitional dependency ({signal_text}) to the section where the term is actually defined.",
        "answer_type": "extractive",
    },
    "aggregation": {
        "template": 'The rule at {target_path} requires combining information from multiple sections. What values or provisions must be aggregated to determine the correct outcome?',
        "answer_template": 'The provision at {target_path} requires aggregation ({signal_text}). The text states: "{target_text_short}". The correct answer requires collecting data from multiple branches of the regulatory tree.',
        "rag_fail": "RAG retrieves only one branch of the aggregation. The correct answer requires combining information from multiple subtrees referenced by {signal_text}.",
        "answer_type": "extractive",
    },
    "negative_space": {
        "template": 'Does the regulatory framework at {target_path} address {signal_text}? If not, what is the implication of this absence?',
        "answer_template": 'The regulatory subtree at {target_path} does not explicitly address this topic. {signal_text}. The absence of coverage means this scenario is either not regulated under this section or falls under a different regulatory framework.',
        "rag_fail": "RAG retrieves the nearest semantically similar section, producing a plausible-looking but incorrect answer. The correct answer is that the topic is NOT covered in this subtree — a negative finding that cosine similarity cannot detect.",
        "answer_type": "yes_no",
    },
    "depth_gated_specificity": {
        "template": 'What is the specific value, rate, or threshold provided at {target_path}? Does this differ from the general statement in the parent provision?',
        "answer_template": 'The specific provision at {target_path} provides: {signal_text}. The full text states: "{target_text_short}". This specific value is only found at the leaf level and is not stated in the parent summary.',
        "rag_fail": "RAG retrieves the parent-level summary which gives the general rule but omits the specific value ({signal_text}). The precise number/rate/threshold is only available at tree depth {tree_depth}.",
        "answer_type": "extractive",
    },
}

# Difficulty classification
DIFFICULTY_RULES = {
    # (min_depth, min_required_nodes, has_cross_ref) -> difficulty
    "easy": lambda d, n, x: d <= 4 and n <= 2,
    "medium": lambda d, n, x: d <= 6 and n <= 4,
    "hard": lambda d, n, x: d > 6 or n > 4 or x,
}


@dataclass
class TreeBenchQuestion:
    """One question in the TreeBench dataset — matches v0 workbook 24-column schema."""
    id: str
    domain: str
    subdomain: str
    failure_type_primary: str
    failure_type_secondary: str
    question: str
    correct_answer: str
    answer_type: str
    gold_evidence: str  # JSON string of evidence list
    tree_path: str
    required_nodes: str  # JSON string of node ID list
    distractor_nodes: str  # JSON string of node ID list
    tree_depth: int
    jurisdiction: str
    effective_date: str
    source_doc_type: str
    difficulty: str
    reasoning_steps_required: int
    why_tree_structure_matters: str
    expected_rag_failure_pattern: str
    evaluation_target: str
    release_status: str
    review_decision: str
    reviewer_notes: str


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _classify_difficulty(depth: int, num_required: int, has_cross_ref: bool) -> tuple[str, int]:
    """Return (difficulty, reasoning_steps_required)."""
    if depth <= 4 and num_required <= 2:
        return "easy", 2
    elif depth <= 6 and num_required <= 4:
        return "medium", 3
    else:
        return "hard", max(4, num_required)


def generate_question(candidate: CandidateMatch, store: TreeStore,
                      counter: int) -> Optional[TreeBenchQuestion]:
    """Generate a TreeBench question from a candidate match and its tree context."""
    target = store.get(candidate.target_node_id)
    if not target:
        return None

    # Get domain info
    domain, subdomain = SOURCE_DOMAIN_MAP.get(
        candidate.source_id, ("unknown", "unknown")
    )

    # Get template for this failure type
    tmpl = QUESTION_TEMPLATES.get(candidate.failure_type)
    if not tmpl:
        return None

    # Build context variables
    ancestors = store.ancestors(candidate.target_node_id)
    parent = ancestors[-1] if ancestors else target
    parent_path = parent.path if parent else ""
    parent_text_short = _truncate(parent.text, 200) if parent else ""
    target_text_short = _truncate(target.text, 300)
    target_heading = target.heading or f"{target.node_type} {target.number}"
    required_path = " > ".join(
        f"{store.get(nid).node_type} {store.get(nid).number}"
        for nid in candidate.required_node_ids
        if store.get(nid)
    )

    # Format templates
    fmt_vars = {
        "parent_path": parent_path,
        "parent_text_short": parent_text_short,
        "target_path": target.path,
        "target_heading": target_heading,
        "target_text_short": target_text_short,
        "signal_text": candidate.signal_text,
        "required_path": required_path,
        "tree_depth": candidate.tree_depth,
    }

    try:
        question_text = tmpl["template"].format(**fmt_vars)
        answer_text = tmpl["answer_template"].format(**fmt_vars)
        rag_fail_text = tmpl["rag_fail"].format(**fmt_vars)
    except (KeyError, IndexError):
        return None

    # Skip if question is too short or empty
    if len(question_text) < 30 or len(answer_text) < 30:
        return None

    # Classify difficulty
    has_xref = bool(target.cross_refs)
    difficulty, reasoning_steps = _classify_difficulty(
        candidate.tree_depth, len(candidate.required_node_ids), has_xref
    )

    # Build gold evidence
    evidence = []
    for nid in candidate.required_node_ids[:3]:
        node = store.get(nid)
        if node:
            evidence.append({
                "source_doc_type": "eCFR",
                "source_name": candidate.source_id,
                "section_id": node.number or node.id,
                "evidence": _truncate(node.text, 200),
            })

    # Generate stable ID
    failure_abbrev = candidate.failure_type.upper().replace("_", "-")
    domain_abbrev = domain.upper()[:3]
    id_str = f"TB-{domain_abbrev}-{failure_abbrev}-{counter:04d}"

    # Detect secondary failure type from co-occurring signals
    secondary = ""
    if target.cross_refs and candidate.failure_type != "cross_reference":
        secondary = "cross_reference"
    elif candidate.tree_depth > 5 and candidate.failure_type != "depth_gated_specificity":
        secondary = "depth_gated_specificity"

    return TreeBenchQuestion(
        id=id_str,
        domain=domain,
        subdomain=subdomain,
        failure_type_primary=candidate.failure_type,
        failure_type_secondary=secondary,
        question=question_text,
        correct_answer=answer_text,
        answer_type=tmpl["answer_type"],
        gold_evidence=json.dumps(evidence, ensure_ascii=False),
        tree_path=target.path,
        required_nodes=json.dumps(candidate.required_node_ids[:5]),
        distractor_nodes=json.dumps(candidate.distractor_node_ids[:5]),
        tree_depth=candidate.tree_depth,
        jurisdiction="US federal",
        effective_date="2026-01-01",
        source_doc_type="eCFR",
        difficulty=difficulty,
        reasoning_steps_required=reasoning_steps,
        why_tree_structure_matters=rag_fail_text,
        expected_rag_failure_pattern=QUESTION_TEMPLATES[candidate.failure_type]["rag_fail"].split(".")[0] + ".",
        evaluation_target="answer_accuracy_and_path_accuracy",
        release_status="GENERATED",
        review_decision="",
        reviewer_notes="",
    )


def generate_from_candidates_file(candidates_path: str, tree_path: str,
                                   source_id: str) -> list[TreeBenchQuestion]:
    """Load candidates + tree, generate questions for all candidates."""
    store = TreeStore.load(tree_path)

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_by_type = json.load(f)

    questions: list[TreeBenchQuestion] = []
    counter = 0

    for failure_type, candidate_dicts in candidates_by_type.items():
        for cd in candidate_dicts:
            candidate = CandidateMatch(**cd)
            counter += 1
            q = generate_question(candidate, store, counter)
            if q:
                questions.append(q)

    return questions


if __name__ == "__main__":
    import sys, os
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parsed_dir = "../data/parsed"
    q_dir = "../data/questions"
    out_dir = "../data/generated"
    os.makedirs(out_dir, exist_ok=True)

    titles = [
        ("ECFR-title12", "ECFR_TITLE12_XML"),
        ("ECFR-title15", "ECFR_TITLE15_XML"),
        ("ECFR-title17", "ECFR_TITLE17_XML"),
        ("ECFR-title21", "ECFR_TITLE21_XML"),
        ("ECFR-title26", "ECFR_TITLE26_XML"),
        ("ECFR-title29", "ECFR_TITLE29_XML"),
        ("ECFR-title31", "ECFR_TITLE31_XML"),
        ("ECFR-title40", "ECFR_TITLE40_XML"),
        ("ECFR-title42", "ECFR_TITLE42_XML"),
        ("ECFR-title45", "ECFR_TITLE45_XML"),
    ]

    all_questions: list[TreeBenchQuestion] = []

    for name, source_id in titles:
        tree_file = f"{parsed_dir}/{name}_tree.json"
        cand_file = f"{q_dir}/{name}_candidates.json"

        if not os.path.exists(tree_file) or not os.path.exists(cand_file):
            print(f"SKIP {name} — files not found")
            continue

        print(f"Generating questions for {name}...", flush=True)
        qs = generate_from_candidates_file(cand_file, tree_file, source_id)
        all_questions.extend(qs)
        print(f"  Generated {len(qs)} questions")

    # Save all questions
    out_path = f"{out_dir}/treebench_all_generated.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in all_questions], f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_questions)} questions saved to {out_path}")

    # Print distribution
    from collections import Counter
    domain_counts = Counter(q.domain for q in all_questions)
    type_counts = Counter(q.failure_type_primary for q in all_questions)
    diff_counts = Counter(q.difficulty for q in all_questions)

    print(f"\nBy domain:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:15s}: {c}")
    print(f"\nBy failure type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:30s}: {c}")
    print(f"\nBy difficulty:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d:10s}: {c}")
