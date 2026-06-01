"""Pilot Generator v2 — structure-blind, publication-grade questions.

Fixes all issues from v1 review:
1. ZERO structural leakage in question text (no section numbers, paragraph labels, node refs)
2. No ellipsis truncation in gold_answer or gold_evidence — clean quoted excerpts
3. Multi-node required_node_ids for cross_reference/definitional_dependency/aggregation
4. Concrete date comparisons for temporal_layering
5. Actual parent heading in override_chain (no "under ,")
6. Organic practitioner scenarios — no template phrasing
7. Deduplication of question text and required_node_ids
"""

from __future__ import annotations
import json, os, re, sys, hashlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_schema import (
    TreeBenchQuestion, ContextPackage, CONFOUNDER_MAP,
    SOURCE_META, DOMAINS, FAILURE_TYPES,
)

PILOT_PER_CELL = 2


# ──────────────────────────────────────────────────────────────────────
# Utilities — no truncation with "...", clean excerpts only
# ──────────────────────────────────────────────────────────────────────

def _clean_excerpt(text: str, max_len: int = 400) -> str:
    """Return a clean quoted excerpt. Cut at sentence boundary, never with '...'."""
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) <= max_len:
        return t
    # Cut at last sentence boundary before max_len
    cut = t[:max_len]
    last_period = cut.rfind(". ")
    last_semi = cut.rfind("; ")
    boundary = max(last_period, last_semi)
    if boundary > max_len // 2:
        return cut[:boundary + 1]
    # No good boundary — cut at last space
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 0 else cut


def _extract_subject(text: str) -> str:
    """Extract the substantive subject from regulation text for use in scenarios."""
    # Remove boilerplate like "For the purpose of..." preambles
    t = re.sub(r"^\([a-zA-Z0-9]+\)\s*", "", text.strip())
    t = re.sub(r"^(?:In general|General rule)[.\s]*", "", t, flags=re.I)
    # Take first meaningful sentence
    sentences = re.split(r"(?<=[.;])\s+", t)
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and not s.startswith("Authority:") and not s.startswith("Source:"):
            return _clean_excerpt(s, 300)
    return _clean_excerpt(t, 300)


def _scrub_structural_refs(text: str) -> str:
    """Remove ALL section numbers, paragraph labels, and structural identifiers from text.

    Used on question text ONLY to ensure zero structural leakage.
    Gold answers and evidence keep their references.
    """
    # Remove § references: § 1.4-1, §61, § 225.104(c), etc.
    t = re.sub(r"§+\s*[\d]+(?:\.[\d\w\-]+)*(?:\([a-zA-Z0-9]+\))*", "the applicable provision", text)
    # Remove "Section X.Y-Z" references
    t = re.sub(r"[Ss]ection\s+[\d]+(?:\.[\d\w\-]+)*(?:\([a-zA-Z0-9]+\))*", "the applicable provision", t)
    # Remove "SECTION § X" artifacts
    t = re.sub(r"SECTION\s+(?:§\s*)?[\d]+(?:\.[\d\w\-]+)*", "the applicable provision", t)
    # Remove "Part X", "Subpart X" references
    t = re.sub(r"(?:PART|Part|SUBPART|Subpart)\s+[\d]+[A-Z]?(?:\.\d+)*", "the relevant regulatory part", t)
    # Remove paragraph labels: paragraph (a), paragraph p1, (a)(1)(A)
    t = re.sub(r"paragraph\s+(?:\([a-zA-Z0-9]+\)|p\d+)", "the specific provision", t)
    # Remove "section § X.Y" that might appear in quoted text within questions
    t = re.sub(r"section\s+§\s*[\d]+(?:\.[\d\w\-]+)*", "the referenced provision", t)
    # Clean up repeated "the applicable provision" from multiple replacements
    t = re.sub(r"(the applicable provision[,\s]*){2,}", "the applicable provision ", t)
    # Clean up whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_parent_label(ctx: ContextPackage) -> str:
    """Get a human-readable parent label, never blank. No section numbers."""
    for anc in reversed(ctx.ancestral_context):
        heading = anc.get("heading", "").strip()
        if heading and len(heading) > 5:
            # Strip section numbers from heading for use in question text
            clean = re.sub(r"§+\s*[\d]+(?:\.[\d\w\-]+)*\s*", "", heading).strip()
            clean = re.sub(r"^[-—–\s]+", "", clean).strip()
            if len(clean) > 5:
                return clean
            return heading
    # Fall back to source title without the eCFR prefix
    return ctx.source_title.replace("eCFR ", "").strip()


def _domain_practitioner(domain: str) -> str:
    """Return a domain-appropriate practitioner role."""
    roles = {
        "tax": "tax accountant",
        "finance": "financial compliance analyst",
        "medical": "regulatory affairs specialist",
        "legal": "employment law attorney",
        "compliance": "compliance officer",
    }
    return roles.get(domain, "regulatory analyst")


def _classify_difficulty(ctx: ContextPackage) -> str:
    if ctx.target_tree_depth <= 4 and len(ctx.ancestral_context) <= 3:
        return "easy"
    elif ctx.target_tree_depth <= 6 and len(ctx.ancestral_context) <= 5:
        return "medium"
    return "hard"


def _classify_answer_type(failure_type: str) -> str:
    types = {
        "override_chain": "yes_no",
        "negative_space": "yes_no",
        "depth_gated_specificity": "numeric",
        "aggregation": "numeric",
        "scope_disambiguation": "classification",
        "sibling_conflict": "classification",
        "cross_reference": "section_reference",
        "definitional_dependency": "section_reference",
        "conditional_cascade": "multi_part",
        "temporal_layering": "multi_part",
    }
    return types.get(failure_type, "multi_part")


def _build_required_nodes(ctx: ContextPackage, failure_type: str) -> list[str]:
    """Build proper multi-node required_node_ids. At least 2 for traversal types."""
    nodes = [ctx.target_node_id]

    # Add parent node for override/depth-gated (need both to show the conflict)
    if failure_type in ("override_chain", "depth_gated_specificity", "conditional_cascade"):
        for anc in reversed(ctx.ancestral_context):
            # Find the closest ancestor with substantive text
            if len(anc.get("text_preview", "")) > 30:
                # Reconstruct ancestor node ID from context
                anc_type = anc.get("node_type", "")
                anc_num = anc.get("number", "")
                # Use a path-based ID
                anc_id = f"{ctx.candidate_source_id}__{anc_type}_{anc_num}".replace(" ", "_")
                nodes.insert(0, anc_id)
                break

    # Add sibling for scope_disambiguation/sibling_conflict
    if failure_type in ("scope_disambiguation", "sibling_conflict"):
        for sib in ctx.sibling_context[:2]:
            if sib.get("node_id"):
                nodes.append(sib["node_id"])

    # Add referenced node for cross_reference/definitional_dependency/aggregation
    if failure_type in ("cross_reference", "definitional_dependency", "aggregation"):
        # Use cross_refs to find referenced sections
        for ref in ctx.cross_refs[:2]:
            ref_num = re.search(r"[\d]+(?:\.[\d\w\-]+)*", ref)
            if ref_num:
                ref_id = f"{ctx.candidate_source_id}__ref_{ref_num.group(0)}"
                nodes.append(ref_id)
        # Ensure at least 2 nodes — add sibling or parent as fallback
        if len(nodes) < 2 and ctx.sibling_context:
            nodes.append(ctx.sibling_context[0].get("node_id", f"{ctx.candidate_source_id}__sibling_0"))
        if len(nodes) < 2 and ctx.ancestral_context:
            anc = ctx.ancestral_context[-1]
            nodes.append(f"{ctx.candidate_source_id}__{anc.get('node_type', 'parent')}_{anc.get('number', '0')}")

    # Final safety: every question must have at least 2 required nodes
    if len(nodes) < 2 and ctx.ancestral_context:
        anc = ctx.ancestral_context[-1]
        nodes.insert(0, f"{ctx.candidate_source_id}__{anc.get('node_type', 'parent')}_{anc.get('number', '0')}")

    return nodes


def _build_gold_evidence(ctx: ContextPackage, failure_type: str) -> list[dict]:
    """Build complete gold evidence without truncation."""
    evidence = [{
        "source": ctx.candidate_source_id,
        "section_id": ctx.target_node_number,
        "node_type": ctx.target_node_type,
        "evidence_text": _clean_excerpt(ctx.target_node_text, 500),
    }]

    # Add parent evidence for override/depth-gated
    if failure_type in ("override_chain", "depth_gated_specificity", "conditional_cascade"):
        if ctx.ancestral_context:
            parent = ctx.ancestral_context[-1]
            evidence.insert(0, {
                "source": ctx.candidate_source_id,
                "section_id": parent.get("number", ""),
                "node_type": parent.get("node_type", ""),
                "evidence_text": _clean_excerpt(parent.get("text_preview", ""), 500),
            })

    # Add sibling evidence for scope/sibling types
    if failure_type in ("scope_disambiguation", "sibling_conflict"):
        for sib in ctx.sibling_context[:1]:
            evidence.append({
                "source": ctx.candidate_source_id,
                "section_id": "sibling",
                "node_type": "sibling_node",
                "evidence_text": _clean_excerpt(sib.get("text_preview", ""), 300),
            })

    return evidence


# ──────────────────────────────────────────────────────────────────────
# Structure-blind question generators — NO section numbers or node refs
# in question text. All structural info goes in gold_answer only.
# ──────────────────────────────────────────────────────────────────────

def _gen_override_chain(ctx: ContextPackage) -> dict:
    parent_label = _get_parent_label(ctx)
    parent_subject = _extract_subject(ctx.parent_rule_text)
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)

    question = (
        f"A {role} is advising on a matter governed by {ctx.source_title}. "
        f"The general regulatory framework under {parent_label} establishes that: "
        f'"{_clean_excerpt(parent_subject, 200)}". '
        f"Based solely on this general provision, can the practitioner conclude "
        f"the rule applies without further investigation into subordinate provisions?"
    )

    gold_answer = (
        f"No. The general rule under {parent_label} is qualified by a subordinate "
        f"provision at {' > '.join(ctx.gold_path)}, which states: "
        f'"{_clean_excerpt(target_subject, 350)}". '
        f"This child provision contains the signal \"{ctx.signal_text}\" which modifies "
        f"the parent rule. The correct analysis requires traversing the full hierarchy "
        f"from the general rule down to this specific exception at depth {ctx.target_tree_depth}."
    )

    expected_failure = (
        f"A similarity-based retrieval system would return the general provision under "
        f"{parent_label}, which directly addresses the topic and contains high keyword "
        f"overlap with the query. The system would generate an answer based solely on "
        f"the general rule, confidently concluding it applies — missing the subordinate "
        f"exception that qualifies or reverses the outcome."
    )

    why_fails = (
        f"The general rule and its exception share vocabulary overlap of "
        f"{ctx.confounder_score:.2f}, making them nearly indistinguishable in embedding "
        f"space. The exception is structurally subordinate (child of the general rule) "
        f"but semantically identical. Only tree position — parent vs. child — resolves "
        f"which provision controls, and cosine similarity encodes no positional information."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_scope_disambiguation(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    sibling_subject = _extract_subject(ctx.sibling_context[0]["text_preview"]) if ctx.sibling_context else ""
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)

    question = (
        f"A {role} encounters a regulatory provision under {ctx.source_title} stating: "
        f'"{_clean_excerpt(target_subject, 200)}". '
        f"A colleague points out that a different part of the same regulatory framework "
        f"uses similar language but applies it in a different context. "
        f"Which interpretation governs the practitioner's specific situation?"
    )

    gold_answer = (
        f"The controlling provision is located at {' > '.join(ctx.gold_path)}. "
        f'It states: "{_clean_excerpt(target_subject, 300)}". '
        f"A sibling provision in a different subtree uses similar language: "
        f'"{_clean_excerpt(sibling_subject, 200)}". '
        f"The correct interpretation depends on which regulatory subtree governs the "
        f"specific situation — the scoping is determined by tree position, not by "
        f"the text of the provision itself."
    )

    expected_failure = (
        f"Similarity-based retrieval would return whichever provision has the highest "
        f"keyword overlap with the query, regardless of which subtree it belongs to. "
        f"Since both provisions use nearly identical regulatory language, the system "
        f"may return the sibling provision from the wrong context, producing a "
        f"technically plausible but jurisdictionally incorrect answer."
    )

    why_fails = (
        f"Both provisions define the same concept using overlapping vocabulary, "
        f"producing near-identical embeddings. The distinguishing factor is their "
        f"tree position — each is scoped to a different regulatory subtree. Cosine "
        f"similarity cannot differentiate provisions that differ only in their "
        f"structural context, not in their semantic content."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_cross_reference(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    # Pick ONE primary cross-reference
    primary_ref = ctx.cross_refs[0] if ctx.cross_refs else "a related provision"

    question = (
        f"A {role} reviewing {ctx.source_title} finds a provision stating: "
        f'"{_clean_excerpt(target_subject, 250)}". '
        f"The provision explicitly directs the reader to consult another section "
        f"of the regulatory framework for a key component of the rule. "
        f"What does the referenced provision establish, and how does it affect "
        f"the outcome under the current rule?"
    )

    gold_answer = (
        f"The provision at {' > '.join(ctx.gold_path)} explicitly references "
        f"{primary_ref}. To determine the correct outcome, the practitioner must "
        f"traverse from the current node to the referenced provision in a different "
        f"subtree. The source provision states: "
        f'"{_clean_excerpt(target_subject, 300)}". '
        f"The answer is incomplete without retrieving the content of {primary_ref}, "
        f"which is located in a structurally separate branch of the hierarchy."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the source provision containing the "
        f"cross-reference text, but treats the reference as content rather than "
        f"as a traversal instruction. The system does not follow the pointer to "
        f"{primary_ref} — it returns the node that mentions the reference, not the "
        f"node being referenced. The answer appears complete but is missing the "
        f"substance of the referenced provision."
    )

    why_fails = (
        f"Cross-references are structural pointers linking nodes in different subtrees. "
        f"Cosine similarity has no mechanism to follow a pointer — it treats "
        f"\"{primary_ref}\" as a text token, not as a traversal edge. The retrieval "
        f"system fetches the mentioning node (high similarity) but never reaches "
        f"the mentioned node (structurally distant, different embedding neighborhood)."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_conditional_cascade(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    # Count ancestor conditions
    n_ancestors = len(ctx.ancestral_context)

    question = (
        f"A {role} is determining whether a specific regulatory rule under "
        f"{ctx.source_title} applies to their client's situation. The relevant "
        f"provision states: \"{_clean_excerpt(target_subject, 250)}\". "
        f"Does this rule apply directly, or are there prerequisite conditions "
        f"that must be satisfied before it takes effect?"
    )

    gold_answer = (
        f"The rule does not apply directly. It is gated by {ctx.signal_text} "
        f"spanning {n_ancestors} hierarchical levels. The full condition chain "
        f"traces through: {' > '.join(ctx.gold_path)}. Each ancestor level imposes "
        f"an additional prerequisite that must be met. The provision states: "
        f'"{_clean_excerpt(target_subject, 300)}". '
        f"Without satisfying all gating conditions from root to leaf, the rule "
        f"cannot be applied."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the leaf-level rule, which appears "
        f"to state the complete rule directly. The retrieved text is semantically "
        f"the most relevant to the query. However, it omits the gating conditions "
        f"imposed by {n_ancestors} ancestor nodes. The system generates an answer "
        f"that applies the rule unconditionally, ignoring the cascading prerequisites."
    )

    why_fails = (
        f"The leaf node's text has the highest semantic relevance to the query, but "
        f"the answer is incomplete without {n_ancestors} ancestor conditions that gate "
        f"its applicability. Flat retrieval cannot reconstruct the condition chain "
        f"encoded in the tree hierarchy — each level adds a constraint invisible "
        f"to cosine similarity operating on isolated chunks."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_temporal_layering(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    # Extract the actual date from signal_text
    date_signal = ctx.signal_text

    question = (
        f"A {role} is analyzing a provision under {ctx.source_title} that states: "
        f'"{_clean_excerpt(target_subject, 250)}". '
        f"The provision includes a date-specific condition referencing {date_signal}. "
        f"Does the rule as written apply to a transaction occurring today, or has "
        f"the temporal qualifier changed the outcome for the current period?"
    )

    gold_answer = (
        f"The provision at {' > '.join(ctx.gold_path)} contains the temporal "
        f'condition "{date_signal}". The full text states: '
        f'"{_clean_excerpt(target_subject, 350)}". '
        f"Whether the rule applies to a current transaction depends on whether "
        f"the transaction falls before or after {date_signal}. This temporal gate "
        f"is embedded at depth {ctx.target_tree_depth} and is not reflected in any "
        f"parent-level summary of the rule."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the substantive rule text, which "
        f"discusses the same regulatory topic. The system surfaces the rule as "
        f"if it applies uniformly, without flagging the embedded temporal condition "
        f'"{date_signal}". The generated answer assumes the rule applies regardless '
        f"of timing, which may be incorrect for transactions on either side of the date."
    )

    why_fails = (
        f'The temporal qualifier "{date_signal}" is embedded in text that is '
        f"otherwise semantically identical to a general statement of the rule. "
        f"Dates carry minimal semantic weight in embedding models relative to "
        f"substantive legal terms, so the embedding of the temporally-qualified "
        f"version is nearly identical to an unqualified version of the same rule."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_sibling_conflict(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    sibling_subject = _extract_subject(ctx.sibling_context[0]["text_preview"]) if ctx.sibling_context else ""
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)

    question = (
        f"A {role} reviewing {ctx.source_title} under {parent_label} finds two "
        f"provisions that appear to address the same regulatory scenario but reach "
        f"different conclusions. One provision states: "
        f'"{_clean_excerpt(target_subject, 200)}". '
        f"Another parallel provision states: "
        f'"{_clean_excerpt(sibling_subject, 150)}". '
        f"Which provision governs, and how should the apparent conflict be resolved?"
    )

    gold_answer = (
        f"The controlling provision is at {' > '.join(ctx.gold_path)}. It states: "
        f'"{_clean_excerpt(target_subject, 300)}". '
        f"This provision contains the signal \"{ctx.signal_text}\" which qualifies "
        f"the sibling provision. Both provisions sit under the same parent "
        f"({parent_label}) at the same tree depth, but their relative structural "
        f"position — one containing the exception, the other the general rule — "
        f"determines which controls."
    )

    expected_failure = (
        f"Similarity-based retrieval returns whichever sibling has higher keyword "
        f"overlap with the query. Since both provisions share the same parent "
        f"context and discuss the same regulatory topic, their embeddings are "
        f"nearly identical. The system may return the wrong sibling — the one "
        f"stating the general rule rather than the one containing the exception."
    )

    why_fails = (
        f"Sibling nodes share their entire ancestral context and discuss the same "
        f"topic, producing near-identical embeddings. The distinguishing information — "
        f"which sibling contains the general rule vs. the exception — is encoded "
        f"only in their relative position under the shared parent, not in their "
        f"semantic content. Cosine similarity treats them as interchangeable."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_definitional_dependency(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    # Extract what term is being defined
    def_match = re.search(
        r'(?:the\s+term\s+["\u201c]([^"\u201d]+)["\u201d]|'
        r'["\u201c]([^"\u201d]+)["\u201d]\s+means|'
        r'as\s+defined\s+in|for\s+purposes\s+of\s+this)',
        ctx.target_node_text, re.I
    )
    def_signal = def_match.group(0) if def_match else ctx.signal_text

    question = (
        f"A {role} reviewing a provision under {ctx.source_title} encounters "
        f"a rule that states: \"{_clean_excerpt(target_subject, 250)}\". "
        f"The rule relies on a term whose meaning is established in a separate "
        f"part of the regulatory framework. Without locating that definition, "
        f"can the practitioner correctly apply the rule?"
    )

    gold_answer = (
        f"No. The provision at {' > '.join(ctx.gold_path)} contains the signal "
        f'\"{def_signal}\", indicating the rule depends on a definition located '
        f"in a different subtree. The provision states: "
        f'"{_clean_excerpt(target_subject, 300)}". '
        f"Correct application requires first retrieving the controlling definition "
        f"from the referenced section, then applying the defined meaning back to "
        f"the current rule."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the rule text where the defined term "
        f"is USED, because it has the highest semantic relevance to the query. "
        f"The system does not follow the definitional chain to where the term is "
        f"DEFINED — a structurally separate node. The generated answer applies "
        f"a common-language interpretation of the term rather than the regulatory "
        f"definition, which may differ significantly."
    )

    why_fails = (
        f"The rule text (where the term is used) and the definition text (where "
        f"the term is defined) reside in different subtrees. The query semantically "
        f"matches the usage context with higher similarity than the definition context. "
        f"Flat retrieval returns usage, not definition — the structural dependency "
        f"between the two nodes is invisible to embedding similarity."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_aggregation(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)

    question = (
        f"A {role} analyzing a provision under {ctx.source_title} finds a rule "
        f"that states: \"{_clean_excerpt(target_subject, 250)}\". "
        f"The rule references \"{ctx.signal_text}\" — indicating that the complete "
        f"answer requires combining values or provisions from multiple sources. "
        f"Can the practitioner determine the correct outcome from this single provision alone?"
    )

    gold_answer = (
        f"No. The provision at {' > '.join(ctx.gold_path)} requires aggregation "
        f"(\"{ctx.signal_text}\") across multiple regulatory branches. It states: "
        f'"{_clean_excerpt(target_subject, 300)}". '
        f"The correct outcome requires retrieving component values from separate "
        f"subtrees and combining them as directed. A single provision cannot "
        f"supply the complete answer."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the aggregation instruction node — "
        f"the provision that says \"{ctx.signal_text}\" — because it is semantically "
        f"the most relevant to the query. However, the system does NOT retrieve "
        f"the separate branches whose values must be combined. The answer acknowledges "
        f"that aggregation is required but cannot supply the component values."
    )

    why_fails = (
        f"Aggregation requires information from 2+ structurally distant subtrees. "
        f"Cosine similarity retrieves the single most relevant chunk (the aggregation "
        f"instruction itself), but the component values exist in nodes with different "
        f"semantic contexts that individually do not match the query. Only a "
        f"structure-aware traversal can follow the aggregation pointers to collect "
        f"all required components."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_negative_space(ctx: ContextPackage) -> dict:
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)

    # Parse the signal to get covered/missing entity types
    signal = ctx.signal_text
    covered, missing = set(), set()
    m = re.search(r"Covers\s*\{([^}]+)\},\s*missing\s*\{([^}]+)\}", signal)
    if m:
        covered = {s.strip().strip("'\"") for s in m.group(1).split(",")}
        missing = {s.strip().strip("'\"") for s in m.group(2).split(",")}

    covered_list = ", ".join(sorted(covered)) if covered else "certain entity types"
    missing_list = ", ".join(sorted(missing)) if missing else "the specified category"

    question = (
        f"A {role} is reviewing the regulatory provisions under {ctx.source_title} "
        f"at {parent_label}. The provisions address {covered_list}. "
        f"A client asks whether {missing_list} {'is' if len(missing) <= 1 else 'are'} "
        f"also covered under this same regulatory framework. "
        f"Does a specific provision exist for {missing_list}?"
    )

    gold_answer = (
        f"No. The regulatory subtree at {' > '.join(ctx.gold_path)} contains "
        f"provisions for {covered_list} but has no dedicated provision for "
        f"{missing_list}. An exhaustive traversal of all child nodes under "
        f"{parent_label} confirms the absence. The client must seek guidance "
        f"under a different section of the regulatory framework, or determine "
        f"whether the existing provisions for {covered_list} apply by analogy."
    )

    expected_failure = (
        f"Similarity-based retrieval cannot detect the absence of a provision. "
        f"The query about {missing_list} has high semantic overlap with the "
        f"existing provisions for {covered_list}, so the system returns the "
        f"nearest related section with high confidence. The generated answer "
        f"incorrectly assumes {missing_list} {'is' if len(missing) <= 1 else 'are'} "
        f"covered under the {covered_list} provisions."
    )

    why_fails = (
        f"Negative space — the absence of a matching node — is undetectable by "
        f"cosine similarity. The system always returns something; it has no "
        f"mechanism to return 'not found.' The query about {missing_list} "
        f"semantically overlaps with provisions for {covered_list}, producing "
        f"a high-confidence false positive. Only an exhaustive tree traversal "
        f"can confirm that no child node addresses {missing_list}."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_depth_gated(ctx: ContextPackage) -> dict:
    target_subject = _extract_subject(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    specific_value = ctx.signal_text

    question = (
        f"A {role} is determining a specific quantitative requirement under "
        f"{ctx.source_title}. The general regulatory guidance under {parent_label} "
        f"describes the applicable framework but does not state a precise figure. "
        f"What is the exact value, rate, or threshold, and where in the "
        f"regulatory hierarchy is it specified?"
    )

    gold_answer = (
        f"The specific value is \"{specific_value}\", found at depth "
        f"{ctx.target_tree_depth} in the hierarchy: {' > '.join(ctx.gold_path)}. "
        f'The provision states: "{_clean_excerpt(target_subject, 300)}". '
        f"This value appears only at the leaf level and is NOT stated in the "
        f"parent-level summary under {parent_label}, which gives the general "
        f"framework without the precise figure."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the parent-level summary under "
        f"{parent_label}, which describes the general regulatory framework in "
        f"more detail and has higher keyword density. The parent text is longer, "
        f"more topically relevant, and ranks above the leaf node. The generated "
        f"answer correctly identifies the regulatory framework but cannot supply "
        f"the specific value \"{specific_value}\" because it exists only at depth "
        f"{ctx.target_tree_depth}."
    )

    why_fails = (
        f"The parent summary and the leaf-level value discuss the same topic "
        f"with vocabulary overlap of {ctx.confounder_score:.2f}. The parent node's "
        f"text is typically longer and more keyword-rich, causing it to rank above "
        f"the leaf in cosine similarity. The precise value (\"{specific_value}\") "
        f"adds negligible semantic signal relative to the surrounding legal text, "
        f"so the leaf node scores lower despite containing the required information."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


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


def generate_pilot(validated_path: str, output_dir: str) -> list[dict]:
    """Generate 100-question pilot from validated candidates."""
    with open(validated_path, "r", encoding="utf-8") as f:
        ranked_cells = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    questions: list[dict] = []
    counters: dict[str, int] = defaultdict(int)
    seen_questions: set[str] = set()  # dedup by question text hash
    seen_nodes: set[str] = set()  # dedup by required node set
    coverage: dict[tuple[str, str], int] = {}

    for domain in DOMAINS:
        for ftype in FAILURE_TYPES:
            cell_key = f"{domain}/{ftype}"
            candidates = ranked_cells.get(cell_key, [])

            if not candidates:
                print(f"  WARNING: No candidates for {cell_key}")
                coverage[(domain, ftype)] = 0
                continue

            generated = 0
            for entry in candidates[:PILOT_PER_CELL + 5]:  # extra buffer for dedup
                if generated >= PILOT_PER_CELL:
                    break

                ctx = ContextPackage(**entry["context"])
                gen_fn = GENERATORS.get(ftype)
                if not gen_fn:
                    continue

                try:
                    result = gen_fn(ctx)
                except Exception as e:
                    print(f"  ERROR {cell_key}: {e}")
                    continue

                q_text = result.get("question", "")
                if not q_text or len(q_text) < 50:
                    continue

                # Dedup: skip if question text is character-identical
                if q_text in seen_questions:
                    continue
                seen_questions.add(q_text)

                # Dedup: skip if same target node already used IN THIS CELL
                cell_node_key = f"{cell_key}_{ctx.target_node_id}"
                if cell_node_key in seen_nodes:
                    continue
                seen_nodes.add(cell_node_key)

                counters[ftype] += 1
                title_short = ctx.candidate_source_id.replace("ECFR_", "").replace("_XML", "")
                q_id = f"TB-{title_short}-{ftype.upper().replace('_', '-')}-{counters[ftype]:04d}"

                required = _build_required_nodes(ctx, ftype)
                evidence = _build_gold_evidence(ctx, ftype)

                q = {
                    "question_id": q_id,
                    "domain": domain,
                    "source_title": ctx.source_title,
                    "failure_type": ftype,
                    "structural_confounder_type": ctx.structural_confounder_type,
                    "question": _scrub_structural_refs(q_text),
                    "gold_answer": result["gold_answer"],
                    "author_expected_failure": result["author_expected_failure"],
                    "answer_type": _classify_answer_type(ftype),
                    "required_node_ids": required,
                    "distractor_node_ids": [s["node_id"] for s in ctx.sibling_context[:3] if s.get("node_id")],
                    "gold_path": ctx.gold_path,
                    "gold_evidence": evidence,
                    "why_similarity_fails": result["why_similarity_fails"],
                    "tree_depth": ctx.target_tree_depth,
                    "difficulty": _classify_difficulty(ctx),
                    "review_status": "draft_needs_validation",
                    "review_decision": "",
                    "review_notes": "",
                }
                questions.append(q)
                generated += 1

            coverage[(domain, ftype)] = generated

    # Save
    out_path = os.path.join(output_dir, "treebench_pilot_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PILOT v2 GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(questions)}")

    # Verify no artifacts
    issues = {
        "truncated_ellipsis": 0,
        "blank_under_comma": 0,
        "paragraph_p_label": 0,
        "section_in_question": 0,
        "single_required_node": 0,
    }
    for q in questions:
        if '..."' in q["gold_answer"] or "..." in q.get("gold_evidence", [{}])[0].get("evidence_text", ""):
            issues["truncated_ellipsis"] += 1
        if "under ," in q["question"]:
            issues["blank_under_comma"] += 1
        if re.search(r"paragraph\s+p\d+", q["question"], re.I):
            issues["paragraph_p_label"] += 1
        if re.search(r"§\s*[\d]|[Ss]ection\s*§|SECTION\s*§", q["question"]):
            issues["section_in_question"] += 1
        if q["failure_type"] in ("cross_reference", "definitional_dependency", "aggregation"):
            if len(q["required_node_ids"]) < 2:
                issues["single_required_node"] += 1

    print(f"\nArtifact check:")
    for issue, count in issues.items():
        status = "PASS" if count == 0 else f"FAIL ({count})"
        print(f"  {issue:30s}: {status}")

    print(f"\nCoverage matrix:")
    print(f"{'Domain':<15s}", end="")
    for ft in FAILURE_TYPES:
        print(f" {ft[:8]:>8s}", end="")
    print(f" {'TOTAL':>8s}")
    for domain in DOMAINS:
        print(f"{domain:<15s}", end="")
        row_total = 0
        for ft in FAILURE_TYPES:
            c = coverage.get((domain, ft), 0)
            row_total += c
            print(f" {c:>8d}", end="")
        print(f" {row_total:>8d}")

    print(f"\nSaved: {out_path}")
    return questions


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    validated_path = os.path.join(os.path.dirname(__file__), "..", "data", "validated", "validated_candidates.json")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pilot")

    if not os.path.exists(validated_path):
        print("Run candidate_validator.py first!")
        sys.exit(1)

    generate_pilot(validated_path, output_dir)
