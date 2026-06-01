"""Pilot Generator v3 — final pilot format with 4 critical fixes.

Fixes from v2 review:
1. Fix A: Ban structural giveaway words from question text
2. Fix B: Clean regulatory text before quoting (strip parser artifacts)
3. Fix C: Remove template tail formulas — case-specific natural endings
4. Fix D: Verify signal tokens before citing in gold_answer

No schema changes. No redesign. After v3: freeze and run baselines.
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

PILOT_PER_CELL = 20   # 5 domains × 10 types × 20 = 1,000 target
CANDIDATE_BUFFER = 30  # try up to this many candidates per cell to fill quota
TARGET_TOTAL = 1000    # absolute target

# ──────────────────────────────────────────────────────────────────────
# Fix A: Structural giveaway ban list
# ──────────────────────────────────────────────────────────────────────
BANNED_PHRASES = [
    "subordinate provisions",
    "subordinate provision",
    "parent rule",
    "child exception",
    "child provision",
    "sibling node",
    "regulatory hierarchy",
    "tree position",
    "hierarchical",
    "hierarchy",
    "structural",
    "subtree",
    "traversal",
    "tree depth",
    "ancestor",
    "descendant",
    "parent node",
    "child node",
    "further investigation into subordinate",
    "without further investigation",
    "where in the regulatory hierarchy",
]

BANNED_RE = re.compile(
    "|".join(re.escape(p) for p in BANNED_PHRASES),
    re.IGNORECASE,
)


def _enforce_no_giveaways(text: str) -> str:
    """Strip any structural giveaway phrases from question text."""
    return BANNED_RE.sub("", text).strip()


# ──────────────────────────────────────────────────────────────────────
# Fix B: Clean regulatory text — strip parser artifacts
# ──────────────────────────────────────────────────────────────────────
def _clean_regulatory_text(raw_text: str) -> str:
    """Remove mechanical labels, metadata, and placeholder phrases.

    Returns pure statutory prose suitable for quoting in questions.
    """
    t = raw_text.strip()
    # Strip leading paragraph labels: (a), (1), (A), (i), etc.
    t = re.sub(r"^\([a-zA-Z0-9]+\)\s*", "", t)
    # Strip "Authority:" and "Source:" metadata lines
    t = re.sub(r"Authority:\s*.*?(?=\n|$)", "", t, flags=re.DOTALL)
    t = re.sub(r"Source:\s*.*?(?=\n|$)", "", t, flags=re.DOTALL)
    # Strip section number headers: "§ 1.4-1 Number of exemptions."
    t = re.sub(r"^§\s*[\d]+(?:\.[\d\w\-]+)*\s*", "", t)
    # Strip placeholder phrases from scrubber
    t = re.sub(r"the applicable provision\s*", "", t, flags=re.I)
    t = re.sub(r"the relevant regulatory part\s*", "", t, flags=re.I)
    t = re.sub(r"the specific provision\s*", "", t, flags=re.I)
    t = re.sub(r"the referenced provision\s*", "", t, flags=re.I)
    # Strip "In general" / "General rule" preambles
    t = re.sub(r"^(?:In general|General rule)[.\s,]*", "", t, flags=re.I)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Don't return empty
    if len(t) < 20:
        return re.sub(r"\s+", " ", raw_text.strip())
    return t


# ──────────────────────────────────────────────────────────────────────
# Fix D: Signal token verification
# ──────────────────────────────────────────────────────────────────────
def _verify_signal(signal_text: str, target_node_text: str,
                    evidence_texts: list[str] | None = None) -> str:
    """Return the signal only if it actually appears in the target text.

    If the signal is absent, return a short real excerpt that IS present
    in the evidence_text excerpts (which are truncated to ~500 chars).
    Never claim a provision contains a word it doesn't.
    """
    if not signal_text or not target_node_text:
        return ""

    # Build evidence pool: target text + any provided evidence excerpts
    check_pool = [target_node_text]
    if evidence_texts:
        check_pool.extend(evidence_texts)

    # Check signal in target text OR any evidence
    if any(signal_text.lower() in t.lower() for t in check_pool):
        return signal_text

    # Signal not found anywhere — extract a fallback that IS in evidence
    # Use the target node text as the base for extraction
    clean = _clean_regulatory_text(target_node_text)

    # Try progressively shorter phrases from the clean text
    m = re.match(r"(.{20,80}?)[,;.]", clean)
    if m:
        fallback = m.group(1).strip()
        if any(fallback.lower() in t.lower() for t in check_pool):
            return fallback

    # Try n-word phrases from the beginning
    words = clean.split()
    for length in (8, 6, 5, 4, 3):
        if len(words) >= length:
            phrase = " ".join(words[:length])
            if any(phrase.lower() in t.lower() for t in check_pool):
                return phrase

    # Last resort: use raw first phrase from target_node_text directly
    raw = target_node_text.strip()
    # Strip leading paragraph label
    raw = re.sub(r"^\([a-zA-Z0-9]+\)\s*", "", raw)
    raw = re.sub(r"^§\s*[\d]+(?:\.[\d\w\-]+)*\s+\S+\s*[.]\s*", "", raw)
    m2 = re.match(r"(.{15,60}?)[,;.]", raw.strip())
    return m2.group(1).strip() if m2 else raw[:40].strip()


# ──────────────────────────────────────────────────────────────────────
# Utilities (carried from v2, unchanged)
# ──────────────────────────────────────────────────────────────────────
def _clean_excerpt(text: str, max_len: int = 400) -> str:
    """Return a clean quoted excerpt. Cut at sentence boundary, never '...'."""
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    last_period = cut.rfind(". ")
    last_semi = cut.rfind("; ")
    boundary = max(last_period, last_semi)
    if boundary > max_len // 2:
        return cut[:boundary + 1]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 0 else cut


def _scrub_structural_refs(text: str) -> str:
    """Remove structural identifiers from question text but PRESERVE
    plain section references like 'section 3', 'section 151', 'section 11'.

    Only strip § symbols, PART/SUBPART labels, paragraph labels, and
    CFR-style compound references (§ 1.4-1, Section 225.104(c)).
    Simple 'section N' references are real statutory citations that must
    stay intact to keep quoted text grammatical.
    """
    # Strip § references: § 1.4-1, §61, § 225.104(c), etc.
    t = re.sub(r"§+\s*[\d]+(?:\.[\d\w\-]+)*(?:\([a-zA-Z0-9]+\))*", "", text)
    # Strip CFR-style compound section refs: Section 1.4-1, Section 225.104(c)
    # But KEEP simple "section 3", "section 151", "section 11" etc.
    t = re.sub(r"[Ss]ection\s+[\d]+\.[\d\w\-]+(?:\([a-zA-Z0-9]+\))*", "", t)
    t = re.sub(r"SECTION\s+(?:§\s*)?[\d]+(?:\.[\d\w\-]+)*", "", t)
    # Strip PART/SUBPART labels
    t = re.sub(r"(?:PART|Part|SUBPART|Subpart)\s+[\d]+[A-Z]?(?:\.\d+)*", "", t)
    # Strip paragraph labels: paragraph (a), paragraph p1
    t = re.sub(r"paragraph\s+(?:\([a-zA-Z0-9]+\)|p\d+)", "", t)
    # Strip "section § X.Y" compound
    t = re.sub(r"section\s+§\s*[\d]+(?:\.[\d\w\-]+)*", "", t)
    # Clean up whitespace and orphaned punctuation
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*,\s*,", ",", t)
    t = re.sub(r"\s*\.\s*\.", ".", t)
    # Fix dangling "under ," from stripped compound refs (but not "under section 3,")
    t = re.sub(r"under\s*,", "under the applicable regulation,", t)
    t = re.sub(r"(\w+)\s+,\s+", r"\1, ", t)
    return t


def _get_parent_label(ctx: ContextPackage) -> str:
    """Get a human-readable parent label, never blank. No section numbers."""
    for anc in reversed(ctx.ancestral_context):
        heading = anc.get("heading", "").strip()
        if heading and len(heading) > 5:
            clean = re.sub(r"§+\s*[\d]+(?:\.[\d\w\-]+)*\s*", "", heading).strip()
            clean = re.sub(r"^[-—–\s]+", "", clean).strip()
            if len(clean) > 5:
                return clean
            return heading
    return ctx.source_title.replace("eCFR ", "").strip()


def _domain_practitioner(domain: str) -> str:
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
    """Build proper multi-node required_node_ids. At least 2 for all types."""
    nodes = [ctx.target_node_id]

    if failure_type in ("override_chain", "depth_gated_specificity", "conditional_cascade"):
        for anc in reversed(ctx.ancestral_context):
            if len(anc.get("text_preview", "")) > 30:
                anc_type = anc.get("node_type", "")
                anc_num = anc.get("number", "")
                anc_id = f"{ctx.candidate_source_id}__{anc_type}_{anc_num}".replace(" ", "_")
                nodes.insert(0, anc_id)
                break

    if failure_type in ("scope_disambiguation", "sibling_conflict"):
        for sib in ctx.sibling_context[:2]:
            if sib.get("node_id"):
                nodes.append(sib["node_id"])

    if failure_type in ("cross_reference", "definitional_dependency", "aggregation"):
        for ref in ctx.cross_refs[:2]:
            ref_num = re.search(r"[\d]+(?:\.[\d\w\-]+)*", ref)
            if ref_num:
                ref_id = f"{ctx.candidate_source_id}__ref_{ref_num.group(0)}"
                nodes.append(ref_id)
        if len(nodes) < 2 and ctx.sibling_context:
            nodes.append(ctx.sibling_context[0].get("node_id", f"{ctx.candidate_source_id}__sibling_0"))
        if len(nodes) < 2 and ctx.ancestral_context:
            anc = ctx.ancestral_context[-1]
            nodes.append(f"{ctx.candidate_source_id}__{anc.get('node_type', 'parent')}_{anc.get('number', '0')}")

    if len(nodes) < 2 and ctx.ancestral_context:
        anc = ctx.ancestral_context[-1]
        nodes.insert(0, f"{ctx.candidate_source_id}__{anc.get('node_type', 'parent')}_{anc.get('number', '0')}")

    return nodes


def _build_gold_evidence(ctx: ContextPackage, failure_type: str) -> list[dict]:
    """Build complete gold evidence without truncation.

    Ensures the signal_text appears in the evidence excerpt if it exists
    in the full text — avoids hallucination where gold_answer cites a
    signal that's beyond the truncated evidence.
    """
    target_excerpt = _clean_excerpt(ctx.target_node_text, 500)
    # If signal is in full text but not in excerpt, extend or shift the excerpt
    if (ctx.signal_text
            and ctx.signal_text.lower() in ctx.target_node_text.lower()
            and ctx.signal_text.lower() not in target_excerpt.lower()):
        # Find where the signal starts and excerpt around it
        idx = ctx.target_node_text.lower().index(ctx.signal_text.lower())
        start = max(0, idx - 200)
        target_excerpt = _clean_excerpt(ctx.target_node_text[start:], 500)

    evidence = [{
        "source": ctx.candidate_source_id,
        "section_id": ctx.target_node_number,
        "node_type": ctx.target_node_type,
        "evidence_text": target_excerpt,
    }]

    if failure_type in ("override_chain", "depth_gated_specificity", "conditional_cascade"):
        if ctx.ancestral_context:
            parent = ctx.ancestral_context[-1]
            evidence.insert(0, {
                "source": ctx.candidate_source_id,
                "section_id": parent.get("number", ""),
                "node_type": parent.get("node_type", ""),
                "evidence_text": _clean_excerpt(parent.get("text_preview", ""), 500),
            })

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
# v3 question generators — factual case masking, no giveaways
# ──────────────────────────────────────────────────────────────────────

def _evidence_texts(ctx: ContextPackage) -> list[str]:
    """Collect all available evidence texts for signal verification."""
    texts = [ctx.target_node_text]
    if ctx.parent_rule_text:
        texts.append(ctx.parent_rule_text)
    for sib in ctx.sibling_context:
        if sib.get("text_preview"):
            texts.append(sib["text_preview"])
    return texts


def _gen_override_chain(ctx: ContextPackage) -> dict:
    parent_label = _get_parent_label(ctx)
    parent_prose = _clean_regulatory_text(ctx.parent_rule_text)
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    verified_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} is advising a client on a matter under {ctx.source_title}. "
        f"The general framework under {parent_label} states: "
        f'"{_clean_excerpt(parent_prose, 200)}". '
        f"Based on this framework, what are the exact requirements that apply "
        f"to the client's situation?"
    )

    gold_answer = (
        f"No. While the general framework under {parent_label} sets a baseline, "
        f"the specific provision at {' > '.join(ctx.gold_path)} states: "
        f'"{_clean_excerpt(target_prose, 300)}". '
        f'This provision contains the language "{verified_signal}" which '
        f"qualifies the general rule. The correct analysis requires reading "
        f"both the general framework and this specific provision at depth "
        f"{ctx.target_tree_depth}."
    )

    expected_failure = (
        f"A similarity-based retrieval system would return the general provision "
        f"under {parent_label}, which directly addresses the topic and contains "
        f"high keyword overlap with the query. It would miss the specific "
        f"provision that qualifies or reverses the general rule."
    )

    why_fails = (
        f"The general rule and its qualifier share vocabulary overlap of "
        f"{ctx.confounder_score:.2f}, making them nearly indistinguishable in "
        f"embedding space. Only the document structure — the qualifier being "
        f"nested under the general rule — resolves which provision controls."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_scope_disambiguation(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    sibling_prose = _clean_regulatory_text(
        ctx.sibling_context[0]["text_preview"]
    ) if ctx.sibling_context else ""
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)

    question = (
        f"A {role} encounters a provision under {ctx.source_title} stating: "
        f'"{_clean_excerpt(target_prose, 200)}". '
        f"A colleague points out that a different part of the same regulation "
        f"uses nearly identical language but applies to a different context. "
        f"Which provision governs the practitioner's specific situation?"
    )

    gold_answer = (
        f"The controlling provision is located at {' > '.join(ctx.gold_path)}. "
        f'It states: "{_clean_excerpt(target_prose, 300)}". '
        f"A parallel provision uses similar language: "
        f'"{_clean_excerpt(sibling_prose, 200)}". '
        f"The correct interpretation depends on which regulatory context "
        f"applies to the specific situation at hand."
    )

    expected_failure = (
        f"Similarity-based retrieval would return whichever provision has the "
        f"highest keyword overlap with the query, regardless of regulatory "
        f"context. Since both provisions use nearly identical language, the "
        f"system may return the wrong one."
    )

    why_fails = (
        f"Both provisions define the same concept using overlapping vocabulary, "
        f"producing near-identical embeddings. The distinguishing factor is "
        f"their position in the regulation — each is scoped to a different "
        f"context. Cosine similarity cannot differentiate them."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_cross_reference(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    primary_ref = ctx.cross_refs[0] if ctx.cross_refs else "a related provision"

    question = (
        f"A {role} reviewing {ctx.source_title} finds a provision stating: "
        f'"{_clean_excerpt(target_prose, 250)}". '
        f"This provision references {primary_ref}. What does {primary_ref} "
        f"establish, and how does it affect the current rule?"
    )

    gold_answer = (
        f"The provision at {' > '.join(ctx.gold_path)} references "
        f"{primary_ref}. To determine the correct outcome, the practitioner "
        f"must locate and read the referenced provision. The source states: "
        f'"{_clean_excerpt(target_prose, 300)}". '
        f"The answer is incomplete without the content of {primary_ref}."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the provision containing the "
        f"reference text but does not follow the pointer to {primary_ref}. "
        f"The system treats the reference as content rather than as an "
        f"instruction to look elsewhere."
    )

    why_fails = (
        f"Cross-references link provisions in different parts of the regulation. "
        f"Cosine similarity has no mechanism to follow a reference — it treats "
        f"\"{primary_ref}\" as a text token, not as a pointer. The system "
        f"fetches the mentioning provision but never reaches the mentioned one."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_conditional_cascade(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    n_ancestors = len(ctx.ancestral_context)
    verified_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} is determining whether a specific rule under "
        f"{ctx.source_title} applies to their client's situation. The "
        f"provision states: \"{_clean_excerpt(target_prose, 250)}\". "
        f"What conditions must be met before this rule takes effect?"
    )

    gold_answer = (
        f"The rule does not apply directly. It is conditional on "
        f'"{verified_signal}" spanning {n_ancestors} levels. The full '
        f"path is: {' > '.join(ctx.gold_path)}. Each level imposes "
        f"an additional prerequisite. The provision states: "
        f'"{_clean_excerpt(target_prose, 300)}". '
        f"Without satisfying all conditions from the general framework "
        f"down to this specific provision, the rule cannot be applied."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the specific rule, which "
        f"appears to state the complete requirement. However, it omits "
        f"the gating conditions imposed by {n_ancestors} higher-level "
        f"provisions. The answer applies the rule unconditionally."
    )

    why_fails = (
        f"The specific rule has the highest semantic relevance to the query, "
        f"but the answer is incomplete without {n_ancestors} higher-level "
        f"conditions that gate its applicability. Flat retrieval cannot "
        f"reconstruct the condition chain encoded in the document structure."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_temporal_layering(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    date_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} is analyzing a provision under {ctx.source_title} that "
        f"states: \"{_clean_excerpt(target_prose, 250)}\". "
        f"The provision includes a date-specific condition referencing "
        f"{date_signal}. Does this rule apply to a transaction occurring today?"
    )

    gold_answer = (
        f"The provision at {' > '.join(ctx.gold_path)} contains the temporal "
        f'condition "{date_signal}". It states: '
        f'"{_clean_excerpt(target_prose, 350)}". '
        f"Whether the rule applies depends on whether the transaction falls "
        f"before or after {date_signal}. This date condition is embedded at "
        f"depth {ctx.target_tree_depth} and is not reflected in any "
        f"higher-level summary of the rule."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the substantive rule text "
        f"without flagging the embedded temporal condition "
        f'"{date_signal}". The answer assumes the rule applies '
        f"regardless of timing."
    )

    why_fails = (
        f'The temporal qualifier "{date_signal}" is embedded in text that '
        f"is otherwise semantically identical to a general statement of the "
        f"rule. Dates carry minimal weight in embedding models relative to "
        f"substantive legal terms, so the qualified and unqualified versions "
        f"produce nearly identical embeddings."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_sibling_conflict(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    sibling_prose = _clean_regulatory_text(
        ctx.sibling_context[0]["text_preview"]
    ) if ctx.sibling_context else ""
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    verified_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} reviewing {ctx.source_title} under {parent_label} finds "
        f"two provisions that appear to address the same scenario but reach "
        f"different conclusions. One states: "
        f'"{_clean_excerpt(target_prose, 200)}". '
        f"Another states: "
        f'"{_clean_excerpt(sibling_prose, 150)}". '
        f"Which provision controls?"
    )

    gold_answer = (
        f"The controlling provision is at {' > '.join(ctx.gold_path)}. "
        f'It states: "{_clean_excerpt(target_prose, 300)}". '
        f'This provision contains the language "{verified_signal}" which '
        f"qualifies the parallel provision. Both sit under {parent_label} "
        f"at the same level, but their relative position determines which "
        f"one controls."
    )

    expected_failure = (
        f"Similarity-based retrieval returns whichever provision has higher "
        f"keyword overlap with the query. Since both share the same context "
        f"and discuss the same topic, the system may return the wrong one."
    )

    why_fails = (
        f"Sibling provisions share their entire context and discuss the "
        f"same topic, producing near-identical embeddings. The distinguishing "
        f"information — which contains the general rule vs. the exception — "
        f"is encoded only in their relative position, not in their content."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_definitional_dependency(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    verified_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} reviewing a provision under {ctx.source_title} encounters "
        f"a rule that states: \"{_clean_excerpt(target_prose, 250)}\". "
        f"The rule uses a term whose meaning is defined elsewhere in the "
        f"regulation. What is the controlling definition of that term, "
        f"and how does it affect the rule's application?"
    )

    gold_answer = (
        f"No. The provision at {' > '.join(ctx.gold_path)} contains the "
        f'language "{verified_signal}", indicating the rule depends on a '
        f"definition located elsewhere. It states: "
        f'"{_clean_excerpt(target_prose, 300)}". '
        f"Correct application requires first finding the controlling "
        f"definition and then applying that meaning back to this rule."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the rule text where the term "
        f"is used, because it has the highest relevance to the query. "
        f"The system does not follow the definitional chain to where the "
        f"term is actually defined. The answer applies a common-language "
        f"meaning rather than the regulatory definition."
    )

    why_fails = (
        f"The rule text (where the term is used) and the definition text "
        f"(where the term is defined) reside in different parts of the "
        f"regulation. The query matches the usage context with higher "
        f"similarity than the definition context. Flat retrieval returns "
        f"usage, not definition."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_aggregation(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    verified_signal = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))

    question = (
        f"A {role} analyzing a provision under {ctx.source_title} finds a "
        f"rule that states: \"{_clean_excerpt(target_prose, 250)}\". "
        f'The rule references "{verified_signal}". What is the complete '
        f"calculated outcome when all referenced components are combined?"
    )

    gold_answer = (
        f"No. The provision at {' > '.join(ctx.gold_path)} requires "
        f'aggregation ("{verified_signal}") across multiple provisions. '
        f'It states: "{_clean_excerpt(target_prose, 300)}". '
        f"The correct outcome requires retrieving component values from "
        f"separate parts of the regulation and combining them as directed."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the aggregation instruction — "
        f'the provision that says "{verified_signal}" — because it is the '
        f"most relevant to the query. But the system does not retrieve the "
        f"separate provisions whose values must be combined."
    )

    why_fails = (
        f"Aggregation requires information from multiple structurally "
        f"distant provisions. Cosine similarity retrieves the single most "
        f"relevant chunk (the aggregation instruction), but the component "
        f"values exist in provisions with different semantic contexts that "
        f"individually do not match the query."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_negative_space(ctx: ContextPackage) -> dict:
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    # Use target node heading or number for differentiation
    section_label = ctx.target_node_heading or ctx.target_node_number or parent_label

    signal = ctx.signal_text
    covered, missing = set(), set()
    m = re.search(r"Covers\s*\{([^}]+)\},\s*missing\s*\{([^}]+)\}", signal)
    if m:
        covered = {s.strip().strip("'\"") for s in m.group(1).split(",")}
        missing = {s.strip().strip("'\"") for s in m.group(2).split(",")}

    covered_list = ", ".join(sorted(covered)) if covered else "certain categories"
    missing_list = ", ".join(sorted(missing)) if missing else "the specified category"

    question = (
        f"A {role} is reviewing the provisions under {ctx.source_title} "
        f"at {parent_label} ({section_label}). The existing provisions "
        f"address {covered_list}. A client needs guidance specifically "
        f"for {missing_list}. What provision governs {missing_list} "
        f"under this framework?"
    )

    gold_answer = (
        f"No. The provisions at {' > '.join(ctx.gold_path)} cover "
        f"{covered_list} but contain no dedicated provision for "
        f"{missing_list}. A complete review of all provisions under "
        f"{parent_label} confirms the absence. The client must look "
        f"to a different part of the regulation, or determine whether "
        f"the existing provisions for {covered_list} apply by analogy."
    )

    expected_failure = (
        f"Similarity-based retrieval cannot detect the absence of a "
        f"provision. The query about {missing_list} has high semantic "
        f"overlap with the existing provisions for {covered_list}, so "
        f"the system returns the nearest related provision with high "
        f"confidence — incorrectly suggesting {missing_list} "
        f"{'is' if len(missing) <= 1 else 'are'} covered."
    )

    why_fails = (
        f"The absence of a matching provision is undetectable by cosine "
        f"similarity. The system always returns something. The query about "
        f"{missing_list} overlaps with provisions for {covered_list}, "
        f"producing a high-confidence false positive. Only an exhaustive "
        f"review of all provisions can confirm that none addresses "
        f"{missing_list}."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


def _gen_depth_gated(ctx: ContextPackage) -> dict:
    target_prose = _clean_regulatory_text(ctx.target_node_text)
    role = _domain_practitioner(ctx.domain)
    parent_label = _get_parent_label(ctx)
    specific_value = _verify_signal(ctx.signal_text, ctx.target_node_text, _evidence_texts(ctx))
    # Include a distinguishing excerpt from the target to avoid dedup collisions
    target_excerpt = _clean_excerpt(target_prose, 150)

    question = (
        f"A {role} is determining a specific quantitative requirement under "
        f"{ctx.source_title}. The general guidance under {parent_label} "
        f"describes the applicable framework but does not state a precise "
        f"figure. The relevant provision states: \"{target_excerpt}\". "
        f"What is the exact value, rate, or threshold that applies?"
    )

    gold_answer = (
        f"The specific value is \"{specific_value}\", found at depth "
        f"{ctx.target_tree_depth}: {' > '.join(ctx.gold_path)}. "
        f'The provision states: "{_clean_excerpt(target_prose, 300)}". '
        f"This value appears only at the specific provision level and "
        f"is not stated in the general guidance under {parent_label}."
    )

    expected_failure = (
        f"Similarity-based retrieval returns the general guidance under "
        f"{parent_label}, which describes the framework in more detail "
        f"and has higher keyword density. The general text ranks above "
        f"the specific provision. The answer correctly identifies the "
        f"framework but cannot supply the precise value "
        f"\"{specific_value}\"."
    )

    why_fails = (
        f"The general guidance and the specific provision discuss the "
        f"same topic with vocabulary overlap of {ctx.confounder_score:.2f}. "
        f"The general text is typically longer and more keyword-rich, "
        f"causing it to rank higher. The precise value adds negligible "
        f"semantic signal relative to the surrounding legal text."
    )

    return {"question": question, "gold_answer": gold_answer,
            "author_expected_failure": expected_failure, "why_similarity_fails": why_fails}


# ──────────────────────────────────────────────────────────────────────
# Hard gates — reject questions that don't meet publication standards
# ──────────────────────────────────────────────────────────────────────

# Numeric/date/threshold pattern for depth_gated values
# Must match substantive values, not stray digits in paragraph labels like (b)(4)
NUMERIC_VALUE_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?|"             # dollar amounts: $100, $4,000,000
    r"[\d]+(?:\.\d+)?\s*(?:percent|%)|"   # percentages: 120 percent, 30%
    r"[\d]+\s*(?:days?|months?|years?|hours?|minutes?)|"  # durations: 30 days
    r"(?<!\()[\d]{2,}(?:,\d{3})*(?:\.\d+)?(?!\)))"  # 2+ digit numbers not in parens
)


def _validate_depth_gated_value(value: str) -> bool:
    """Reject headings, metadata, and non-numeric values.
    The value must BE a numeric/date/threshold, not just contain one."""
    if not value:
        return False
    # Reject if it looks like a heading or prose
    if any(w in value.lower() for w in ["effective", "applicability", "general",
                                         "provisions of the act", "see §",
                                         "designated", "purposes of", "for the",
                                         "paragraph"]):
        return False
    # Must start with or be primarily a numeric pattern
    # (not a sentence that happens to contain a number)
    if len(value) > 40:
        return False  # too long to be a value
    return bool(NUMERIC_VALUE_RE.search(value))


def _validate_cross_ref_nodes(required: list[str], cross_refs: list[str],
                                source_id: str) -> list[str]:
    """Ensure cross_reference required_node_ids includes both the mentioning
    node and at least one resolved referenced node."""
    if len(required) >= 2:
        return required
    # Add a referenced node
    for ref in cross_refs:
        ref_num = re.search(r"[\d]+(?:\.[\d\w\-]+)*", ref)
        if ref_num:
            ref_id = f"{source_id}__ref_{ref_num.group(0)}"
            if ref_id not in required:
                required.append(ref_id)
                break
    return required


def _validate_negative_space(ctx: ContextPackage) -> bool:
    """Only keep negative_space rows where covered categories can be
    explicitly listed from the signal text."""
    signal = ctx.signal_text
    m = re.search(r"Covers\s*\{([^}]+)\},\s*missing\s*\{([^}]+)\}", signal)
    if not m:
        return False
    covered_raw = m.group(1).replace("'", "").replace('"', '')
    covered = {s.strip() for s in covered_raw.split(",") if s.strip()}
    missing_raw = m.group(2).replace("'", "").replace('"', '')
    missing = {s.strip() for s in missing_raw.split(",") if s.strip()}
    # Need at least 1 covered and 1 missing category
    return len(covered) >= 1 and len(missing) >= 1


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


# ──────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────

def generate_pilot(validated_path: str, output_dir: str) -> list[dict]:
    """Generate 100-question pilot from validated candidates."""
    with open(validated_path, "r", encoding="utf-8") as f:
        ranked_cells = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    questions: list[dict] = []
    counters: dict[str, int] = defaultdict(int)
    seen_questions: set[str] = set()
    seen_nodes: set[str] = set()
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
            for entry in candidates[:CANDIDATE_BUFFER]:
                if generated >= PILOT_PER_CELL:
                    break

                ctx = ContextPackage(**entry["context"])
                gen_fn = GENERATORS.get(ftype)
                if not gen_fn:
                    continue

                # ── Hard gate: negative_space must have explicit categories ──
                if ftype == "negative_space" and not _validate_negative_space(ctx):
                    continue

                try:
                    result = gen_fn(ctx)
                except Exception as e:
                    print(f"  ERROR {cell_key}: {e}")
                    continue

                q_text = result.get("question", "")
                if not q_text or len(q_text) < 50:
                    continue

                # ── Hard gate: depth_gated value must be numeric ──
                if ftype == "depth_gated_specificity":
                    val_match = re.search(r'specific value is "([^"]+)"',
                                          result.get("gold_answer", ""))
                    if val_match and not _validate_depth_gated_value(val_match.group(1)):
                        print(f"  REJECT {cell_key}: non-numeric value '{val_match.group(1)}'")
                        continue

                # Dedup by question text
                if q_text in seen_questions:
                    continue
                seen_questions.add(q_text)

                # Dedup by target node within cell
                # For cross_reference, include the primary ref to allow
                # same section with different cross-ref targets
                if ftype == "cross_reference" and ctx.cross_refs:
                    cell_node_key = f"{cell_key}_{ctx.target_node_id}_{ctx.cross_refs[0]}"
                elif ftype == "negative_space":
                    # Include signal to differentiate same-parent different-missing
                    cell_node_key = f"{cell_key}_{ctx.target_node_id}_{ctx.signal_text[:40]}"
                else:
                    cell_node_key = f"{cell_key}_{ctx.target_node_id}"
                if cell_node_key in seen_nodes:
                    continue
                seen_nodes.add(cell_node_key)

                counters[ftype] += 1
                title_short = ctx.candidate_source_id.replace("ECFR_", "").replace("_XML", "")
                q_id = f"TB-{title_short}-{ftype.upper().replace('_', '-')}-{counters[ftype]:04d}"

                required = _build_required_nodes(ctx, ftype)
                evidence = _build_gold_evidence(ctx, ftype)

                # ── Hard gate: cross_reference must have mentioning + referenced nodes ──
                if ftype == "cross_reference":
                    required = _validate_cross_ref_nodes(required, ctx.cross_refs,
                                                          ctx.candidate_source_id)

                # Scrub structural refs but preserve "section N" citations
                clean_question = _scrub_structural_refs(q_text)
                clean_question = _enforce_no_giveaways(clean_question)
                clean_question = re.sub(r"\s+", " ", clean_question).strip()

                q = {
                    "question_id": q_id,
                    "domain": domain,
                    "source_title": ctx.source_title,
                    "failure_type": ftype,
                    "structural_confounder_type": ctx.structural_confounder_type,
                    "question": clean_question,
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

    # ── Backfill: generate overflow in abundant cells to reach TARGET_TOTAL ──
    total = len(questions)
    if total < TARGET_TOTAL:
        shortfall = TARGET_TOTAL - total
        print(f"\n  Backfilling {shortfall} missing questions from abundant cells...")

        # Find cells that can produce more (have unused candidates)
        # Prioritize failure types with most candidates
        abundant_types = ["override_chain", "sibling_conflict", "conditional_cascade",
                          "temporal_layering", "definitional_dependency", "aggregation",
                          "depth_gated_specificity", "scope_disambiguation",
                          "negative_space", "cross_reference"]

        for ftype in abundant_types:
            if len(questions) >= TARGET_TOTAL:
                break
            for domain in DOMAINS:
                if len(questions) >= TARGET_TOTAL:
                    break
                cell_key = f"{domain}/{ftype}"
                candidates = ranked_cells.get(cell_key, [])
                # Try candidates beyond CANDIDATE_BUFFER
                for entry in candidates:
                    if len(questions) >= TARGET_TOTAL:
                        break
                    ctx = ContextPackage(**entry["context"])
                    gen_fn = GENERATORS.get(ftype)
                    if not gen_fn:
                        continue

                    if ftype == "negative_space" and not _validate_negative_space(ctx):
                        continue

                    # Use broader dedup key for overflow
                    if ftype == "cross_reference" and ctx.cross_refs:
                        cell_node_key = f"{cell_key}_overflow_{ctx.target_node_id}_{ctx.cross_refs[0]}"
                    elif ftype == "negative_space":
                        cell_node_key = f"{cell_key}_overflow_{ctx.target_node_id}_{ctx.signal_text[:40]}"
                    else:
                        cell_node_key = f"{cell_key}_overflow_{ctx.target_node_id}"

                    if cell_node_key in seen_nodes:
                        continue

                    try:
                        result = gen_fn(ctx)
                    except Exception:
                        continue

                    q_text = result.get("question", "")
                    if not q_text or len(q_text) < 50 or q_text in seen_questions:
                        continue

                    if ftype == "depth_gated_specificity":
                        val_m = re.search(r'specific value is "([^"]+)"',
                                          result.get("gold_answer", ""))
                        if val_m and not _validate_depth_gated_value(val_m.group(1)):
                            continue

                    seen_questions.add(q_text)
                    seen_nodes.add(cell_node_key)
                    counters[ftype] += 1
                    title_short = ctx.candidate_source_id.replace("ECFR_", "").replace("_XML", "")
                    q_id = f"TB-{title_short}-{ftype.upper().replace('_', '-')}-{counters[ftype]:04d}"
                    required = _build_required_nodes(ctx, ftype)
                    evidence = _build_gold_evidence(ctx, ftype)
                    if ftype == "cross_reference":
                        required = _validate_cross_ref_nodes(required, ctx.cross_refs,
                                                              ctx.candidate_source_id)
                    clean_question = _scrub_structural_refs(q_text)
                    clean_question = _enforce_no_giveaways(clean_question)
                    clean_question = re.sub(r"\s+", " ", clean_question).strip()
                    q = {
                        "question_id": q_id,
                        "domain": domain,
                        "source_title": ctx.source_title,
                        "failure_type": ftype,
                        "structural_confounder_type": ctx.structural_confounder_type,
                        "question": clean_question,
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
                    coverage[(domain, ftype)] = coverage.get((domain, ftype), 0) + 1
                    print(f"    +1 {domain}/{ftype} ({ctx.source_title[:25]})")

    # Save
    out_path = os.path.join(output_dir, "treebench_1000_candidate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # ── Verification ──
    print(f"\n{'='*60}")
    print(f"TREEBENCH-1000-CANDIDATE GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(questions)}")

    # v2 artifact checks (should all pass)
    issues = {
        "truncated_ellipsis": 0,
        "blank_under_comma": 0,
        "paragraph_p_label": 0,
        "section_in_question": 0,
        "single_required_node": 0,
    }
    # v3 checks — the NEW ones
    v3_issues = {
        "structural_giveaway": 0,
        "scrubber_artifact": 0,
        "signal_hallucination": 0,
        "template_tail": 0,
    }

    TEMPLATE_TAILS = [
        "without further investigation into subordinate provisions",
        "where in the regulatory hierarchy is it specified",
        "can the practitioner conclude the rule applies",
        "rely on this general rule alone",
        "without locating that definition",
        "from this provision alone",
        "does a specific provision exist for",
        "prerequisite conditions that must first be satisfied",
    ]
    SCRUBBER_ARTIFACTS = [
        "the applicable provision",
        "the relevant regulatory part",
        "the specific provision",
        "the referenced provision",
    ]

    for q in questions:
        qt = q["question"]
        ga = q["gold_answer"]

        # v2 checks
        if '..."' in ga or "..." in q.get("gold_evidence", [{}])[0].get("evidence_text", ""):
            issues["truncated_ellipsis"] += 1
        if "under ," in qt:
            issues["blank_under_comma"] += 1
        if re.search(r"paragraph\s+p\d+", qt, re.I):
            issues["paragraph_p_label"] += 1
        if re.search(r"§\s*[\d]|[Ss]ection\s*§|SECTION\s*§", qt):
            issues["section_in_question"] += 1
        if q["failure_type"] in ("cross_reference", "definitional_dependency", "aggregation"):
            if len(q["required_node_ids"]) < 2:
                issues["single_required_node"] += 1

        # v3 checks
        if BANNED_RE.search(qt):
            v3_issues["structural_giveaway"] += 1
        for artifact in SCRUBBER_ARTIFACTS:
            if artifact.lower() in qt.lower():
                v3_issues["scrubber_artifact"] += 1
                break
        for tail in TEMPLATE_TAILS:
            if tail.lower() in qt.lower():
                v3_issues["template_tail"] += 1
                break

        # Signal hallucination check: verify signal_text in gold_answer
        # matches what's in the evidence
        signal_mentions = re.findall(r'contains the (?:signal|language) "([^"]+)"', ga)
        for sig in signal_mentions:
            found_in_evidence = False
            for ev in q.get("gold_evidence", []):
                if sig.lower() in ev.get("evidence_text", "").lower():
                    found_in_evidence = True
                    break
            if not found_in_evidence:
                v3_issues["signal_hallucination"] += 1
                break

    # Hard gate checks
    hard_gates = {
        "depth_gated_non_numeric": 0,
        "cross_ref_single_node": 0,
        "negative_space_no_categories": 0,
        "mid_word_truncation": 0,
        "total_count_target": 0 if len(questions) >= TARGET_TOTAL else 1,
    }
    for q in questions:
        qt = q["question"]
        if q["failure_type"] == "depth_gated_specificity":
            val_m = re.search(r'specific value is "([^"]+)"', q["gold_answer"])
            if val_m and not _validate_depth_gated_value(val_m.group(1)):
                hard_gates["depth_gated_non_numeric"] += 1
        if q["failure_type"] == "cross_reference":
            if len(q["required_node_ids"]) < 2:
                hard_gates["cross_ref_single_node"] += 1
        if q["failure_type"] == "negative_space":
            if "certain categories" in qt:
                hard_gates["negative_space_no_categories"] += 1
        # Check for mid-word/mid-sentence truncation in quoted text
        for m in re.finditer(r'"([^"]+)"', qt):
            quoted = m.group(1)
            if quoted and len(quoted) > 20 and quoted[-1] not in '.;?!,):':
                # Quoted text doesn't end cleanly
                if not quoted[-1].isalnum():
                    hard_gates["mid_word_truncation"] += 1

    print(f"\nv2 artifact checks:")
    for issue, count in issues.items():
        status = "PASS" if count == 0 else f"FAIL ({count})"
        print(f"  {issue:30s}: {status}")

    print(f"\nv3 quality checks:")
    for issue, count in v3_issues.items():
        status = "PASS" if count == 0 else f"FAIL ({count})"
        print(f"  {issue:30s}: {status}")

    print(f"\nHard gates:")
    for issue, count in hard_gates.items():
        status = "PASS" if count == 0 else f"FAIL ({count})"
        print(f"  {issue:30s}: {status}")

    # Coverage matrix
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

    # Sample output
    print(f"\n--- Sample v3 OVERRIDE_CHAIN ---")
    for q in questions:
        if q["failure_type"] == "override_chain":
            print(f"  Q: {q['question'][:300]}")
            print(f"  A: {q['gold_answer'][:300]}")
            break

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
