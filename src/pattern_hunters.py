"""10 Taxonomy Pattern Hunters for TreeBench.

Each hunter scans a TreeStore and returns candidate subgraphs where
RAG's cosine similarity would structurally fail.

Returns list of CandidateMatch dicts ready for Phase 3 context assembly.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from tree_node import TreeNode, TreeStore


@dataclass
class CandidateMatch:
    failure_type: str
    target_node_id: str
    required_node_ids: list[str]
    distractor_node_ids: list[str]
    tree_path: str
    tree_depth: int
    source_id: str
    signal_text: str  # the text snippet that triggered the match
    confidence: float = 0.0  # 0-1, how strong the signal is


# ──────────────────────────────────────────────────────────────────────
# 1. Override Chain
# ──────────────────────────────────────────────────────────────────────
_OVERRIDE_PATTERNS = re.compile(
    r"(?:except\s+as\s+provided|notwithstanding|subject\s+to\s+(?:paragraph|subsection|section)"
    r"|unless\s+otherwise|shall\s+not\s+apply\s+to|does\s+not\s+include"
    r"|except\s+that|provided,?\s+however)",
    re.IGNORECASE,
)


def hunt_override_chain(store: TreeStore) -> list[CandidateMatch]:
    """Find child nodes that override/carve-out parent rules."""
    matches = []
    for node in store.nodes.values():
        if not node.text or node.depth < 2:
            continue
        m = _OVERRIDE_PATTERNS.search(node.text)
        if m:
            ancestors = store.ancestors(node.id)
            if not ancestors:
                continue
            parent = ancestors[-1]
            matches.append(CandidateMatch(
                failure_type="override_chain",
                target_node_id=node.id,
                required_node_ids=[a.id for a in ancestors] + [node.id],
                distractor_node_ids=[s.id for s in store.siblings(node.id)[:3]],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=m.group(0),
                confidence=0.8,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 2. Scope Disambiguation
# ──────────────────────────────────────────────────────────────────────
_DEF_TERM_RE = re.compile(
    r'(?:the\s+term\s+["\u201c]([^"\u201d]+)["\u201d]|["\u201c]([^"\u201d]+)["\u201d]\s+means)',
    re.IGNORECASE,
)


def hunt_scope_disambiguation(store: TreeStore) -> list[CandidateMatch]:
    """Find terms defined differently in multiple subtrees."""
    # Map term → list of (node_id, path)
    term_locations: dict[str, list[tuple[str, str]]] = {}
    for node in store.nodes.values():
        if not node.text:
            continue
        for m in _DEF_TERM_RE.finditer(node.text):
            term = (m.group(1) or m.group(2)).lower().strip()
            if len(term) > 3:  # skip tiny terms
                term_locations.setdefault(term, []).append((node.id, node.path))

    matches = []
    for term, locations in term_locations.items():
        if len(locations) < 2:
            continue
        # Check that definitions are in different subtrees (different grandparent)
        grandparents = set()
        for nid, _ in locations:
            anc = store.ancestors(nid)
            gp = anc[1].id if len(anc) > 1 else anc[0].id if anc else nid
            grandparents.add(gp)

        if len(grandparents) >= 2:
            # Use all locations as required, first as target
            matches.append(CandidateMatch(
                failure_type="scope_disambiguation",
                target_node_id=locations[0][0],
                required_node_ids=[loc[0] for loc in locations],
                distractor_node_ids=[],
                tree_path=locations[0][1],
                tree_depth=store.nodes[locations[0][0]].depth,
                source_id=store.source_id,
                signal_text=f'Term "{term}" defined in {len(locations)} locations',
                confidence=0.7,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 3. Cross-Reference Traversal
# ──────────────────────────────────────────────────────────────────────
_XREF_SECTION_RE = re.compile(
    r"(?:§+\s*([\d]+(?:\.[\d\w\-]+)*))"
    r"|(?:(?:section|sec\.)\s+([\d]+(?:\.[\d\w\-]+)*))",
    re.IGNORECASE,
)


def hunt_cross_reference(store: TreeStore) -> list[CandidateMatch]:
    """Find nodes that reference sections in different subtrees."""
    matches = []
    # Build index: section number → node_id
    section_index: dict[str, str] = {}
    for node in store.nodes.values():
        if node.number and node.node_type in ("section", "part"):
            section_index[node.number] = node.id

    for node in store.nodes.values():
        if not node.text or node.depth < 2:
            continue
        refs = _XREF_SECTION_RE.findall(node.text)
        for ref_tuple in refs:
            ref_num = ref_tuple[0] or ref_tuple[1]
            if ref_num and ref_num in section_index:
                target_id = section_index[ref_num]
                if target_id == node.id:
                    continue
                # Check they're in different subtrees
                node_anc = store.ancestors(node.id)
                target_anc = store.ancestors(target_id)
                if len(node_anc) > 1 and len(target_anc) > 1 and node_anc[1].id != target_anc[1].id:
                    matches.append(CandidateMatch(
                        failure_type="cross_reference",
                        target_node_id=target_id,
                        required_node_ids=[node.id, target_id],
                        distractor_node_ids=[s.id for s in store.siblings(node.id)[:2]],
                        tree_path=node.path,
                        tree_depth=node.depth,
                        source_id=node.source_id,
                        signal_text=f"References §{ref_num}",
                        confidence=0.75,
                    ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 4. Conditional Cascade
# ──────────────────────────────────────────────────────────────────────
_COND_RE = re.compile(
    r"(?:in\s+the\s+case\s+of|if\s+.*(?:then|,)|provided\s+that|except\s+that\s+if|"
    r"in\s+any\s+case\s+in\s+which)",
    re.IGNORECASE,
)


def hunt_conditional_cascade(store: TreeStore) -> list[CandidateMatch]:
    """Find nodes with 3+ chained conditions across levels."""
    matches = []
    for node in store.nodes.values():
        if not node.text or node.depth < 3:
            continue
        conds_in_node = len(_COND_RE.findall(node.text))
        if conds_in_node < 1:
            continue
        # Check ancestors for additional conditions
        ancestors = store.ancestors(node.id)
        total_conds = conds_in_node
        cond_nodes = [node.id]
        for anc in reversed(ancestors[-3:]):  # check up to 3 ancestors
            if anc.text and _COND_RE.search(anc.text):
                total_conds += 1
                cond_nodes.append(anc.id)
        if total_conds >= 3:
            matches.append(CandidateMatch(
                failure_type="conditional_cascade",
                target_node_id=node.id,
                required_node_ids=cond_nodes,
                distractor_node_ids=[s.id for s in store.siblings(node.id)[:2]],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=f"{total_conds} chained conditions across {len(cond_nodes)} levels",
                confidence=0.7,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 5. Temporal Layering
# ──────────────────────────────────────────────────────────────────────
_TEMPORAL_RE = re.compile(
    r"(?:for\s+taxable\s+years?\s+beginning\s+after|effective\s+(?:date|for)|"
    r"as\s+amended\s+by|applicable\s+to\s+.*(?:after|before|beginning)|"
    r"sunset|shall\s+not\s+apply\s+.*after\s+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


def hunt_temporal_layering(store: TreeStore) -> list[CandidateMatch]:
    """Find nodes with temporal/effective date dependencies."""
    matches = []
    for node in store.nodes.values():
        if not node.text:
            continue
        m = _TEMPORAL_RE.search(node.text)
        if m:
            ancestors = store.ancestors(node.id)
            matches.append(CandidateMatch(
                failure_type="temporal_layering",
                target_node_id=node.id,
                required_node_ids=[a.id for a in ancestors] + [node.id],
                distractor_node_ids=[s.id for s in store.siblings(node.id)[:2]],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=m.group(0),
                confidence=0.65,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 6. Sibling Conflict
# ──────────────────────────────────────────────────────────────────────
def hunt_sibling_conflict(store: TreeStore) -> list[CandidateMatch]:
    """Find parallel sibling nodes with overlapping scope but different rules."""
    matches = []
    seen_parents = set()

    for node in store.nodes.values():
        if not node.parent_id or node.parent_id in seen_parents:
            continue
        siblings = store.siblings(node.id)
        if len(siblings) < 2:
            continue

        # Look for siblings where one says "shall" and another says "shall not" or "except"
        shall_nodes = []
        except_nodes = []
        for sib in [node] + siblings:
            if not sib.text:
                continue
            if re.search(r"\bshall\s+not\b|\bexcept\b|\bdoes\s+not\b", sib.text, re.I):
                except_nodes.append(sib)
            elif re.search(r"\bshall\b|\bmust\b|\bis\s+required\b", sib.text, re.I):
                shall_nodes.append(sib)

        if shall_nodes and except_nodes:
            seen_parents.add(node.parent_id)
            matches.append(CandidateMatch(
                failure_type="sibling_conflict",
                target_node_id=except_nodes[0].id,
                required_node_ids=[n.id for n in shall_nodes[:2] + except_nodes[:2]],
                distractor_node_ids=[shall_nodes[0].id],
                tree_path=except_nodes[0].path,
                tree_depth=except_nodes[0].depth,
                source_id=store.source_id,
                signal_text="Conflicting shall/shall-not at same level",
                confidence=0.6,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 7. Definitional Dependency
# ──────────────────────────────────────────────────────────────────────
_DEFN_DEP_RE = re.compile(
    r"(?:as\s+defined\s+in|for\s+purposes\s+of\s+this\s+(?:section|part|chapter|title)"
    r"|within\s+the\s+meaning\s+of|has\s+the\s+meaning\s+given)",
    re.IGNORECASE,
)


def hunt_definitional_dependency(store: TreeStore) -> list[CandidateMatch]:
    """Find nodes whose answer depends on a definition in a different subtree."""
    matches = []
    for node in store.nodes.values():
        if not node.text or node.depth < 2:
            continue
        m = _DEFN_DEP_RE.search(node.text)
        if m:
            # Check if there's a section reference nearby
            ref_match = _XREF_SECTION_RE.search(node.text)
            ref_id = None
            if ref_match:
                ref_num = ref_match.group(1) or ref_match.group(2)
                # Find the referenced section
                for other in store.nodes.values():
                    if other.number == ref_num and other.node_type in ("section", "part"):
                        ref_id = other.id
                        break

            required = [node.id]
            if ref_id:
                required.append(ref_id)

            matches.append(CandidateMatch(
                failure_type="definitional_dependency",
                target_node_id=node.id,
                required_node_ids=required,
                distractor_node_ids=[s.id for s in store.siblings(node.id)[:2]],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=m.group(0),
                confidence=0.7,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 8. Aggregation Across Branches
# ──────────────────────────────────────────────────────────────────────
_AGG_RE = re.compile(
    r"(?:the\s+sum\s+of|taken\s+together\s+with|combined\s+with|aggregate|"
    r"total\s+of\s+all|in\s+the\s+aggregate|cumulative)",
    re.IGNORECASE,
)


def hunt_aggregation(store: TreeStore) -> list[CandidateMatch]:
    """Find nodes requiring data from multiple separate subtrees."""
    matches = []
    for node in store.nodes.values():
        if not node.text or node.depth < 2:
            continue
        m = _AGG_RE.search(node.text)
        if m:
            # Look for cross-refs that point to the branches being aggregated
            refs = _XREF_SECTION_RE.findall(node.text)
            ref_ids = []
            for ref_tuple in refs:
                ref_num = ref_tuple[0] or ref_tuple[1]
                for other in store.nodes.values():
                    if other.number == ref_num and other.id != node.id:
                        ref_ids.append(other.id)
                        break

            matches.append(CandidateMatch(
                failure_type="aggregation",
                target_node_id=node.id,
                required_node_ids=[node.id] + ref_ids[:3],
                distractor_node_ids=[s.id for s in store.siblings(node.id)[:2]],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=m.group(0),
                confidence=0.65,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 9. Negative Space
# ──────────────────────────────────────────────────────────────────────
def hunt_negative_space(store: TreeStore) -> list[CandidateMatch]:
    """Find subtrees where an expected topic is NOT covered (answer = 'not addressed').

    Strategy: find sections that reference other sections but whose subtree
    doesn't contain a counterpart node for a common regulatory pattern.
    E.g., a section covers 'individuals' and 'corporations' but NOT 'partnerships'.
    """
    matches = []
    # Look for parent nodes whose children enumerate categories but miss common ones
    common_groups = [
        {"individual", "corporation", "partnership", "trust", "estate"},
        {"resident", "nonresident", "citizen"},
        {"employer", "employee", "self-employed"},
    ]

    for node in store.nodes.values():
        if len(node.children) < 2 or node.depth < 2:
            continue
        child_texts = []
        for cid in node.children:
            child = store.nodes.get(cid)
            if child:
                child_texts.append((child.heading + " " + child.text[:200]).lower())

        combined = " ".join(child_texts)
        for group in common_groups:
            found = {term for term in group if term in combined}
            missing = group - found
            if len(found) >= 2 and 0 < len(missing) <= 2:
                matches.append(CandidateMatch(
                    failure_type="negative_space",
                    target_node_id=node.id,
                    required_node_ids=[node.id] + node.children[:5],
                    distractor_node_ids=node.children[:3],
                    tree_path=node.path,
                    tree_depth=node.depth,
                    source_id=node.source_id,
                    signal_text=f"Covers {found}, missing {missing}",
                    confidence=0.5,
                ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# 10. Depth-Gated Specificity
# ──────────────────────────────────────────────────────────────────────
_SPECIFIC_VALUE_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?|[\d,]+\s*percent|\d+\s*%|\d+\s*(?:days?|years?|months?)\b"
    r"|\d+/\d+\s*of\s+the)",
    re.IGNORECASE,
)


def hunt_depth_gated(store: TreeStore) -> list[CandidateMatch]:
    """Find cases where specific values/rates appear only at leaf level, not in parent summaries."""
    matches = []
    for node in store.nodes.values():
        if not node.text or node.depth < 3 or node.children:
            continue  # only check leaf nodes
        m = _SPECIFIC_VALUE_RE.search(node.text)
        if not m:
            continue
        # Check that parent does NOT have this specific value
        ancestors = store.ancestors(node.id)
        if not ancestors:
            continue
        parent = ancestors[-1]
        if parent.text and not _SPECIFIC_VALUE_RE.search(parent.text):
            matches.append(CandidateMatch(
                failure_type="depth_gated_specificity",
                target_node_id=node.id,
                required_node_ids=[a.id for a in ancestors] + [node.id],
                distractor_node_ids=[parent.id],
                tree_path=node.path,
                tree_depth=node.depth,
                source_id=node.source_id,
                signal_text=m.group(0),
                confidence=0.75,
            ))
    return matches


# ──────────────────────────────────────────────────────────────────────
# Registry — run all hunters
# ──────────────────────────────────────────────────────────────────────
ALL_HUNTERS = {
    "override_chain": hunt_override_chain,
    "scope_disambiguation": hunt_scope_disambiguation,
    "cross_reference": hunt_cross_reference,
    "conditional_cascade": hunt_conditional_cascade,
    "temporal_layering": hunt_temporal_layering,
    "sibling_conflict": hunt_sibling_conflict,
    "definitional_dependency": hunt_definitional_dependency,
    "aggregation": hunt_aggregation,
    "negative_space": hunt_negative_space,
    "depth_gated_specificity": hunt_depth_gated,
}


def run_all_hunters(store: TreeStore, verbose: bool = True) -> dict[str, list[CandidateMatch]]:
    """Run all 10 hunters on a TreeStore and return results by failure type."""
    results: dict[str, list[CandidateMatch]] = {}
    total = 0
    for name, hunter in ALL_HUNTERS.items():
        candidates = hunter(store)
        results[name] = candidates
        total += len(candidates)
        if verbose:
            print(f"  {name:30s} -> {len(candidates):5d} candidates")
    if verbose:
        print(f"  {'TOTAL':30s} -> {total:5d} candidates")
    return results
