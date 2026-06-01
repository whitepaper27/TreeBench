"""TreeBench v1 Question Schema.

Matches the paper's requirements:
- structural_confounder_type for sharper analysis
- rag_likely_answer showing what cosine similarity returns wrong
- gold_path as proper array for programmatic evaluation
- review_status for pilot validation workflow
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# Mapping: failure_type -> structural_confounder_type
CONFOUNDER_MAP = {
    "override_chain": "parent_child",
    "scope_disambiguation": "sibling",
    "cross_reference": "cross_reference",
    "conditional_cascade": "parent_child",
    "temporal_layering": "temporal",
    "sibling_conflict": "sibling",
    "definitional_dependency": "definition",
    "aggregation": "cross_reference",
    "negative_space": "missing_scope",
    "depth_gated_specificity": "parent_child",
}

# Source ID -> (domain, human-readable title)
SOURCE_META = {
    "ECFR_TITLE26_XML": ("tax", "eCFR Title 26 — Internal Revenue"),
    "ECFR_TITLE12_XML": ("finance", "eCFR Title 12 — Banks and Banking"),
    "ECFR_TITLE17_XML": ("finance", "eCFR Title 17 — Securities"),
    "ECFR_TITLE21_XML": ("medical", "eCFR Title 21 — Food and Drugs"),
    "ECFR_TITLE42_XML": ("medical", "eCFR Title 42 — Public Health"),
    "ECFR_TITLE29_XML": ("legal", "eCFR Title 29 — Labor"),
    "ECFR_TITLE15_XML": ("legal", "eCFR Title 15 — Commerce"),
    "ECFR_TITLE40_XML": ("compliance", "eCFR Title 40 — Environment"),
    "ECFR_TITLE45_XML": ("compliance", "eCFR Title 45 — HHS/HIPAA"),
    "ECFR_TITLE31_XML": ("compliance", "eCFR Title 31 — Treasury/AML"),
}

DOMAINS = ["tax", "finance", "medical", "legal", "compliance"]

FAILURE_TYPES = [
    "override_chain", "scope_disambiguation", "cross_reference",
    "conditional_cascade", "temporal_layering", "sibling_conflict",
    "definitional_dependency", "aggregation", "negative_space",
    "depth_gated_specificity",
]

ANSWER_TYPES = ["yes_no", "numeric", "classification", "section_reference", "multi_part"]

DIFFICULTIES = ["easy", "medium", "hard"]


@dataclass
class TreeBenchQuestion:
    question_id: str
    domain: str
    source_title: str
    failure_type: str
    structural_confounder_type: str
    question: str
    gold_answer: str
    rag_likely_answer: str
    answer_type: str
    required_node_ids: list[str]
    distractor_node_ids: list[str]
    gold_path: list[str]
    gold_evidence: list[dict]
    why_similarity_fails: str
    tree_depth: int
    difficulty: str
    review_status: str = "draft_needs_validation"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class ContextPackage:
    """Everything needed to generate a question from a validated candidate."""
    candidate_source_id: str
    failure_type: str
    structural_confounder_type: str
    domain: str
    source_title: str

    # Target node (where the answer lives)
    target_node_id: str
    target_node_type: str
    target_node_number: str
    target_node_heading: str
    target_node_text: str
    target_tree_path: str
    target_tree_depth: int

    # Ancestral context (root -> parent chain)
    ancestral_context: list[dict]  # [{node_type, number, heading, text_preview}]
    parent_rule_text: str  # The parent's text that RAG would likely retrieve

    # Sibling/distractor context
    sibling_context: list[dict]  # [{node_id, heading, text_preview}]

    # Cross-references found
    cross_refs: list[str]

    # Gold path as array
    gold_path: list[str]  # ["Title 26", "Chapter I", "Subchapter A", ...]

    # Quality metrics
    confounder_score: float  # How strong is the structural trap (0-1)
    signal_text: str  # What triggered the pattern hunter

    def to_dict(self) -> dict:
        return asdict(self)
