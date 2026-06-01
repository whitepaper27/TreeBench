"""TreeNode data model shared by all parsers."""

from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from typing import Optional


# Regex for cross-reference extraction: matches §61, §1.61-1, section 61(a)(1), etc.
_XREF_RE = re.compile(
    r"(?:§+\s*[\d]+(?:\.[\d\w\-]+)*(?:\([a-zA-Z0-9]+\))*)"
    r"|(?:(?:[Ss]ection|[Ss]ec\.|[Pp]art|[Ss]ubpart)\s+[\d]+(?:\.[\d\w\-]+)*(?:\([a-zA-Z0-9]+\))*)",
    re.UNICODE,
)


@dataclass
class TreeNode:
    id: str
    node_type: str          # title, chapter, subchapter, part, subpart, section, subsection, paragraph
    number: str             # e.g. "26", "I", "A", "1.61-1", "(a)"
    heading: str
    text: str               # full text content of this node (leaf text only, not children)
    path: str               # human-readable lineage: "Title 26 > Chapter I > Subchapter A > ..."
    source_id: str          # e.g. "ECFR_TITLE26_XML"
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    cross_refs: list[str] = field(default_factory=list)

    def extract_cross_refs(self) -> list[str]:
        """Pull §/section references from this node's text."""
        self.cross_refs = list(set(_XREF_RE.findall(self.text)))
        return self.cross_refs

    def to_dict(self) -> dict:
        return asdict(self)


class TreeStore:
    """In-memory store for a parsed document tree."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.nodes: dict[str, TreeNode] = {}
        self.root_ids: list[str] = []

    def add(self, node: TreeNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root_ids.append(node.id)

    def get(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def ancestors(self, node_id: str) -> list[TreeNode]:
        """Return list from root down to (but not including) node_id."""
        chain: list[TreeNode] = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes[current.parent_id]
            chain.append(parent)
            current = parent
        chain.reverse()
        return chain

    def subtree_nodes(self, node_id: str) -> list[TreeNode]:
        """BFS all descendants including node_id itself."""
        result: list[TreeNode] = []
        queue = [node_id]
        while queue:
            nid = queue.pop(0)
            node = self.nodes.get(nid)
            if node:
                result.append(node)
                queue.extend(node.children)
        return result

    def siblings(self, node_id: str) -> list[TreeNode]:
        """Return sibling nodes (same parent, excluding self)."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        parent = self.nodes[node.parent_id]
        return [self.nodes[cid] for cid in parent.children if cid != node_id]

    def depth_stats(self) -> dict:
        if not self.nodes:
            return {"count": 0, "max_depth": 0, "avg_depth": 0}
        depths = [n.depth for n in self.nodes.values()]
        return {
            "count": len(depths),
            "max_depth": max(depths),
            "avg_depth": round(sum(depths) / len(depths), 2),
            "depth_distribution": {d: depths.count(d) for d in sorted(set(depths))},
        }

    def save(self, path: str) -> None:
        data = {
            "source_id": self.source_id,
            "stats": self.depth_stats(),
            "root_ids": self.root_ids,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TreeStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls(data["source_id"])
        store.root_ids = data["root_ids"]
        for nid, nd in data["nodes"].items():
            store.nodes[nid] = TreeNode(**nd)
        return store
