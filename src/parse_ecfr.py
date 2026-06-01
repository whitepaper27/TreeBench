"""Parse eCFR bulk XML into a TreeStore.

eCFR XML uses nested <DIV1>…<DIV8> elements:
    DIV1 TYPE="TITLE"      → depth 0
    DIV2 TYPE="CHAPTER"    → depth 1
    DIV3 TYPE="SUBCHAPTER" → depth 2
    DIV4 TYPE="PART"       → depth 3
    DIV5 TYPE="SUBPART"    → depth 4
    DIV6 TYPE="SECTION"    → depth 5
    DIV7                   → depth 6
    DIV8                   → depth 7

Within sections, <P> tags hold paragraph text.  Subsection markers like
(a), (b), (1), (A) appear at the start of <P> text.
"""

from __future__ import annotations
import re, sys, os
from pathlib import Path
from lxml import etree
from tree_node import TreeNode, TreeStore

# Match eCFR DIV elements
_DIV_RE = re.compile(r"^DIV(\d)$", re.IGNORECASE)

# Match paragraph labels: (a), (1), (A), (i), (I), etc.
_PARA_LABEL_RE = re.compile(r"^\(([a-zA-Z0-9]+)\)")


def _text_of(el: etree._Element) -> str:
    """Get direct text content of an element, excluding child DIV text."""
    parts: list[str] = []
    if el.text:
        parts.append(el.text.strip())
    for child in el:
        tag = child.tag if isinstance(child.tag, str) else ""
        # Skip nested DIVs — their text belongs to child nodes
        if _DIV_RE.match(tag):
            continue
        # Include text from P, HD, NOTE, AUTH, etc.
        inner = etree.tostring(child, method="text", encoding="unicode") or ""
        inner = inner.strip()
        if inner:
            parts.append(inner)
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(parts)


def _heading_of(el: etree._Element) -> str:
    """Extract heading from HD (header) child elements."""
    for child in el:
        tag = child.tag if isinstance(child.tag, str) else ""
        if tag.upper() == "HD":
            return (etree.tostring(child, method="text", encoding="unicode") or "").strip()
    return ""


def _make_id(source_id: str, div_type: str, number: str, parent_id: str) -> str:
    """Create a deterministic node ID."""
    safe_num = re.sub(r"[^a-zA-Z0-9._\-]", "", number) or "X"
    safe_type = div_type.lower().replace(" ", "_")
    if parent_id:
        return f"{parent_id}__{safe_type}_{safe_num}"
    return f"{source_id}__{safe_type}_{safe_num}"


def _parse_paragraphs(section_el: etree._Element, parent_node: TreeNode, store: TreeStore) -> None:
    """Parse <P> elements within a section into paragraph-level child nodes."""
    p_elements = section_el.findall("P")
    if not p_elements:
        return

    for i, p_el in enumerate(p_elements):
        p_text = (etree.tostring(p_el, method="text", encoding="unicode") or "").strip()
        if not p_text:
            continue

        # Try to extract paragraph label like (a), (1), etc.
        label_match = _PARA_LABEL_RE.match(p_text)
        label = f"({label_match.group(1)})" if label_match else f"p{i+1}"

        p_id = f"{parent_node.id}__para_{label.strip('()')}"
        p_path = f"{parent_node.path} > {label}"

        p_node = TreeNode(
            id=p_id,
            node_type="paragraph",
            number=label,
            heading="",
            text=p_text,
            path=p_path,
            source_id=parent_node.source_id,
            parent_id=parent_node.id,
            depth=parent_node.depth + 1,
        )
        p_node.extract_cross_refs()
        store.add(p_node)
        parent_node.children.append(p_id)


def _walk_div(el: etree._Element, source_id: str, parent_id: str | None,
              parent_path: str, depth: int, store: TreeStore) -> None:
    """Recursively walk DIV elements and build TreeNode entries."""
    tag = el.tag if isinstance(el.tag, str) else ""
    m = _DIV_RE.match(tag)
    if not m:
        return

    div_type = el.get("TYPE", f"div{m.group(1)}").strip()
    number = el.get("N", "").strip()
    heading = _heading_of(el)

    node_id = _make_id(source_id, div_type, number, parent_id or source_id)
    label = f"{div_type} {number}".strip() if number else div_type
    node_path = f"{parent_path} > {label}" if parent_path else label

    # Collect direct text (not from child DIVs)
    text = _text_of(el)

    node = TreeNode(
        id=node_id,
        node_type=div_type.lower(),
        number=number,
        heading=heading,
        text=text[:2000],  # cap text to avoid massive leaf nodes
        path=node_path,
        source_id=source_id,
        parent_id=parent_id,
        depth=depth,
    )
    node.extract_cross_refs()
    store.add(node)

    if parent_id and parent_id in store.nodes:
        store.nodes[parent_id].children.append(node_id)

    # Recurse into child DIVs
    has_child_divs = False
    for child in el:
        child_tag = child.tag if isinstance(child.tag, str) else ""
        if _DIV_RE.match(child_tag):
            has_child_divs = True
            _walk_div(child, source_id, node_id, node_path, depth + 1, store)

    # If this is a SECTION-level node (no child DIVs), parse paragraphs
    if not has_child_divs and div_type.lower() == "section":
        _parse_paragraphs(el, node, store)


def parse_ecfr(xml_path: str, source_id: str | None = None) -> TreeStore:
    """Parse an eCFR bulk XML file into a TreeStore.

    Args:
        xml_path: Path to the eCFR XML file (e.g. ECFR-title26.xml)
        source_id: Override source ID; defaults to filename-based ID
    """
    path = Path(xml_path)
    if source_id is None:
        # ECFR-title26.xml → ECFR_TITLE26_XML
        source_id = path.stem.upper().replace("-", "_") + "_XML"

    print(f"Parsing {path.name} as {source_id} ...")
    tree = etree.parse(str(path))
    root = tree.getroot()

    store = TreeStore(source_id)

    # eCFR XML: root is typically <ECFR> or similar, containing DIV1 children
    for el in root.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        if _DIV_RE.match(tag):
            # Only process top-level DIV1s (titles) — deeper ones are handled by recursion
            if tag.upper() == "DIV1":
                _walk_div(el, source_id, None, "", 0, store)

    stats = store.depth_stats()
    print(f"  Parsed {stats['count']} nodes, max depth {stats['max_depth']}, avg depth {stats['avg_depth']:.1f}")
    return store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_ecfr.py <ecfr_xml_path> [output_json_path]")
        sys.exit(1)

    xml_file = sys.argv[1]
    store = parse_ecfr(xml_file)

    out_path = sys.argv[2] if len(sys.argv) > 2 else xml_file.replace(".xml", "_tree.json")
    store.save(out_path)
    print(f"  Saved to {out_path}")
