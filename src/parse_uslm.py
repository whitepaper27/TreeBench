"""Parse US Code USLM XML into a TreeStore.

USLM XML uses nested <level levelType="..."> elements:
    <level levelType="title">       → depth 0
    <level levelType="subtitle">    → depth 1
    <level levelType="chapter">     → depth 2
    <level levelType="subchapter">  → depth 3
    <level levelType="part">        → depth 4
    <level levelType="section">     → depth 5
    <level levelType="subsection">  → depth 6
    <level levelType="paragraph">   → depth 7
    <level levelType="subparagraph"> → depth 8
    <level levelType="clause">      → depth 9

<num> and <heading> children provide number and title.
<content> or <chapeau> hold the text.
"""

from __future__ import annotations
import re, sys, zipfile, tempfile, os
from pathlib import Path
from lxml import etree
from tree_node import TreeNode, TreeStore


# USLM namespaces vary; we'll handle both namespaced and plain
_NS_MAP = {
    "uslm": "http://xml.house.gov/schemas/uslm/1.0",
    "uslm2": "https://xml.house.gov/schemas/uslm/2.0",
}


def _find_elements(root, tag: str):
    """Find elements by local name regardless of namespace."""
    return root.iter(f"{{{_NS_MAP['uslm']}}}{tag}") if root.nsmap else root.iter(tag)


def _local_tag(el) -> str:
    """Get local tag name without namespace."""
    tag = el.tag if isinstance(el.tag, str) else ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_child_text(el, child_tag: str) -> str:
    """Get text from a direct child element by local tag name."""
    for child in el:
        if _local_tag(child) == child_tag:
            return (etree.tostring(child, method="text", encoding="unicode") or "").strip()
    return ""


def _collect_text(el) -> str:
    """Collect text from content/chapeau/text children, excluding nested levels."""
    parts: list[str] = []
    text_tags = {"content", "chapeau", "text", "p", "note"}
    for child in el:
        lt = _local_tag(child)
        if lt == "level":
            continue  # skip nested levels
        if lt in text_tags:
            t = (etree.tostring(child, method="text", encoding="unicode") or "").strip()
            if t:
                parts.append(t)
        # Also check further nested non-level elements
        if lt not in ("level", "num", "heading"):
            for sub in child.iter():
                if _local_tag(sub) in text_tags and sub != child:
                    t = (etree.tostring(sub, method="text", encoding="unicode") or "").strip()
                    if t:
                        parts.append(t)
    return " ".join(parts)


def _make_id(source_id: str, level_type: str, number: str, parent_id: str | None) -> str:
    safe_num = re.sub(r"[^a-zA-Z0-9._\-]", "", number) or "X"
    safe_type = level_type.lower().replace(" ", "_")
    if parent_id:
        return f"{parent_id}__{safe_type}_{safe_num}"
    return f"{source_id}__{safe_type}_{safe_num}"


def _walk_level(el, source_id: str, parent_id: str | None,
                parent_path: str, depth: int, store: TreeStore) -> None:
    """Recursively walk <level> elements."""
    lt = _local_tag(el)
    if lt != "level":
        return

    level_type = el.get("levelType", "unknown")
    number = _get_child_text(el, "num")
    heading = _get_child_text(el, "heading")

    node_id = _make_id(source_id, level_type, number, parent_id)
    label = f"{level_type} {number}".strip() if number else level_type
    node_path = f"{parent_path} > {label}" if parent_path else label

    text = _collect_text(el)

    node = TreeNode(
        id=node_id,
        node_type=level_type.lower(),
        number=number,
        heading=heading,
        text=text[:2000],
        path=node_path,
        source_id=source_id,
        parent_id=parent_id,
        depth=depth,
    )
    node.extract_cross_refs()
    store.add(node)

    if parent_id and parent_id in store.nodes:
        store.nodes[parent_id].children.append(node_id)

    # Recurse into child <level> elements
    for child in el:
        if _local_tag(child) == "level":
            _walk_level(child, source_id, node_id, node_path, depth + 1, store)


def parse_uslm(xml_path: str, source_id: str | None = None) -> TreeStore:
    """Parse a USLM XML file (or zip containing XML) into a TreeStore."""
    path = Path(xml_path)

    # Handle zip files — extract XML first
    if path.suffix.lower() == ".zip":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(str(path), "r") as zf:
            xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
            if not xml_files:
                raise ValueError(f"No XML files found in {path}")
            # Use the largest XML file (the main title file)
            xml_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
            zf.extract(xml_files[0], tmpdir)
            actual_path = os.path.join(tmpdir, xml_files[0])
    else:
        actual_path = str(path)

    if source_id is None:
        source_id = path.stem.upper().replace("-", "_").replace("@", "_AT_") + "_XML"

    print(f"Parsing {path.name} as {source_id} ...")

    # Parse with recovery mode for large/complex docs
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(actual_path, parser)
    root = tree.getroot()

    store = TreeStore(source_id)

    # Find all top-level <level> elements
    # In USLM, the root may be <usLaw>, <usc>, or <lawDoc> containing <level>
    def find_top_levels(el):
        for child in el:
            if _local_tag(child) == "level":
                _walk_level(child, source_id, None, "", 0, store)
            elif _local_tag(child) in ("main", "body", "legisBody"):
                find_top_levels(child)

    find_top_levels(root)

    # If no levels found via children, try iter
    if not store.nodes:
        for el in root.iter():
            if _local_tag(el) == "level" and el.get("levelType") == "title":
                _walk_level(el, source_id, None, "", 0, store)
                break

    stats = store.depth_stats()
    print(f"  Parsed {stats['count']} nodes, max depth {stats['max_depth']}, avg depth {stats['avg_depth']:.1f}")
    return store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_uslm.py <uslm_xml_or_zip_path> [output_json_path]")
        sys.exit(1)

    xml_file = sys.argv[1]
    store = parse_uslm(xml_file)

    out_path = sys.argv[2] if len(sys.argv) > 2 else Path(xml_file).stem + "_tree.json"
    store.save(out_path)
    print(f"  Saved to {out_path}")
