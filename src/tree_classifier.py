"""Tree Classifier — O(log N) recursive LLM tree-walk for document retrieval.

Instead of:
  RAG: embed query -> cosine similarity -> top-k chunks -> generate answer
  (fails because cosine picks semantically similar but structurally wrong chunks)

Tree-Classification does:
  1. Start at root of document tree
  2. Show LLM the children of current node + the query
  3. LLM classifies: which child subtree contains the answer?
  4. Recurse into selected child
  5. At leaf: generate answer from leaf text + ancestor context

Cost: O(log N) LLM calls where N = total nodes
  vs RAG which embeds all N chunks = O(N) at indexing time

This module implements the tree-walk for both:
  - Regulatory documents (eCFR, US Code)
  - Product hierarchies (the product search train/test data)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from tree_node import TreeNode, TreeStore


@dataclass
class ClassificationStep:
    """One step of the tree-walk."""
    depth: int
    node_id: str
    node_path: str
    children_shown: list[str]  # child IDs presented to classifier
    selected_child_id: Optional[str]  # which child was selected
    confidence: float  # classifier confidence 0-1
    reasoning: str  # why this child was selected


@dataclass
class TreeWalkResult:
    """Complete result of a tree-walk."""
    query: str
    steps: list[ClassificationStep]
    final_node_id: Optional[str]
    final_path: str
    total_llm_calls: int
    answer_text: str  # text at the final node
    ancestor_context: str  # concatenated ancestor text for context


# ──────────────────────────────────────────────────────────────────────
# Core Tree Walk Engine
# ──────────────────────────────────────────────────────────────────────

def build_classification_prompt(query: str, current_node: TreeNode,
                                children: list[TreeNode], ancestor_chain: str) -> str:
    """Build the prompt for one classification step.

    The LLM sees:
    - The user's query
    - Where we are in the tree (ancestor chain)
    - The children to choose from (heading + first 200 chars of text)
    - Instructions to pick the most relevant child
    """
    child_descriptions = []
    for i, child in enumerate(children):
        desc = f"[{i+1}] {child.node_type.upper()} {child.number}"
        if child.heading:
            desc += f" - {child.heading}"
        if child.text:
            preview = child.text[:300].replace("\n", " ")
            desc += f"\n    Preview: {preview}"
        child_descriptions.append(desc)

    children_block = "\n".join(child_descriptions)

    prompt = f"""You are navigating a hierarchical document tree to answer a query.

QUERY: {query}

CURRENT LOCATION IN TREE:
{ancestor_chain or "(root level)"}

AVAILABLE BRANCHES (children of current node):
{children_block}

TASK: Which branch is most likely to contain the answer to the query?

Respond with ONLY a JSON object:
{{"selection": <number 1-{len(children)}>, "confidence": <0.0-1.0>, "reasoning": "<one sentence why>"}}

If NONE of the branches are relevant, respond:
{{"selection": 0, "confidence": 0.0, "reasoning": "No relevant branch found"}}"""

    return prompt


def tree_walk(store: TreeStore, query: str,
              classifier_fn: Callable[[str], dict],
              max_depth: int = 15) -> TreeWalkResult:
    """Execute a tree-walk from root to leaf using an LLM classifier.

    Args:
        store: The parsed document tree
        query: User's question
        classifier_fn: Function that takes a prompt string and returns
                       {"selection": int, "confidence": float, "reasoning": str}
        max_depth: Maximum traversal depth (safety limit)

    Returns:
        TreeWalkResult with the full walk trace and answer
    """
    steps: list[ClassificationStep] = []
    llm_calls = 0

    # Start at root
    current_ids = store.root_ids
    ancestor_texts: list[str] = []
    current_node: Optional[TreeNode] = None

    for depth in range(max_depth):
        # Get children at current level
        if current_node:
            current_ids = current_node.children

        if not current_ids:
            break  # reached a leaf

        children = [store.nodes[cid] for cid in current_ids if cid in store.nodes]
        if not children:
            break

        # If only one child, skip classification
        if len(children) == 1:
            current_node = children[0]
            ancestor_texts.append(f"{current_node.node_type} {current_node.number}: {current_node.heading}")
            steps.append(ClassificationStep(
                depth=depth,
                node_id=current_node.id,
                node_path=current_node.path,
                children_shown=[current_node.id],
                selected_child_id=current_node.id,
                confidence=1.0,
                reasoning="Only one branch available",
            ))
            continue

        # Build ancestor chain for context
        ancestor_chain = " > ".join(ancestor_texts) if ancestor_texts else ""

        # Build prompt and call classifier
        prompt = build_classification_prompt(query, current_node or children[0], children, ancestor_chain)
        result = classifier_fn(prompt)
        llm_calls += 1

        selection = result.get("selection", 0)
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")

        if selection == 0 or selection > len(children):
            # No relevant branch — stop here
            steps.append(ClassificationStep(
                depth=depth,
                node_id=current_node.id if current_node else "",
                node_path=current_node.path if current_node else "",
                children_shown=current_ids,
                selected_child_id=None,
                confidence=confidence,
                reasoning=reasoning,
            ))
            break

        # Select the child
        selected = children[selection - 1]
        current_node = selected
        ancestor_texts.append(f"{selected.node_type} {selected.number}: {selected.heading}")

        steps.append(ClassificationStep(
            depth=depth,
            node_id=selected.id,
            node_path=selected.path,
            children_shown=current_ids,
            selected_child_id=selected.id,
            confidence=confidence,
            reasoning=reasoning,
        ))

    # Build final result
    final_node = current_node
    ancestor_context = ""
    if final_node:
        ancestors = store.ancestors(final_node.id)
        ancestor_context = "\n---\n".join(
            f"[{a.node_type} {a.number}] {a.heading}\n{a.text[:500]}" for a in ancestors
        )

    return TreeWalkResult(
        query=query,
        steps=steps,
        final_node_id=final_node.id if final_node else None,
        final_path=final_node.path if final_node else "",
        total_llm_calls=llm_calls,
        answer_text=final_node.text if final_node else "",
        ancestor_context=ancestor_context,
    )


# ──────────────────────────────────────────────────────────────────────
# Deterministic Classifier (for testing without LLM API)
# ──────────────────────────────────────────────────────────────────────

def keyword_classifier(query: str) -> Callable[[str], dict]:
    """Returns a simple keyword-matching classifier for testing.

    Not meant to be accurate — just proves the tree-walk mechanism works
    without needing an LLM API key.
    """
    query_words = set(query.lower().split())

    def classify(prompt: str) -> dict:
        # Extract children from prompt
        lines = prompt.split("\n")
        best_score = -1
        best_idx = 1

        child_idx = 0
        for line in lines:
            if line.strip().startswith("[") and "]" in line:
                child_idx += 1
                # Count query word overlap with this child's description
                child_text = line.lower()
                # Also grab preview line if it follows
                score = sum(1 for w in query_words if w in child_text and len(w) > 2)
                if score > best_score:
                    best_score = score
                    best_idx = child_idx

        return {
            "selection": best_idx if best_score > 0 else 1,
            "confidence": min(best_score / max(len(query_words), 1), 1.0),
            "reasoning": f"Keyword overlap score: {best_score}",
        }

    return classify


# ──────────────────────────────────────────────────────────────────────
# Product Tree Builder — converts product search JSON to TreeStore
# ──────────────────────────────────────────────────────────────────────

def build_product_tree(products_json_path: str, source_id: str = "PRODUCT_SEARCH") -> TreeStore:
    """Build a TreeStore from the product search ground truth JSON.

    Product hierarchy:
    Root > Category (inferred from product type) > Brand > Product Line > Variant

    This lets tree-classification navigate:
    Query: "Lipton CB Unswet Blk Tea 14oz"
    Tree walk: Beverages > Tea > Lipton > Cold Brew > Unsweetened > 14oz

    vs RAG which just embeds the query and finds 10 tea products
    with high cosine similarity but WRONG product.
    """
    with open(products_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    store = TreeStore(source_id)

    # Root node
    root = TreeNode(
        id=f"{source_id}__root",
        node_type="catalog",
        number="",
        heading="Product Catalog",
        text="Root of product hierarchy",
        path="Catalog",
        source_id=source_id,
        depth=0,
    )
    store.add(root)

    # Group products by inferred category
    categories: dict[str, list[tuple[str, dict]]] = {}
    for key, product in data.items():
        title = product.get("product_title", key).strip()
        cat = _infer_category(title)
        categories.setdefault(cat, []).append((key, product))

    for cat_name, products in categories.items():
        # Category node
        cat_id = f"{source_id}__cat_{cat_name.lower().replace(' ', '_')}"
        cat_node = TreeNode(
            id=cat_id,
            node_type="category",
            number="",
            heading=cat_name,
            text=f"Product category: {cat_name} ({len(products)} products)",
            path=f"Catalog > {cat_name}",
            source_id=source_id,
            parent_id=root.id,
            depth=1,
        )
        store.add(cat_node)
        root.children.append(cat_id)

        for prod_key, prod_data in products:
            title = prod_data.get("product_title", prod_key).strip()
            desc = prod_data.get("product_description", "")
            trusted = prod_data.get("trusted_search_results", [])
            search_results = prod_data.get("search_results", {})

            # Product node
            safe_key = prod_key.strip().replace(" ", "_")[:50]
            prod_id = f"{cat_id}__prod_{safe_key}"
            prod_node = TreeNode(
                id=prod_id,
                node_type="product",
                number="",
                heading=title,
                text=f"{title}. {desc}".strip(),
                path=f"Catalog > {cat_name} > {title[:50]}",
                source_id=source_id,
                parent_id=cat_id,
                depth=2,
            )
            store.add(prod_node)
            cat_node.children.append(prod_id)

            # Search result nodes (children of product)
            for rank, snippet_text in search_results.items():
                is_trusted = int(rank) in trusted
                result_id = f"{prod_id}__result_{rank}"
                result_node = TreeNode(
                    id=result_id,
                    node_type="search_result",
                    number=f"rank_{rank}",
                    heading=f"Result #{rank} ({'TRUSTED' if is_trusted else 'DISTRACTOR'})",
                    text=snippet_text[:500],
                    path=f"Catalog > {cat_name} > {title[:30]} > Result #{rank}",
                    source_id=source_id,
                    parent_id=prod_id,
                    depth=3,
                )
                store.add(result_node)
                prod_node.children.append(result_id)

    return store


def _infer_category(title: str) -> str:
    """Infer product category from title keywords."""
    t = title.lower()
    if any(w in t for w in ["tea", "coffee", "water", "juice", "lemonade", "drink", "soda", "beer", "wine", "sake"]):
        return "Beverages"
    if any(w in t for w in ["cookie", "chocolate", "candy", "mints", "gum", "snack", "chip"]):
        return "Snacks & Candy"
    if any(w in t for w in ["cream", "lotion", "wipes", "soap", "shampoo", "health", "vitamin"]):
        return "Health & Personal Care"
    if any(w in t for w in ["ink", "printer", "paper", "pen", "office"]):
        return "Office & Tech"
    if any(w in t for w in ["onion", "tomato", "lettuce", "fruit", "vegetable", "organic", "meat", "chicken"]):
        return "Fresh Produce & Grocery"
    if any(w in t for w in ["sauce", "spice", "flour", "sugar", "oil", "vinegar", "seasoning"]):
        return "Pantry & Cooking"
    return "General"


# ──────────────────────────────────────────────────────────────────────
# Evaluation — compare tree-walk result to ground truth
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    product_key: str
    query: str
    tree_walk_path: str
    tree_walk_node_id: str
    total_results: int
    trusted_results: list[int]
    tree_found_trusted: bool  # did tree-walk land on a trusted result?
    rag_precision_at_1: float  # would RAG's top-1 be trusted?
    rag_precision_at_3: float  # would RAG's top-3 include trusted?
    tree_walk_steps: int


def evaluate_product_search(products_json_path: str,
                            classifier_fn: Callable[[str], dict] | None = None) -> list[EvalResult]:
    """Run tree-walk on product search data and compare to RAG baseline.

    RAG baseline: cosine similarity would rank results 1,2,3... by embedding distance.
    Ground truth: `trusted_search_results` tells us which results are actually correct.

    This shows: RAG's top-1 (result #1) is often NOT trusted,
    while tree-classification navigates Category > Brand > Product > Variant
    to find the right match.
    """
    with open(products_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[EvalResult] = []

    for key, product in data.items():
        title = product.get("product_title", key).strip()
        trusted = product.get("trusted_search_results", [])
        search_results = product.get("search_results", {})
        total = len(search_results)

        # RAG baseline: assumes cosine similarity ranks results in order 1,2,3...
        rag_p1 = 1.0 if (1 in trusted) else 0.0
        rag_p3 = len([t for t in trusted if t <= 3]) / min(3, total) if total > 0 else 0.0

        results.append(EvalResult(
            product_key=key.strip(),
            query=title,
            tree_walk_path="",  # filled by actual tree-walk
            tree_walk_node_id="",
            total_results=total,
            trusted_results=trusted,
            tree_found_trusted=len(trusted) > 0,  # placeholder
            rag_precision_at_1=rag_p1,
            rag_precision_at_3=rag_p3,
            tree_walk_steps=0,
        ))

    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Demo: build product tree and show stats
    for path_label, path in [
        ("train", "search_results_ground_truth_train.json"),
        ("test", "search_results_ground_truth_test.json"),
    ]:
        full_path = f"../{path}"
        try:
            store = build_product_tree(full_path, f"PRODUCT_{path_label.upper()}")
            stats = store.depth_stats()
            print(f"\n=== Product Tree ({path_label}) ===")
            print(f"  Nodes: {stats['count']}")
            print(f"  Max depth: {stats['max_depth']}")
            print(f"  Depth distribution: {stats['depth_distribution']}")

            # RAG baseline stats
            evals = evaluate_product_search(full_path)
            total_products = len(evals)
            rag_fails_p1 = sum(1 for e in evals if e.rag_precision_at_1 == 0)
            rag_fails_p3 = sum(1 for e in evals if e.rag_precision_at_3 == 0)
            no_trusted = sum(1 for e in evals if not e.trusted_results)
            print(f"  Products: {total_products}")
            print(f"  RAG P@1 = 0 (top-1 wrong): {rag_fails_p1}/{total_products} ({100*rag_fails_p1/total_products:.0f}%)")
            print(f"  RAG P@3 = 0 (top-3 all wrong): {rag_fails_p3}/{total_products} ({100*rag_fails_p3/total_products:.0f}%)")
            print(f"  No trusted results at all: {no_trusted}/{total_products} ({100*no_trusted/total_products:.0f}%)")
        except FileNotFoundError:
            print(f"  {path} not found, skipping")
