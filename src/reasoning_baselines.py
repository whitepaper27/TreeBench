"""Reasoning baselines — RAG + CoT, RAG + Judge, Tree traversal.

Methods 5-7 use LLM calls. Uses OpenAI GPT-4.1 via openai SDK.
"""

from __future__ import annotations
import json, re, time, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_runner import MethodResult, load_store
from retrieval_baselines import dense_retrieve, _get_dense_index, _get_embedder
from tree_node import TreeStore, TreeNode
import numpy as np

# LLM config
MODEL = "gpt-4.1-mini"
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _llm_call(prompt: str, max_tokens: int = 500) -> tuple[str, float]:
    """Call GPT-4.1. Returns (response_text, cost_usd)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    # GPT-4.1-mini pricing: $0.40/M input, $1.60/M output
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 0.40 + output_tokens * 1.60) / 1_000_000
    return text, cost


# ──────────────────────────────────────────────────────────────────────
# Method 5: Dense RAG + CoT reasoning
# ──────────────────────────────────────────────────────────────────────

def rag_cot(question: dict, store: TreeStore) -> MethodResult:
    """Dense RAG retrieval + chain-of-thought reasoning over context."""
    t0 = time.time()

    # Retrieve top-10
    base_result = dense_retrieve(question, store)
    retrieved = base_result.retrieved_ids

    # Build context from retrieved nodes
    context_parts = []
    for nid in retrieved[:5]:  # top 5 for context window
        node = store.get(nid)
        if node:
            context_parts.append(f"[{node.path}]\n{node.text[:400]}")
    context = "\n---\n".join(context_parts)

    # CoT prompt
    prompt = f"""You are a regulatory expert answering a question using the provided context.

CONTEXT (retrieved regulatory provisions):
{context}

QUESTION: {question['question']}

Think step by step:
1. What does the question ask?
2. Which provision(s) in the context are relevant?
3. Do any provisions qualify, override, or condition other provisions?
4. What is the complete answer?

ANSWER:"""

    answer, cost = _llm_call(prompt, max_tokens=400)

    return MethodResult(
        question_id=question["question_id"],
        method="rag_cot",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=cost,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 6: Dense RAG + Judge/Verifier
# ──────────────────────────────────────────────────────────────────────

def rag_judge(question: dict, store: TreeStore) -> MethodResult:
    """Dense RAG + CoT answer + judge verification step."""
    t0 = time.time()

    # First: get CoT answer
    base_result = dense_retrieve(question, store)
    retrieved = base_result.retrieved_ids

    context_parts = []
    for nid in retrieved[:5]:
        node = store.get(nid)
        if node:
            context_parts.append(f"[{node.path}]\n{node.text[:400]}")
    context = "\n---\n".join(context_parts)

    # Generate answer
    gen_prompt = f"""Answer this regulatory question using ONLY the provided context.

CONTEXT:
{context}

QUESTION: {question['question']}

ANSWER:"""

    answer, cost1 = _llm_call(gen_prompt, max_tokens=300)

    # Judge step: verify the answer
    judge_prompt = f"""You are a regulatory compliance judge. Verify whether this answer is correct and complete.

CONTEXT:
{context}

QUESTION: {question['question']}

PROPOSED ANSWER: {answer}

Evaluate:
1. Is the answer supported by the provided context?
2. Are there provisions in the context that contradict or qualify this answer?
3. Is the answer complete, or is critical information missing?

VERDICT (CORRECT/INCORRECT/INCOMPLETE):
CORRECTED ANSWER (if needed):"""

    verdict, cost2 = _llm_call(judge_prompt, max_tokens=400)

    # Use corrected answer if judge says incorrect
    final_answer = answer
    if "INCORRECT" in verdict.upper() or "INCOMPLETE" in verdict.upper():
        # Extract corrected answer
        corrected = re.search(r"CORRECTED ANSWER[:\s]*(.*)", verdict, re.DOTALL)
        if corrected:
            final_answer = corrected.group(1).strip()

    return MethodResult(
        question_id=question["question_id"],
        method="rag_judge",
        retrieved_ids=retrieved,
        answer_text=final_answer,
        cost=cost1 + cost2,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 7: Tree traversal (LLM-guided tree walk)
# ──────────────────────────────────────────────────────────────────────

def _find_best_root(store: TreeStore, question_text: str) -> TreeNode | None:
    """Find the best starting root node by keyword overlap with question.

    Skips identical volume-level roots (e.g., Title 26 Volume 1-22)
    by looking at their children/grandchildren for distinguishing content.
    """
    query_words = set(re.findall(r"[a-z]{3,}", question_text.lower()))
    best_node = None
    best_score = -1

    for rid in store.root_ids:
        root = store.get(rid)
        if not root:
            continue
        # Collect text from this root + its first 2 levels of children
        texts = [root.text[:200]]
        for cid in root.children[:10]:
            child = store.get(cid)
            if child:
                texts.append(child.text[:200])
                texts.append(child.heading or "")
                for gcid in child.children[:10]:
                    gc = store.get(gcid)
                    if gc:
                        texts.append(gc.heading or "")
                        texts.append(gc.text[:100])
        combined = " ".join(texts).lower()
        child_words = set(re.findall(r"[a-z]{3,}", combined))
        score = len(query_words & child_words)
        if score > best_score:
            best_score = score
            best_node = root

    return best_node


def tree_traversal(question: dict, store: TreeStore) -> MethodResult:
    """Structure-aware tree walk using LLM to select child branches.

    Handles many-children nodes by:
    1. Pre-selecting best root via keyword overlap (skips identical volumes)
    2. Auto-descending single-child levels
    3. Capping shown children at 15 with keyword pre-filter for large fan-outs
    """
    t0 = time.time()
    total_cost = 0.0
    MAX_SHOWN = 15

    # Find best root instead of showing all roots to LLM
    best_root = _find_best_root(store, question["question"])
    if best_root:
        current_node = best_root
        visited_path = [current_node.id]
    else:
        current_node = None
        visited_path = []

    for depth in range(12):
        if current_node:
            child_ids = current_node.children
        else:
            child_ids = store.root_ids

        if not child_ids:
            break

        children = [store.nodes[cid] for cid in child_ids if cid in store.nodes]
        if not children:
            break

        # Single child: auto-descend
        if len(children) == 1:
            current_node = children[0]
            visited_path.append(current_node.id)
            continue

        # For large fan-outs: pre-filter by keyword overlap
        if len(children) > MAX_SHOWN:
            query_words = set(re.findall(r"[a-z]{3,}", question["question"].lower()))
            scored = []
            for ch in children:
                ch_text = (ch.heading + " " + ch.text[:200]).lower()
                ch_words = set(re.findall(r"[a-z]{3,}", ch_text))
                score = len(query_words & ch_words)
                scored.append((score, ch))
            scored.sort(key=lambda x: x[0], reverse=True)
            children = [ch for _, ch in scored[:MAX_SHOWN]]

        # Build classification prompt
        child_desc = []
        for i, ch in enumerate(children):
            label = f"[{i+1}] {ch.node_type.upper()} {ch.number}"
            if ch.heading:
                label += f" — {ch.heading}"
            preview = ch.text[:200].replace("\n", " ").strip()
            if preview:
                label += f"\n    {preview}"
            child_desc.append(label)

        ancestor_str = " > ".join(
            f"{store.get(nid).node_type} {store.get(nid).number}"
            for nid in visited_path[-3:]
            if store.get(nid)
        ) if visited_path else "(root)"

        prompt = f"""Navigate a regulatory document tree to find the provision that answers this question.

QUESTION: {question['question']}

CURRENT LOCATION: {ancestor_str}

AVAILABLE BRANCHES:
{chr(10).join(child_desc)}

Which branch most likely contains the answer? Reply with ONLY the number (1-{len(children)}), or 0 if none."""

        response, cost = _llm_call(prompt, max_tokens=10)
        total_cost += cost

        nums = re.findall(r"\d+", response)
        selection = int(nums[0]) if nums else 0

        if selection == 0 or selection > len(children):
            break

        current_node = children[selection - 1]
        visited_path.append(current_node.id)

    # Build answer from final node + ancestors
    answer = ""
    if current_node:
        answer = current_node.text[:500]
        ancestors = store.ancestors(current_node.id)
        if ancestors:
            ancestor_text = " | ".join(
                f"{a.node_type} {a.number}: {a.text[:100]}" for a in ancestors[-3:]
            )
            answer = f"[Path: {ancestor_text}]\n{answer}"

    return MethodResult(
        question_id=question["question_id"],
        method="tree_traversal",
        retrieved_ids=visited_path,
        answer_text=answer,
        cost=total_cost,
        latency=time.time() - t0,
    )
