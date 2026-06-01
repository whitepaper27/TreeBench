"""Retrieval baselines — BM25, Dense RAG, Hybrid, Reranker.

Each node in the parsed tree = one chunk.
Retrieve per-title (each question searches only its source title's nodes).
Embeddings cached to disk (.npy) to avoid re-computing and OOM.
"""

from __future__ import annotations
import json, os, re, time, threading
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from baseline_runner import MethodResult, load_store
from tree_node import TreeStore

TOP_K = 10  # retrieve top-k chunks
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

# In-memory caches
_bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}
_dense_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
_embedder = None
_embed_lock = threading.Lock()  # prevent concurrent embedding of same title


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(store: TreeStore) -> tuple[BM25Okapi, list[str]]:
    """Build or retrieve cached BM25 index for a store."""
    sid = store.source_id
    if sid in _bm25_cache:
        return _bm25_cache[sid]

    node_ids = []
    corpus = []
    for nid, node in store.nodes.items():
        text = node.text.strip()
        if len(text) < 20:
            continue
        node_ids.append(nid)
        corpus.append(_tokenize(text))

    bm25 = BM25Okapi(corpus)
    _bm25_cache[sid] = (bm25, node_ids)
    return bm25, node_ids


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_dense_index(store: TreeStore) -> tuple[np.ndarray, list[str]]:
    """Build or retrieve cached dense embeddings.

    Disk cache: data/embeddings/{source_id}_emb.npy + _ids.json
    Only one thread embeds at a time (lock prevents OOM from parallel embeds).
    """
    sid = store.source_id
    if sid in _dense_cache:
        return _dense_cache[sid]

    os.makedirs(CACHE_DIR, exist_ok=True)
    emb_path = CACHE_DIR / f"{sid}_emb.npy"
    ids_path = CACHE_DIR / f"{sid}_ids.json"

    # Try loading from disk cache
    if emb_path.exists() and ids_path.exists():
        print(f"    Loading cached embeddings for {sid}...", flush=True)
        embeddings = np.load(str(emb_path))
        with open(ids_path, "r") as f:
            node_ids = json.load(f)
        _dense_cache[sid] = (embeddings, node_ids)
        return embeddings, node_ids

    # Compute embeddings (one at a time to avoid OOM)
    with _embed_lock:
        # Double-check after acquiring lock
        if sid in _dense_cache:
            return _dense_cache[sid]

        embedder = _get_embedder()
        node_ids = []
        texts = []
        for nid, node in store.nodes.items():
            text = node.text.strip()
            if len(text) < 20:
                continue
            node_ids.append(nid)
            texts.append(text[:512])

        print(f"    Embedding {len(texts)} nodes for {sid}...", flush=True)
        # Small batch size to limit peak memory
        embeddings = embedder.encode(texts, show_progress_bar=True,
                                     batch_size=128, normalize_embeddings=True)

        # Save to disk
        np.save(str(emb_path), embeddings)
        with open(ids_path, "w") as f:
            json.dump(node_ids, f)
        print(f"    Cached {sid} embeddings to disk ({emb_path.stat().st_size/1024/1024:.0f} MB)", flush=True)

        _dense_cache[sid] = (embeddings, node_ids)
        return embeddings, node_ids


# ──────────────────────────────────────────────────────────────────────
# Method 1: BM25
# ──────────────────────────────────────────────────────────────────────

def bm25_retrieve(question: dict, store: TreeStore) -> MethodResult:
    """BM25 sparse retrieval — no embeddings, no API calls."""
    t0 = time.time()
    bm25, node_ids = _get_bm25(store)

    query_tokens = _tokenize(question["question"])
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    retrieved = [node_ids[i] for i in top_indices if scores[i] > 0]

    # Generate answer from top retrieved node
    answer = ""
    if retrieved:
        top_node = store.get(retrieved[0])
        if top_node:
            answer = top_node.text[:500]

    return MethodResult(
        question_id=question["question_id"],
        method="bm25",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=0.0,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 2: Dense RAG
# ──────────────────────────────────────────────────────────────────────

def dense_retrieve(question: dict, store: TreeStore) -> MethodResult:
    """Dense embedding retrieval using sentence-transformers + FAISS."""
    t0 = time.time()
    embeddings, node_ids = _get_dense_index(store)
    embedder = _get_embedder()

    query_emb = embedder.encode([question["question"][:512]])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    # Cosine similarity via dot product (embeddings are normalized)
    scores = (embeddings @ query_emb.T).flatten()
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    retrieved = [node_ids[i] for i in top_indices]

    answer = ""
    if retrieved:
        top_node = store.get(retrieved[0])
        if top_node:
            answer = top_node.text[:500]

    return MethodResult(
        question_id=question["question_id"],
        method="dense_rag",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=0.0,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 3: Hybrid (BM25 + Dense)
# ──────────────────────────────────────────────────────────────────────

def hybrid_retrieve(question: dict, store: TreeStore,
                    bm25_weight: float = 0.3) -> MethodResult:
    """Hybrid retrieval — weighted combination of BM25 + dense scores."""
    t0 = time.time()

    # BM25 scores
    bm25, bm25_ids = _get_bm25(store)
    query_tokens = _tokenize(question["question"])
    bm25_scores = bm25.get_scores(query_tokens)

    # Dense scores
    embeddings, dense_ids = _get_dense_index(store)
    embedder = _get_embedder()
    query_emb = embedder.encode([question["question"][:512]])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    dense_scores = (embeddings @ query_emb.T).flatten()

    # Build unified score map (node_id -> combined score)
    # BM25 and dense may have different node_id orders but same set
    score_map: dict[str, float] = {}

    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
    for i, nid in enumerate(bm25_ids):
        score_map[nid] = bm25_weight * (bm25_scores[i] / bm25_max)

    # Add dense scores (already [0, 1] since normalized embeddings)
    for i, nid in enumerate(dense_ids):
        score_map[nid] = score_map.get(nid, 0.0) + (1 - bm25_weight) * float(dense_scores[i])

    # Sort by combined score
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    retrieved = [nid for nid, _ in ranked]

    answer = ""
    if retrieved:
        top_node = store.get(retrieved[0])
        if top_node:
            answer = top_node.text[:500]

    return MethodResult(
        question_id=question["question_id"],
        method="hybrid_rag",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=0.0,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 4: Dense RAG + Reranker (cross-encoder reranking)
# ──────────────────────────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reranker_retrieve(question: dict, store: TreeStore) -> MethodResult:
    """Dense RAG + cross-encoder reranker on top-50 candidates."""
    t0 = time.time()

    # First stage: dense retrieval top-50
    embeddings, node_ids = _get_dense_index(store)
    embedder = _get_embedder()
    query_emb = embedder.encode([question["question"][:512]])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores = (embeddings @ query_emb.T).flatten()
    top_50_indices = np.argsort(scores)[-50:][::-1]
    candidates = [(node_ids[i], store.get(node_ids[i])) for i in top_50_indices]
    candidates = [(nid, node) for nid, node in candidates if node]

    # Second stage: cross-encoder rerank
    reranker = _get_reranker()
    pairs = [(question["question"][:256], node.text[:256]) for _, node in candidates]
    if pairs:
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        retrieved = [nid for (nid, _), _ in ranked[:TOP_K]]
    else:
        retrieved = []

    answer = ""
    if retrieved:
        top_node = store.get(retrieved[0])
        if top_node:
            answer = top_node.text[:500]

    return MethodResult(
        question_id=question["question_id"],
        method="reranker_rag",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=0.0,
        latency=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────
# Method 8: Oracle path + LLM
# ──────────────────────────────────────────────────────────────────────

def oracle_retrieve(question: dict, store: TreeStore) -> MethodResult:
    """Oracle — give the gold evidence directly. Upper bound."""
    t0 = time.time()

    # Use required_node_ids as retrieved set
    retrieved = [nid for nid in question.get("required_node_ids", [])
                 if "__ref_" not in nid]

    # Build answer from gold evidence text
    evidence_texts = []
    for ev in question.get("gold_evidence", []):
        et = ev.get("evidence_text", "").strip()
        if et:
            evidence_texts.append(et)

    answer = " ".join(evidence_texts) if evidence_texts else ""

    return MethodResult(
        question_id=question["question_id"],
        method="oracle",
        retrieved_ids=retrieved,
        answer_text=answer,
        cost=0.0,
        latency=time.time() - t0,
    )
