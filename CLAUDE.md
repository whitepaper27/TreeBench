# CLAUDE.md — TreeBench

## Paper

**Title:** TreeBench: Exposing Structural Evidence Failures in Retrieval over Hierarchical Corpora

**Alt title:** Similarity Is Not Authority: A Benchmark for Hierarchy-Sensitive Retrieval

**Thesis:** The retrieval problem is not semantic relevance — it is authority selection. Similarity-only retrieval fails systematically on hierarchy-sensitive QA. Answer accuracy alone hides structural evidence failure.

**Key concept:** Structural confounder pairs — two nodes that are semantically similar but structurally incompatible. Embedding retrieval cannot distinguish them. Only tree position resolves the correct answer.

**Framing:** Benchmark + negative findings + open challenge. NOT "we solved retrieval."

---

## Current State (as of 2026-05-31)

### Phase 1 — Infrastructure: COMPLETE

```
591,793 tree nodes parsed across 10 eCFR titles (5 domains)
  9,080 candidate failure patterns extracted
  7,635 candidates validated (84% pass rate, depth>=4, text>=50 chars)
```

### Phase 2 — Dataset Generation: COMPLETE

```
TreeBench-861-gold (FINAL)
  861 unique, validated questions
  50/50 cells filled (5 domains × 10 failure types)
  37 cells at exactly 20, 13 cells below 20 (data pool limits)
  Difficulty: 207 easy / 544 medium / 110 hard
  All quality checks pass:
    - 0 required/distractor overlap
    - 0 duplicate question text
    - 0 signal hallucinations
    - 0 mid-word evidence cuts
    - 0 structural giveaways
    - All rows have gold_evidence
    - All rows have 2+ required_node_ids
```

### Phase 3 — Baselines: COMPLETE

```
8 methods evaluated on 861 questions.
Embeddings cached to disk (data/embeddings/, ~900 MB total).
LLM methods checkpointed (data/results/checkpoints/).
Total API cost: $2.74 (GPT-4.1-mini)

RESULTS:

Phase 1: Retrieval-Only Baselines
Method              Acc     Path    Recall  Distr   Cost     Lat
─────────────────────────────────────────────────────────────────
Oracle             100.0%  100.0%  100.0%   0.00   $0.00    0.0s
BM25                76.0%   87.7%   45.1%   0.12   $0.00   24.7s
Hybrid RAG          75.8%   87.7%   44.3%   0.13   $0.00   58.3s
Dense RAG           53.1%   70.3%   34.1%   0.08   $0.00   36.1s
Reranker RAG        36.4%   49.6%   23.0%   0.06   $0.00   61.1s

Phase 2: LLM Reasoning & Structure-Aware Baselines
─────────────────────────────────────────────────────────────────
RAG + CoT           71.8%   70.3%   34.1%   0.08   $0.001   4.1s
RAG + Judge         56.8%   70.3%   34.1%   0.08   $0.002   6.2s
Tree Traversal       2.0%    2.4%    1.0%   0.00   $0.001   1.2s
```

### Phase 4 — Paper: IN PROGRESS

---

## Key Findings (for paper)

1. **The task is answerable**: Oracle = 100% accuracy, 100% recall
2. **Standard retrieval looks good on accuracy**: BM25/Hybrid ≈ 76%
3. **But misses structural evidence**: Best recall only ≈ 45%
4. **Dense is weaker than lexical here**: Dense RAG = 53% acc / 34% recall
5. **Reranking makes it worse**: Reranker = 36% acc / 23% recall (reinforces confounders)
6. **CoT doesn't fix retrieval**: RAG + CoT = 72% acc / 34% recall (same recall as Dense)
7. **Judge doesn't fix authority errors**: RAG + Judge = 57% acc / 34% recall
8. **Naive tree traversal is hard**: Tree Traversal = 2% acc / 1% recall (open challenge)

**Headline finding:**
> Hybrid RAG reaches ~76% answer accuracy but only ~44% required-node recall. No evaluated method recovers the full structural evidence path.

---

## File Structure

```
Dataset/
  data/
    raw/              398 MB  — 10 eCFR XML source files
    parsed/           761 MB  — 591,793 tree nodes as JSON adjacency lists
    questions/        8.6 MB  — 9,080 failure-pattern candidates
    validated/                — 7,635 quality-ranked candidates (30/cell, 50 cells)
    embeddings/       900 MB  — cached sentence-transformer embeddings (10 titles)
    pilot/
      treebench_1000_gold.json     — TreeBench-861-gold (FINAL dataset)
      treebench_1000_candidate.json — 1000 pre-validation candidates
      treebench_1000_rejected.json  — 61 rejected rows
    results/
      baseline_results_latest.json  — combined 8-method results
      checkpoints/                  — per-method checkpoints with timestamps
  src/
    tree_node.py              — TreeNode + TreeStore data model
    parse_ecfr.py             — eCFR XML parser (DIV1-DIV8)
    parse_uslm.py             — US Code USLM parser
    pattern_hunters.py        — 10 taxonomy failure pattern hunters
    question_schema.py        — Question schema with structural_confounder_type
    candidate_validator.py    — Quality-rank candidates (depth, overlap, siblings)
    pilot_generator_v3.py     — v3 generator (factual case masking, signal verify)
    pilot_validate_v3.py      — Post-gen validation (overlap, difficulty, evidence)
    promote_to_gold.py        — Candidate → gold promotion (dedup, signal fix)
    finalize_gold.py          — Final cleanup (trim overflow, answer_type fix)
    baseline_runner.py        — Harness + metrics + checkpointing + results
    retrieval_baselines.py    — BM25, Dense RAG, Hybrid, Reranker, Oracle
    reasoning_baselines.py    — RAG+CoT, RAG+Judge, Tree Traversal (GPT-4.1-mini)
    run_baselines.py          — Parallel runner (Phase 1 + Phase 2)
    tree_classifier.py        — Tree-walk engine + product search eval
```

---

## Failure Taxonomy (10 Types)

| # | Type | Signal | Confounder | Description |
|---|------|--------|------------|-------------|
| 1 | Override Chain | except, notwithstanding | parent_child | Child overrides parent rule |
| 2 | Scope Disambiguation | same term 2+ subtrees | sibling | Position determines scope |
| 3 | Cross-Reference | §XXX, see section | cross_reference | Must follow pointer to other subtree |
| 4 | Conditional Cascade | if...then nested | parent_child | Gated by ancestor conditions |
| 5 | Temporal Layering | effective dates | temporal | Date qualifier changes outcome |
| 6 | Sibling Conflict | conflicting shall/shall-not | sibling | Relative position resolves |
| 7 | Definitional Dependency | as defined in | definition | Term defined in separate subtree |
| 8 | Aggregation | sum of, combined with | cross_reference | Values from multiple branches |
| 9 | Negative Space | coverage gap | missing_scope | No provision exists |
| 10 | Depth-Gated Specificity | rate/amount at leaf | parent_child | Value only at leaf level |

---

## Paper Contribution Stack

1. **TreeBench-861 benchmark** — hierarchy-sensitive QA across 5 domains, 10 failure types
2. **Failure taxonomy** — 10 structural failure modes with confounder scoring
3. **Structural evidence metrics** — required-node recall, path accuracy (not just answer accuracy)
4. **Negative finding** — 76% accuracy + 45% recall = structural blindness
5. **Reasoning failure finding** — CoT/Judge don't recover missing authority evidence
6. **Open challenge** — no current method recovers full structural evidence path

---

## Paper Assets Needed

| Asset | Purpose | Status |
|-------|---------|--------|
| Figure 1: Construction pipeline | eCFR → mining → validation → TreeBench-861 | TODO |
| Figure 2: 5×10 coverage heatmap | Domain × failure type count | TODO |
| Figure 3: Accuracy delta heatmap | Oracle minus each method by cell | TODO |
| Figure 4: Required-node recall heatmap | Structural failure by failure type | TODO |
| Figure 5: Baseline comparison bar chart | All 8 methods side by side | TODO |
| Table 1: Failure taxonomy | 10 types with definitions | TODO |
| Table 2: Dataset statistics | 861 questions, domains, difficulty, depth | TODO |
| Table 3: Main baseline results | All 7 metrics × 8 methods | DONE (data ready) |
| Table 4: Failure-type breakdown | Which structural traps hurt most | TODO |

---

## What NOT to Do

- Do NOT call it TreeBench-1000. It is TreeBench-861.
- Do NOT claim tree traversal beats RAG. It doesn't (2% acc).
- Do NOT overclaim "fully human-authored." It is automated + validated.
- Do NOT frame as "we solved retrieval." Frame as "we expose a structural failure mode."
- Do NOT regenerate questions. Dataset is frozen.
- Do NOT re-run baselines without checkpointing.

---

## Experiment Matrix

| Method | What it tests | Result |
|--------|---------------|--------|
| BM25 | Lexical retrieval | 76% acc / 45% recall |
| Dense RAG | Embedding retrieval | 53% acc / 34% recall |
| Hybrid | Combined retrieval | 76% acc / 44% recall |
| Reranker | Better scoring | 36% acc / 23% recall (WORSE) |
| RAG + CoT | Reasoning over context | 72% acc / 34% recall (same recall) |
| RAG + Judge | Post-hoc validation | 57% acc / 34% recall (same recall) |
| Tree Traversal | Structure-aware nav | 2% acc / 1% recall (hard) |
| Oracle | Upper bound | 100% acc / 100% recall |

**Critical finding:** Methods 1-6 fail at 34-45% recall because the failure is UPSTREAM of reasoning. CoT/Judge/Reranker cannot fix wrong context. The problem is context selection, not reasoning capability.
