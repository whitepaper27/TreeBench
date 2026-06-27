---
language:
  - en
license: cc-by-4.0
task_categories:
  - question-answering
task_ids:
  - closed-domain-qa
tags:
  - rag
  - retrieval
  - information-retrieval
  - benchmark
  - legal-nlp
  - regulatory-ai
  - llm-evaluation
  - hierarchy
  - structural-retrieval
pretty_name: TreeBench
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/treebench_861_gold.jsonl
dataset_info:
  features:
    - name: question_id
      dtype: string
    - name: domain
      dtype: string
    - name: source_title
      dtype: string
    - name: failure_type
      dtype: string
    - name: structural_confounder_type
      dtype: string
    - name: question
      dtype: string
    - name: gold_answer
      dtype: string
    - name: author_expected_failure
      dtype: string
    - name: answer_type
      dtype: string
    - name: required_node_ids
      sequence: string
    - name: distractor_node_ids
      sequence: string
    - name: gold_path
      sequence: string
    - name: gold_evidence
      list:
        - name: source
          dtype: string
        - name: section_id
          dtype: string
        - name: node_type
          dtype: string
        - name: evidence_text
          dtype: string
    - name: why_similarity_fails
      dtype: string
    - name: tree_depth
      dtype: int32
    - name: difficulty
      dtype: string
  splits:
    - name: test
      num_examples: 861
---

# TreeBench: A Benchmark for Hierarchy-Sensitive Retrieval

**Similarity is not authority.** TreeBench is a benchmark of 861 questions designed to expose *structural evidence failures* in retrieval-augmented generation (RAG) over hierarchically structured corpora.

Preprint submitted to arXiv and currently under moderation.
GitHub: [https://github.com/whitepaper27/TreeBench](https://github.com/whitepaper27/TreeBench)

## Key Finding

> The best non-oracle method achieves **76% answer accuracy** but only **45% required-node recall** — producing correct-looking answers while missing the structural evidence that authorizes them.

## What is TreeBench?

TreeBench targets a specific failure mode: **structural confounder pairs** — two nodes in a document tree that are semantically similar (high cosine similarity) but structurally incompatible (only one is authoritative for a given query). Standard embedding-based retrieval cannot distinguish them; only tree position resolves the correct answer.

### Dataset Overview

- **861 gold questions** across 5 regulatory domains and 10 structural failure types
- Source corpus: 591,793 tree nodes from 10 U.S. Electronic Code of Federal Regulations (eCFR) titles
- Each question includes: gold answer, required node IDs, distractor node IDs, gold path, and gold evidence

### Domains

| Domain | eCFR Titles | Questions |
|--------|-------------|-----------|
| Tax | 26 (Internal Revenue), 31 (Treasury) | 170 |
| Finance | 12 (Banks), 17 (Securities) | 173 |
| Medical | 21 (Food & Drugs), 42 (Public Health) | 166 |
| Legal | 29 (Labor), 15 (Commerce) | 176 |
| Compliance | 40 (Environment), 45 (HHS/HIPAA) | 176 |

### Failure Taxonomy (10 Types)

| # | Type | Description |
|---|------|-------------|
| 1 | Override Chain | Child provision overrides parent rule |
| 2 | Scope Disambiguation | Tree position determines which definition applies |
| 3 | Cross-Reference | Must follow pointer to controlling provision |
| 4 | Conditional Cascade | Answer gated by ancestor conditions |
| 5 | Temporal Layering | Date qualifier changes applicable rule |
| 6 | Sibling Conflict | Relative position among siblings resolves conflict |
| 7 | Definitional Dependency | Term defined in separate subtree |
| 8 | Aggregation | Values must be collected from multiple branches |
| 9 | Negative Space | Correct answer is that no provision exists |
| 10 | Depth-Gated Specificity | Specific value exists only at maximum depth |

### Difficulty Distribution

| Difficulty | Count |
|------------|-------|
| Easy | 207 |
| Medium | 544 |
| Hard | 110 |

## Baseline Results

| Method | Accuracy | Path Acc. | Req-Node Recall | Distractor Rate |
|--------|----------|-----------|-----------------|-----------------|
| Oracle | 100.0% | 100.0% | 100.0% | 0.00 |
| BM25 | 76.0% | 87.7% | 45.1% | 0.12 |
| Hybrid RAG | 75.8% | 87.7% | 44.3% | 0.13 |
| RAG + CoT | 71.8% | 70.3% | 34.1% | 0.08 |
| RAG + Judge | 56.8% | 70.3% | 34.1% | 0.08 |
| Dense RAG | 53.1% | 70.3% | 34.1% | 0.08 |
| Reranker RAG | 36.4% | 49.6% | 23.0% | 0.06 |
| Tree Traversal | 2.0% | 2.4% | 1.0% | 0.00 |

**Critical finding:** Methods achieve 53-76% answer accuracy but only 23-45% required-node recall. The failure is upstream of reasoning — CoT and Judge cannot fix wrong context. The problem is context selection, not reasoning capability.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("sahilsoni2409/TreeBench", split="test")
print(f"Loaded {len(ds)} questions")
print(ds[0]["question"][:100])
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | string | Unique identifier (e.g., `TB-TITLE26-OVERRIDE-CHAIN-0001`) |
| `domain` | string | One of: tax, finance, medical, legal, compliance |
| `source_title` | string | eCFR title the question draws from |
| `failure_type` | string | One of the 10 structural failure types |
| `structural_confounder_type` | string | Type of confounder: parent_child, sibling, cross_reference, etc. |
| `question` | string | The question text |
| `gold_answer` | string | The correct answer with structural reasoning |
| `author_expected_failure` | string | Why similarity-based retrieval fails on this question |
| `answer_type` | string | One of: yes_no, numeric, classification, multi_part, section_reference |
| `required_node_ids` | list[string] | Node IDs that must be retrieved for a structurally valid answer |
| `distractor_node_ids` | list[string] | Semantically similar but structurally incorrect node IDs |
| `gold_path` | list[string] | Path from root to the authoritative node |
| `gold_evidence` | list[object] | Evidence objects with source, section_id, node_type, evidence_text |
| `why_similarity_fails` | string | Explanation of the structural confounder pair |
| `tree_depth` | int | Depth of the authoritative node in the tree |
| `difficulty` | string | One of: easy, medium, hard |

## Evaluation Metrics

TreeBench evaluates retrieval beyond answer accuracy:

- **Answer Accuracy** — Does the system produce the correct answer?
- **Path Accuracy** — Does the system find the correct position in the tree?
- **Required-Node Recall** — Did the system retrieve all structural evidence nodes? *(primary metric)*
- **Distractor Hit Rate** — Did the system select structural confounders? *(lower is better)*

## Citation

```bibtex
@article{soni2026treebench,
  title={TreeBench: A Benchmark for Hierarchy-Sensitive Retrieval over Structured Regulatory Corpora},
  author={Soni, Sahil},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This dataset is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The accompanying code is released under [MIT](https://opensource.org/licenses/MIT).
