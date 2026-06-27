[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20978266.svg)](https://doi.org/10.5281/zenodo.20978266)

# TreeBench: A Benchmark for Hierarchy-Sensitive Retrieval over Structured Regulatory Corpora

**Similarity is not authority.** TreeBench is a benchmark of 861 questions designed to expose *structural evidence failures* in retrieval-augmented generation (RAG) over hierarchically structured corpora.

## Key Finding

> The best non-oracle method achieves **76% answer accuracy** but only **45% required-node recall** — producing correct-looking answers while missing the structural evidence that authorizes them.

| Method | Accuracy | Path Acc. | Recall | Distr. |
|--------|----------|-----------|--------|--------|
| Oracle | 100.0% | 100.0% | 100.0% | 0.00 |
| BM25 | 76.0% | 87.7% | 45.1% | 0.12 |
| Hybrid RAG | 75.4% | 87.3% | 44.1% | 0.14 |
| Dense RAG | 53.1% | 70.3% | 34.1% | 0.08 |
| Reranker RAG | 34.5% | 47.2% | 21.2% | 0.05 |
| RAG + CoT | 71.8% | 70.3% | 34.1% | 0.08 |
| RAG + Judge | 56.8% | 70.3% | 34.1% | 0.08 |
| Tree Traversal | 2.0% | 2.4% | 1.0% | 0.00 |

## What is TreeBench?

TreeBench targets a specific failure mode: **structural confounder pairs** — two nodes in a document tree that are semantically similar (high cosine similarity) but structurally incompatible (only one is authoritative for a given query). Standard embedding-based retrieval cannot distinguish them; only tree position resolves the correct answer.

### Dataset: TreeBench-861

- **861 gold questions** across 5 regulatory domains and 10 structural failure types
- Source corpus: 591,793 tree nodes from 10 U.S. Electronic Code of Federal Regulations (eCFR) titles
- Each question includes: gold answer, required node IDs, distractor node IDs, gold path, and gold evidence
- Core annotation fields: `required_node_ids`, `distractor_node_ids`, `gold_path`, `gold_evidence`, `review_status`, `failure_type`, `domain`, `question`, and `gold_answer`

### Domains

| Domain | eCFR Titles | Nodes |
|--------|-------------|-------|
| Tax | 26 (Internal Revenue), 31 (Treasury) | 187,451 |
| Finance | 12 (Banks), 17 (Securities) | 112,809 |
| Medical | 21 (Food & Drugs), 42 (Public Health) | 138,622 |
| Legal | 29 (Labor), 15 (Commerce) | 89,441 |
| Compliance | 40 (Environment), 45 (HHS/HIPAA) | 63,470 |

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

## Repository Structure

```
TreeBench/
  data/
    pilot/
      treebench_v1_861_gold.json          # The dataset (861 gold questions)
      validation_report_treebench_861.json # Automated validation results
    results/
      baseline_results.json               # 8-method baseline results
  src/
    tree_node.py              # TreeNode + TreeStore data model
    parse_ecfr.py             # eCFR XML parser
    pattern_hunters.py        # 10 failure-pattern hunters
    question_schema.py        # Question schema
    baseline_runner.py        # Evaluation harness + metrics
    retrieval_baselines.py    # BM25, Dense, Hybrid, Reranker, Oracle
    reasoning_baselines.py    # RAG+CoT, RAG+Judge, Tree Traversal
    run_baselines.py          # Parallel baseline runner
  paper/
    treebench.tex             # Paper manuscript
    references.bib            # References (22 entries)
    figures/                  # Generated figures (PDF + PNG)
  scripts/
    download_tier1.py         # Download eCFR source XML
    run_pipeline.py           # End-to-end pipeline runner
```

## Metrics

TreeBench evaluates retrieval beyond answer accuracy:

- **Answer Accuracy** — Does the system produce the correct answer?
- **Path Accuracy** — Does the system find the correct position in the tree?
- **Required-Node Recall** — Did the system retrieve the structural evidence nodes? *(primary metric)*
- **Distractor Hit Rate** — Did the system select structural confounders?

## Quick Start

### Load the dataset

```python
import json

with open("data/pilot/treebench_v1_861_gold.json") as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions")
print(f"Example: {questions[0]['question'][:100]}...")
```

### Run evaluation

```bash
pip install -r requirements.txt
python src/run_baselines.py
```

### Generate paper figures

```bash
python paper/generate_figures.py \
  --gold data/pilot/treebench_v1_861_gold.json \
  --results data/results/checkpoints \
  --out paper/figures
```

## Citation

```bibtex
@dataset{soni2026treebench861,
  title={TreeBench-861: A Benchmark for Hierarchy-Sensitive Retrieval over Structured Regulatory Corpora},
  author={Soni, Sahil},
  year={2026},
  publisher={Zenodo},
  version={v1.0.0},
  doi={10.5281/zenodo.20978266},
  url={https://doi.org/10.5281/zenodo.20978266}
}
```

## License

The dataset and documentation are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

The source code is released under the MIT License.

The underlying regulatory text is derived from publicly available U.S. eCFR sources. Users should verify source-specific terms before redistributing raw source text.
