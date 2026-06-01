"""Generate all figures for the TreeBench paper.

Usage:
  python generate_figures.py
  python generate_figures.py --gold data/treebench_v1_861_gold.json --results data/results/checkpoints --out paper/figures

Outputs PDF + PNG figures.
Requires: matplotlib, seaborn, numpy.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate TreeBench paper figures")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "data" / "pilot" / "treebench_v1_861_gold.json",
                        help="Path to gold dataset JSON")
    parser.add_argument("--results", type=Path,
                        default=ROOT / "data" / "results" / "checkpoints",
                        help="Path to checkpoint results directory")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "figures",
                        help="Output directory for figures")
    return parser.parse_args()

args = parse_args()
GOLD_FILE = args.gold
CHECKPOINTS = args.results
FIG_DIR = args.out
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load data — fall back to legacy filename (pre-rename) if canonical name not found
if not GOLD_FILE.exists():
    legacy = GOLD_FILE.parent / "treebench_1000_gold.json"  # legacy name, contains 861 rows
    if legacy.exists():
        print(f"  [warn] Using legacy filename: {legacy.name}")
        GOLD_FILE = legacy
    else:
        sys.exit(f"Gold file not found: {GOLD_FILE}")

gold = json.loads(GOLD_FILE.read_text(encoding='utf-8'))
q_map = {q['question_id']: q for q in gold}

METHODS = ['oracle', 'bm25', 'hybrid_rag', 'dense_rag', 'reranker_rag', 'rag_cot', 'rag_judge', 'tree_traversal']
METHOD_LABELS = {
    'oracle': 'Oracle',
    'bm25': 'BM25',
    'hybrid_rag': 'Hybrid RAG',
    'dense_rag': 'Dense RAG',
    'reranker_rag': 'Reranker',
    'rag_cot': 'RAG + CoT',
    'rag_judge': 'RAG + Judge',
    'tree_traversal': 'Tree Traversal',
}

FAILURE_TYPES = [
    'override_chain', 'scope_disambiguation', 'cross_reference',
    'conditional_cascade', 'temporal_layering', 'sibling_conflict',
    'definitional_dependency', 'aggregation', 'negative_space',
    'depth_gated_specificity'
]
FT_LABELS = {
    'override_chain': 'Override',
    'scope_disambiguation': 'Scope',
    'cross_reference': 'Cross-Ref',
    'conditional_cascade': 'Cond. Cascade',
    'temporal_layering': 'Temporal',
    'sibling_conflict': 'Sibling',
    'definitional_dependency': 'Definitional',
    'aggregation': 'Aggregation',
    'negative_space': 'Neg. Space',
    'depth_gated_specificity': 'Depth-Gated',
}

DOMAINS = ['tax', 'finance', 'medical', 'legal', 'compliance']
DOMAIN_LABELS = {'tax': 'Tax', 'finance': 'Finance', 'medical': 'Medical', 'legal': 'Legal', 'compliance': 'Compliance'}


def load_checkpoint(method):
    f = CHECKPOINTS / f"{method}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding='utf-8'))


def compute_metrics(data):
    n = len(data)
    if n == 0:
        return {'acc': 0, 'path': 0, 'recall': 0, 'distr': 0}
    return {
        'acc': sum(1 for r in data if r.get('answer_correct')) / n,
        'path': sum(1 for r in data if r.get('path_correct')) / n,
        'recall': sum(r.get('required_recall', 0) for r in data) / n,
        'distr': sum(r.get('distractor_hits', 0) for r in data) / n,
    }


# ─── Figure 2: Coverage Heatmap (Domain × Failure Type) ───────────────────

def fig2_coverage_heatmap():
    matrix = np.zeros((len(DOMAINS), len(FAILURE_TYPES)))
    for q in gold:
        d_idx = DOMAINS.index(q['domain'])
        ft_idx = FAILURE_TYPES.index(q['failure_type'])
        matrix[d_idx, ft_idx] += 1

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=[FT_LABELS[ft] for ft in FAILURE_TYPES],
                yticklabels=[DOMAIN_LABELS[d] for d in DOMAINS],
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Questions'})
    ax.set_title('TreeBench-861: Question Coverage (Domain × Failure Type)')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_coverage_heatmap.pdf')
    fig.savefig(FIG_DIR / 'fig2_coverage_heatmap.png')
    plt.close()
    print("  Figure 2: coverage heatmap")


# ─── Figure 3: Accuracy Delta Heatmap (Oracle - Method) ───────────────────

def fig3_accuracy_delta():
    methods_show = ['bm25', 'dense_rag', 'hybrid_rag', 'reranker_rag', 'rag_cot', 'rag_judge']

    # Compute accuracy by failure type for each method
    matrix = np.zeros((len(methods_show), len(FAILURE_TYPES)))

    for mi, m in enumerate(methods_show):
        data = load_checkpoint(m)
        if not data:
            continue
        by_ft = defaultdict(lambda: {'n': 0, 'acc': 0})
        for r in data:
            q = q_map.get(r['question_id'])
            if not q:
                continue
            ft = q['failure_type']
            by_ft[ft]['n'] += 1
            by_ft[ft]['acc'] += int(r.get('answer_correct', False))
        for fi, ft in enumerate(FAILURE_TYPES):
            d = by_ft[ft]
            if d['n'] > 0:
                matrix[mi, fi] = 1.0 - d['acc'] / d['n']  # delta from oracle (100%)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(matrix * 100, annot=True, fmt='.0f', cmap='Reds',
                xticklabels=[FT_LABELS[ft] for ft in FAILURE_TYPES],
                yticklabels=[METHOD_LABELS[m] for m in methods_show],
                ax=ax, linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy Gap vs Oracle (pp)'})
    ax.set_title('Accuracy Degradation by Failure Type (Oracle = 0)')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_accuracy_delta.pdf')
    fig.savefig(FIG_DIR / 'fig3_accuracy_delta.png')
    plt.close()
    print("  Figure 3: accuracy delta heatmap")


# ─── Figure 4: Required-Node Recall Heatmap ───────────────────────────────

def fig4_recall_heatmap():
    methods_show = ['bm25', 'dense_rag', 'hybrid_rag', 'reranker_rag', 'rag_cot']

    matrix = np.zeros((len(methods_show), len(FAILURE_TYPES)))

    for mi, m in enumerate(methods_show):
        data = load_checkpoint(m)
        if not data:
            continue
        by_ft = defaultdict(lambda: {'n': 0, 'recall': 0})
        for r in data:
            q = q_map.get(r['question_id'])
            if not q:
                continue
            ft = q['failure_type']
            by_ft[ft]['n'] += 1
            by_ft[ft]['recall'] += r.get('required_recall', 0)
        for fi, ft in enumerate(FAILURE_TYPES):
            d = by_ft[ft]
            if d['n'] > 0:
                matrix[mi, fi] = d['recall'] / d['n']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(matrix * 100, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=[FT_LABELS[ft] for ft in FAILURE_TYPES],
                yticklabels=[METHOD_LABELS[m] for m in methods_show],
                ax=ax, linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={'label': 'Required-Node Recall (%)'})
    ax.set_title('Structural Evidence Recovery by Failure Type')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_recall_heatmap.pdf')
    fig.savefig(FIG_DIR / 'fig4_recall_heatmap.png')
    plt.close()
    print("  Figure 4: recall heatmap")


# ─── Figure 5: Baseline Comparison Bar Chart ──────────────────────────────

def fig5_baseline_bars():
    methods_show = ['bm25', 'hybrid_rag', 'dense_rag', 'reranker_rag', 'rag_cot', 'rag_judge', 'tree_traversal']

    accs, recalls, paths = [], [], []
    for m in methods_show:
        data = load_checkpoint(m)
        if not data:
            accs.append(0); recalls.append(0); paths.append(0)
            continue
        metrics = compute_metrics(data)
        accs.append(metrics['acc'] * 100)
        recalls.append(metrics['recall'] * 100)
        paths.append(metrics['path'] * 100)

    x = np.arange(len(methods_show))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width, accs, width, label='Answer Accuracy', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, paths, width, label='Path Accuracy', color='#FF9800', alpha=0.85)
    bars3 = ax.bar(x + width, recalls, width, label='Required-Node Recall', color='#F44336', alpha=0.85)

    ax.set_xlabel('')
    ax.set_ylabel('Score (%)')
    ax.set_title('TreeBench-861: Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_show], rotation=20, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.axhline(y=45.1, color='#F44336', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(6.5, 46.5, 'Best recall (45%)', fontsize=8, color='#F44336', alpha=0.7, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_baseline_bars.pdf')
    fig.savefig(FIG_DIR / 'fig5_baseline_bars.png')
    plt.close()
    print("  Figure 5: baseline comparison bars")


# ─── Figure 6: Authority Gap Visualization ────────────────────────────────

def fig6_authority_gap():
    methods_show = ['bm25', 'hybrid_rag', 'dense_rag', 'reranker_rag', 'rag_cot', 'rag_judge']

    accs, recalls = [], []
    for m in methods_show:
        data = load_checkpoint(m)
        metrics = compute_metrics(data)
        accs.append(metrics['acc'] * 100)
        recalls.append(metrics['recall'] * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods_show))

    ax.bar(x, accs, 0.6, color='#2196F3', alpha=0.3, label='Answer Accuracy')
    ax.bar(x, recalls, 0.6, color='#F44336', alpha=0.85, label='Required-Node Recall')

    # Draw gap annotations
    for i in range(len(methods_show)):
        gap = accs[i] - recalls[i]
        mid = recalls[i] + gap / 2
        ax.annotate(f'{gap:.0f}pp', xy=(i, mid), ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_show], rotation=20, ha='right')
    ax.set_ylabel('Score (%)')
    ax.set_title('The Authority Gap: Accuracy vs. Structural Evidence')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 90)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_authority_gap.pdf')
    fig.savefig(FIG_DIR / 'fig6_authority_gap.png')
    plt.close()
    print("  Figure 6: authority gap")


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating TreeBench paper figures...")
    fig2_coverage_heatmap()
    fig3_accuracy_delta()
    fig4_recall_heatmap()
    fig5_baseline_bars()
    fig6_authority_gap()
    print(f"\nAll figures saved to {FIG_DIR}/")
