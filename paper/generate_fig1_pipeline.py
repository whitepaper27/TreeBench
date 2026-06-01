"""Generate Figure 1: TreeBench Construction Pipeline diagram.

Usage:
  python generate_fig1_pipeline.py
  python generate_fig1_pipeline.py --out paper/figures

Produces fig1_pipeline.pdf and fig1_pipeline.png.
Requires: matplotlib.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    return parser.parse_args()


def draw_pipeline(ax):
    """Draw the 4-stage construction pipeline."""

    # ── Layout constants ──
    stage_y = 0.52
    box_h = 0.32
    box_w = 0.19
    gap = 0.03
    arrow_color = '#555555'

    # Stage positions (left edges)
    xs = [0.01, 0.24, 0.47, 0.70]

    # Colors
    colors = {
        'source':  '#E3F2FD',   # light blue
        'mining':  '#FFF3E0',   # light orange
        'gen':     '#F3E5F5',   # light purple
        'gold':    '#E8F5E9',   # light green
    }
    border_colors = {
        'source':  '#1565C0',
        'mining':  '#E65100',
        'gen':     '#6A1B9A',
        'gold':    '#2E7D32',
    }

    stages = [
        {
            'title': 'Stage 1\nTree Parsing',
            'items': [
                '10 eCFR XML titles',
                '591,793 tree nodes',
                'Adjacency-list format',
            ],
            'color': colors['source'],
            'border': border_colors['source'],
            'badge': '10 titles',
        },
        {
            'title': 'Stage 2\nFailure Mining',
            'items': [
                '10 pattern hunters',
                'Confounder pair detection',
                'Depth \u2265 4, text \u2265 50 chars',
            ],
            'color': colors['mining'],
            'border': border_colors['mining'],
            'badge': '9,080 candidates',
        },
        {
            'title': 'Stage 3\nQuestion Generation',
            'items': [
                'LLM-generated questions',
                'Required + distractor nodes',
                '6 automated quality checks',
            ],
            'color': colors['gen'],
            'border': border_colors['gen'],
            'badge': '922 validated',
        },
        {
            'title': 'Stage 4\nGold Promotion',
            'items': [
                'Dedup + signal verification',
                'Difficulty calibration',
                '61 rejected \u2192 861 gold',
            ],
            'color': colors['gold'],
            'border': border_colors['gold'],
            'badge': '861 gold',
        },
    ]

    for i, (x, stage) in enumerate(zip(xs, stages)):
        # Main box
        rect = FancyBboxPatch(
            (x, stage_y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.012",
            facecolor=stage['color'],
            edgecolor=stage['border'],
            linewidth=1.8,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Title
        ax.text(x + box_w / 2, stage_y + box_h / 2 - 0.025, stage['title'],
                ha='center', va='top', fontsize=8, fontweight='bold',
                color=stage['border'], transform=ax.transAxes,
                linespacing=1.15)

        # Bullet items
        item_top = stage_y + box_h / 2 - 0.13
        for j, item in enumerate(stage['items']):
            ax.text(x + 0.012, item_top - j * 0.045, f'\u2022 {item}',
                    ha='left', va='top', fontsize=6.8, color='#333333',
                    transform=ax.transAxes)

        # Badge below box
        badge_y = stage_y - box_h / 2 - 0.06
        ax.text(x + box_w / 2, badge_y, stage['badge'],
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=stage['border'],
                          edgecolor='none', alpha=0.9),
                transform=ax.transAxes)

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate(
                '', xy=(xs[i + 1] - 0.008, stage_y),
                xytext=(x + box_w + 0.008, stage_y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(
                    arrowstyle='->', color=arrow_color,
                    lw=2.0, connectionstyle='arc3,rad=0',
                    shrinkA=0, shrinkB=0,
                ),
            )

    # Title bar
    ax.text(0.5, 0.95, 'TreeBench-861 Construction Pipeline',
            ha='center', va='top', fontsize=12, fontweight='bold',
            color='#222222', transform=ax.transAxes)

    # Bottom annotation: source → benchmark
    ax.annotate(
        '', xy=(0.88, 0.12), xytext=(0.12, 0.12),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0,
                        linestyle='--'),
    )
    ax.text(0.5, 0.09, 'Raw regulatory XML  \u2192  Validated hierarchy-sensitive QA benchmark',
            ha='center', va='center', fontsize=7.5, color='#777777',
            fontstyle='italic', transform=ax.transAxes)


def main():
    args = parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    draw_pipeline(ax)

    fig.savefig(args.out / 'fig1_pipeline.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(args.out / 'fig1_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 1: pipeline diagram -> {args.out}/")


if __name__ == '__main__':
    main()
