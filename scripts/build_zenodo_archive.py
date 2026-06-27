#!/usr/bin/env python3
"""Build a curated zip archive for Zenodo upload.

Usage:
    python scripts/build_zenodo_archive.py

Produces: treebench-861-v1.0.0.zip in the project root.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.0.0"
ARCHIVE_NAME = f"treebench-861-v{VERSION}.zip"
PREFIX = f"treebench-861-v{VERSION}"

# Explicit allowlist of files to include
INCLUDE_FILES = [
    # Root files
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    ".zenodo.json",
    # Dataset
    "data/pilot/treebench_v1_861_gold.json",
    "data/pilot/validation_report_treebench_861.json",
    # Baseline results
    "data/results/baseline_results.json",
    # Source code (evaluation-relevant)
    "src/tree_node.py",
    "src/question_schema.py",
    "src/parse_ecfr.py",
    "src/pattern_hunters.py",
    "src/baseline_runner.py",
    "src/retrieval_baselines.py",
    "src/reasoning_baselines.py",
    "src/run_baselines.py",
    # Scripts
    "scripts/download_tier1.py",
    "scripts/run_pipeline.py",
    # Paper
    "paper/treebench.tex",
    "paper/references.bib",
]

# Figures are globbed separately (PDF + PNG)
FIGURE_GLOB = "paper/figures/*"


def main():
    out_path = ROOT / ARCHIVE_NAME

    # Collect all files
    files: list[tuple[Path, str]] = []

    for rel in INCLUDE_FILES:
        full = ROOT / rel
        if not full.exists():
            print(f"  WARNING: {rel} not found, skipping")
            continue
        files.append((full, f"{PREFIX}/{rel}"))

    for fig in sorted(ROOT.glob(FIGURE_GLOB)):
        if fig.suffix.lower() in (".pdf", ".png"):
            rel = fig.relative_to(ROOT).as_posix()
            files.append((fig, f"{PREFIX}/{rel}"))

    # Build zip
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full_path, arc_name in files:
            zf.write(full_path, arc_name)
            print(f"  + {arc_name}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated {ARCHIVE_NAME} ({size_mb:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
