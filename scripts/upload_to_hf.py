#!/usr/bin/env python3
"""Upload TreeBench-861 to Hugging Face Hub.

Usage:
    # First login (one-time):
    #   huggingface-cli login
    #
    # Then run:
    python scripts/upload_to_hf.py

    # To use a different repo name:
    python scripts/upload_to_hf.py --repo-id sahilsoni/TreeBench
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def convert_to_jsonl(input_path: Path, output_path: Path) -> int:
    """Convert the gold JSON array to JSONL format for HF datasets.

    Drops review_status, review_decision, review_notes (internal fields).
    """
    with open(input_path, encoding="utf-8") as f:
        questions = json.load(f)

    drop_fields = {"review_status", "review_decision", "review_notes"}

    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            row = {k: v for k, v in q.items() if k not in drop_fields}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(questions)


def main():
    parser = argparse.ArgumentParser(description="Upload TreeBench to Hugging Face")
    parser.add_argument(
        "--repo-id",
        default="sahilsoni2409/TreeBench",
        help="HF repo ID (default: whitepaper27/TreeBench)",
    )
    parser.add_argument(
        "--gold-path",
        default="data/pilot/treebench_v1_861_gold.json",
        help="Path to gold dataset JSON",
    )
    parser.add_argument(
        "--dataset-card",
        default="hf_dataset_card.md",
        help="Path to dataset card markdown",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create as private repo (default: public)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    gold_path = root / args.gold_path
    card_path = root / args.dataset_card

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {gold_path}")
    if not card_path.exists():
        raise FileNotFoundError(f"Dataset card not found: {card_path}")

    # --- Prepare staging directory ---
    staging = root / "hf_staging"
    staging.mkdir(exist_ok=True)
    data_dir = staging / "data"
    data_dir.mkdir(exist_ok=True)

    # Convert to JSONL
    jsonl_path = data_dir / "treebench_861_gold.jsonl"
    n = convert_to_jsonl(gold_path, jsonl_path)
    print(f"Converted {n} questions to {jsonl_path}")

    # Copy dataset card as README.md
    readme_path = staging / "README.md"
    shutil.copy2(card_path, readme_path)
    print(f"Copied dataset card to {readme_path}")

    # --- Upload to Hugging Face ---
    api = HfApi()

    # Create repo (no-op if exists)
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    # Upload all files in staging
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload TreeBench-861 gold dataset",
    )
    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")

    # Cleanup staging
    shutil.rmtree(staging)
    print("Cleaned up staging directory.")


if __name__ == "__main__":
    main()
