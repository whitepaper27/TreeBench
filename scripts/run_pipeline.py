"""TreeBench Pipeline — Phase 1-3: Parse → Hunt → Report.

Usage:
    python scripts/run_pipeline.py                    # Process all downloaded XMLs
    python scripts/run_pipeline.py data/raw/ECFR-title26.xml  # Process one file
"""

import sys, os, json
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tree_node import TreeStore
from parse_ecfr import parse_ecfr
from parse_uslm import parse_uslm
from pattern_hunters import run_all_hunters, CandidateMatch

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"
QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "questions"


def process_file(raw_path: Path) -> None:
    """Parse a single XML file and run pattern hunters."""
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    name = raw_path.stem
    tree_path = PARSED_DIR / f"{name}_tree.json"
    candidates_path = QUESTIONS_DIR / f"{name}_candidates.json"

    # Step 1: Parse
    print(f"\n{'='*60}")
    print(f"Processing: {raw_path.name}")
    print(f"{'='*60}")

    if raw_path.suffix.lower() == ".zip" or "usc" in raw_path.name.lower():
        store = parse_uslm(str(raw_path))
    else:
        store = parse_ecfr(str(raw_path))

    store.save(str(tree_path))
    print(f"  Tree saved to {tree_path.name}")

    # Step 2: Hunt
    print(f"\n  Running pattern hunters...")
    results = run_all_hunters(store)

    # Step 3: Save candidates
    candidates_out = {}
    for failure_type, candidates in results.items():
        candidates_out[failure_type] = [asdict(c) for c in candidates[:100]]  # cap per type

    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates_out, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in candidates_out.values())
    print(f"  Candidates saved to {candidates_path.name} ({total} total)")

    # Print summary
    stats = store.depth_stats()
    print(f"\n  Summary for {raw_path.name}:")
    print(f"    Nodes:     {stats['count']:,}")
    print(f"    Max depth: {stats['max_depth']}")
    print(f"    Avg depth: {stats['avg_depth']}")
    print(f"    Candidates: {total}")


def main():
    if len(sys.argv) > 1:
        # Process specific file
        for arg in sys.argv[1:]:
            process_file(Path(arg))
    else:
        # Process all files in data/raw/
        if not RAW_DIR.exists():
            print(f"No raw data directory at {RAW_DIR}")
            print("Run 'python scripts/download_tier1.py' first.")
            sys.exit(1)

        xml_files = sorted(RAW_DIR.glob("*.xml")) + sorted(RAW_DIR.glob("*.zip"))
        if not xml_files:
            print(f"No XML/zip files in {RAW_DIR}")
            sys.exit(1)

        print(f"Found {len(xml_files)} files to process")
        for f in xml_files:
            try:
                process_file(f)
            except Exception as e:
                print(f"  ERROR processing {f.name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
