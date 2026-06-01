"""Download Tier 1 eCFR XML sources into data/raw/."""

import os, sys, urllib.request, time
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TIER1_SOURCES = [
    # (filename, url)
    ("ECFR-title12.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-12/ECFR-title12.xml"),
    ("ECFR-title15.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-15/ECFR-title15.xml"),
    ("ECFR-title17.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-17/ECFR-title17.xml"),
    ("ECFR-title21.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-21/ECFR-title21.xml"),
    ("ECFR-title26.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-26/ECFR-title26.xml"),
    ("ECFR-title29.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-29/ECFR-title29.xml"),
    ("ECFR-title31.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-31/ECFR-title31.xml"),
    ("ECFR-title40.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-40/ECFR-title40.xml"),
    ("ECFR-title42.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-42/ECFR-title42.xml"),
    ("ECFR-title45.xml",  "https://www.govinfo.gov/bulkdata/ECFR/title-45/ECFR-title45.xml"),
    # US Code Title 26 (IRC) — zip
    ("usc26.zip", "https://uscode.house.gov/download/releasepoints/us/pl/119/93/xml_usc26@119-93.zip"),
]


def download(filename: str, url: str) -> None:
    dest = RAW_DIR / filename
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  SKIP {filename} (already exists, {size_mb:.1f} MB)")
        return

    print(f"  Downloading {filename} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TreeBench-Research/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        dest.write_bytes(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"    OK — {size_mb:.1f} MB")
    except Exception as e:
        print(f"    FAILED: {e}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(TIER1_SOURCES)} sources to {RAW_DIR}\n")

    for filename, url in TIER1_SOURCES:
        download(filename, url)
        time.sleep(1)  # Be polite

    print("\nDone.")
    # Show what we have
    for f in sorted(RAW_DIR.iterdir()):
        print(f"  {f.name:30s}  {f.stat().st_size / (1024*1024):8.1f} MB")


if __name__ == "__main__":
    main()
