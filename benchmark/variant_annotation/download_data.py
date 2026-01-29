#!/usr/bin/env python3
"""
Data Downloader for Variant Annotation Benchmark

Downloads MSK-IMPACT NSCLC data from the clinical-data-mining/variant-annotation repository.

Data files:
- msk_impact_nsclc_clinical.csv: Clinical data with survival outcomes (7,965 patients)
- msk_impact_nsclc_missense_mutations.csv: Missense mutations with VEP annotations
- msk_impact_gam.csv: Gene alteration matrix

Usage:
    python download_data.py --output_dir ./data
    python download_data.py --output_dir ./data --dataset clinical mutations
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import List, Optional

# Repository base URL
REPO_URL = "https://raw.githubusercontent.com/clinical-data-mining/variant-annotation/main"

# Available datasets
DATASETS = {
    "clinical": {
        "url": f"{REPO_URL}/data/msk_impact_nsclc_clinical.csv",
        "filename": "msk_impact_nsclc_clinical.csv",
        "description": "Clinical characteristics and survival data for 7,965 NSCLC patients"
    },
    "mutations": {
        "url": f"{REPO_URL}/data/msk_impact_nsclc_missense_mutations.csv",
        "filename": "msk_impact_nsclc_missense_mutations.csv",
        "description": "Missense mutations with VEP annotations (13 predictors)"
    },
    "gam": {
        "url": f"{REPO_URL}/data/msk_impact_gam.csv",
        "filename": "msk_impact_gam.csv",
        "description": "Gene alteration matrix (binary)"
    }
}


def download_file(url: str, output_path: Path, retries: int = 3, timeout: int = 60) -> bool:
    """
    Download a file from URL with retry logic.

    Args:
        url: URL to download from
        output_path: Path to save the file
        retries: Number of retry attempts
        timeout: Timeout in seconds

    Returns:
        True if download succeeded, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            print(f"Downloading: {output_path.name}")
            print(f"  From: {url}")

            # Create request with headers
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (ES Score Benchmark)'}
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                total_size = response.headers.get('Content-Length')
                if total_size:
                    total_size = int(total_size)
                    print(f"  Size: {total_size / 1024 / 1024:.1f} MB")

                # Download with progress
                downloaded = 0
                chunk_size = 8192
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            pct = downloaded / total_size * 100
                            print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

                print(f"\n  Saved to: {output_path}")
                return True

        except urllib.error.HTTPError as e:
            print(f"  HTTP Error {e.code}: {e.reason}")
            if e.code == 404:
                print(f"  File not found at URL: {url}")
                return False
        except urllib.error.URLError as e:
            print(f"  URL Error: {e.reason}")
        except Exception as e:
            print(f"  Error: {e}")

        if attempt < retries - 1:
            wait_time = 2 ** attempt
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print(f"  Failed to download after {retries} attempts")
    return False


def download_dataset(dataset_name: str, output_dir: Path) -> bool:
    """
    Download a specific dataset.

    Args:
        dataset_name: Name of the dataset (clinical, mutations, gam)
        output_dir: Directory to save the file

    Returns:
        True if download succeeded
    """
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False

    dataset = DATASETS[dataset_name]
    output_path = output_dir / dataset["filename"]

    # Check if file already exists
    if output_path.exists():
        print(f"Dataset already exists: {output_path}")
        return True

    return download_file(dataset["url"], output_path)


def download_all(output_dir: Path, datasets: Optional[List[str]] = None) -> dict:
    """
    Download all or specified datasets.

    Args:
        output_dir: Directory to save files
        datasets: List of dataset names to download (default: all)

    Returns:
        Dictionary mapping dataset name to success status
    """
    if datasets is None:
        datasets = list(DATASETS.keys())

    results = {}
    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Description: {DATASETS[name]['description']}")
        print('='*60)
        results[name] = download_dataset(name, output_dir)

    return results


def verify_download(output_dir: Path) -> dict:
    """
    Verify that all required files exist and can be read.

    Args:
        output_dir: Directory containing downloaded files

    Returns:
        Dictionary with verification results
    """
    import pandas as pd

    results = {}
    for name, dataset in DATASETS.items():
        filepath = output_dir / dataset["filename"]

        if not filepath.exists():
            results[name] = {"status": "missing", "rows": 0, "cols": 0}
            continue

        try:
            # Try to read the file
            df = pd.read_csv(filepath, nrows=5)
            full_df = pd.read_csv(filepath)
            results[name] = {
                "status": "ok",
                "rows": len(full_df),
                "cols": len(full_df.columns),
                "columns": list(full_df.columns)
            }
        except Exception as e:
            results[name] = {"status": f"error: {e}", "rows": 0, "cols": 0}

    return results


def print_verification(results: dict):
    """Print verification results in a formatted way."""
    print("\n" + "="*60)
    print("Download Verification")
    print("="*60)

    for name, info in results.items():
        status_emoji = "OK" if info["status"] == "ok" else "FAIL"
        print(f"\n[{status_emoji}] {name}:")
        print(f"    Status: {info['status']}")
        if info["status"] == "ok":
            print(f"    Rows: {info['rows']:,}")
            print(f"    Columns: {info['cols']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MSK-IMPACT NSCLC data for variant annotation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets
    python download_data.py --output_dir ./data

    # Download specific datasets
    python download_data.py --output_dir ./data --dataset clinical mutations

    # Verify downloaded files
    python download_data.py --output_dir ./data --verify

Available datasets:
    clinical  - Clinical data with survival outcomes (7,965 patients)
    mutations - Missense mutations with 13 VEP annotations
    gam       - Gene alteration matrix (binary)
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        # Remove existing files
        for name in (args.dataset or list(DATASETS.keys())):
            filepath = output_dir / DATASETS[name]["filename"]
            if filepath.exists():
                print(f"Removing existing file: {filepath}")
                filepath.unlink()

    # Download
    results = download_all(output_dir, args.dataset)

    # Print summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")

    # Verify if requested or after download
    if args.verify or all(results.values()):
        try:
            verification = verify_download(output_dir)
            print_verification(verification)
        except ImportError:
            print("\nNote: pandas not available for verification")

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
