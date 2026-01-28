#!/usr/bin/env python3
"""
ProteinGym Data Downloader

Downloads ProteinGym benchmark datasets for variant effect prediction evaluation.
Supports both DMS (Deep Mutational Scanning) substitution and indel benchmarks.

Usage:
    python download_data.py --output_dir ./data
    python download_data.py --dataset substitutions --output_dir ./data
    python download_data.py --use_huggingface --output_dir ./data
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
from pathlib import Path


# ProteinGym download URLs (version 1.3)
PROTEINGYM_VERSION = "v1.3"
BASE_URL = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_VERSION}"

DATASETS = {
    "substitutions": {
        "url": f"{BASE_URL}/DMS_ProteinGym_substitutions.zip",
        "filename": "DMS_ProteinGym_substitutions.zip",
        "size_mb": 1000,
        "description": "DMS substitution benchmark (~2.7M variants, 217 assays)"
    },
    "indels": {
        "url": f"{BASE_URL}/DMS_ProteinGym_indels.zip",
        "filename": "DMS_ProteinGym_indels.zip",
        "size_mb": 200,
        "description": "DMS indels benchmark"
    },
    "reference": {
        "url": f"{BASE_URL}/ProteinGym_reference_files.zip",
        "filename": "ProteinGym_reference_files.zip",
        "size_mb": 10,
        "description": "Reference files with assay metadata and UniProt mappings"
    },
    "zero_shot_scores": {
        "url": f"{BASE_URL}/zero_shot_substitutions_scores.zip",
        "filename": "zero_shot_substitutions_scores.zip",
        "size_mb": 4400,
        "description": "Pre-computed zero-shot model scores for comparison"
    },
    "msa": {
        "url": f"{BASE_URL}/MSA_ProteinGym.zip",
        "filename": "MSA_ProteinGym.zip",
        "size_mb": 5200,
        "description": "Multiple Sequence Alignments"
    },
    "structures": {
        "url": f"{BASE_URL}/AlphaFold2_structures.zip",
        "filename": "AlphaFold2_structures.zip",
        "size_mb": 84,
        "description": "AlphaFold2 structure predictions"
    }
}


def download_progress_hook(count, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve"""
    percent = int(count * block_size * 100 / total_size)
    percent = min(percent, 100)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress indication"""
    print(f"\n{'='*60}")
    print(f"Downloading: {description or url}")
    print(f"Destination: {output_path}")
    print(f"{'='*60}")

    try:
        urllib.request.urlretrieve(url, output_path, download_progress_hook)
        print(f"\nDownload complete: {output_path}")
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip file"""
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_from_huggingface(output_dir: Path) -> bool:
    """Download ProteinGym data from Hugging Face (requires datasets library)"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False

    print("\nDownloading from Hugging Face...")

    try:
        # Load DMS substitutions
        print("Loading DMS_substitutions...")
        dms_subs = load_dataset("OATML-Markslab/ProteinGym_v1", name="DMS_substitutions")

        # Save to parquet for efficient storage
        output_dir.mkdir(parents=True, exist_ok=True)
        subs_path = output_dir / "DMS_substitutions.parquet"
        dms_subs['test'].to_parquet(str(subs_path))
        print(f"Saved to: {subs_path}")

        return True
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        return False


def download_dataset(dataset_name: str, output_dir: Path, keep_zip: bool = False) -> bool:
    """Download and extract a specific dataset"""
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False

    dataset = DATASETS[dataset_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / dataset["filename"]

    # Download
    success = download_file(
        dataset["url"],
        zip_path,
        f"{dataset_name} - {dataset['description']} (~{dataset['size_mb']}MB)"
    )

    if not success:
        return False

    # Extract
    success = extract_zip(zip_path, output_dir)

    # Cleanup
    if success and not keep_zip:
        print(f"Removing zip file: {zip_path}")
        zip_path.unlink()

    return success


def verify_data(output_dir: Path) -> dict:
    """Verify downloaded data and return statistics"""
    stats = {
        "substitutions": {"csv_files": 0, "total_variants": 0},
        "reference": {"files": []},
        "structures": {"pdb_files": 0}
    }

    # Check substitutions
    subs_dir = output_dir / "DMS_ProteinGym_substitutions"
    if subs_dir.exists():
        csv_files = list(subs_dir.glob("*.csv"))
        stats["substitutions"]["csv_files"] = len(csv_files)

        # Count variants in a sample
        if csv_files:
            import pandas as pd
            sample = pd.read_csv(csv_files[0])
            print(f"\nSample file columns: {list(sample.columns)}")

    # Check reference files
    ref_dir = output_dir / "reference_files"
    if ref_dir.exists():
        stats["reference"]["files"] = [f.name for f in ref_dir.iterdir()]

    # Check structures
    struct_dir = output_dir / "AlphaFold2_structures"
    if struct_dir.exists():
        stats["structures"]["pdb_files"] = len(list(struct_dir.glob("*.pdb")))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download ProteinGym benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download substitution benchmark (minimal for ES Score evaluation)
    python download_data.py --dataset substitutions reference

    # Download all datasets
    python download_data.py --all

    # Use Hugging Face (requires 'datasets' library)
    python download_data.py --use_huggingface
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        default=["substitutions", "reference"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Use Hugging Face datasets library instead of direct download"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep zip files after extraction"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable ProteinGym datasets:")
        print("="*60)
        for name, info in DATASETS.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Size: ~{info['size_mb']}MB")
        return

    output_dir = Path(args.output_dir)

    if args.use_huggingface:
        success = download_from_huggingface(output_dir)
    else:
        datasets = list(DATASETS.keys()) if args.all else args.dataset

        print(f"\nWill download: {datasets}")
        print(f"Output directory: {output_dir}")

        for dataset_name in datasets:
            success = download_dataset(dataset_name, output_dir, args.keep_zip)
            if not success:
                print(f"Failed to download {dataset_name}")

    # Verify and show stats
    print("\n" + "="*60)
    print("Verification")
    print("="*60)
    stats = verify_data(output_dir)

    if stats["substitutions"]["csv_files"] > 0:
        print(f"Substitution assays: {stats['substitutions']['csv_files']} CSV files")

    if stats["reference"]["files"]:
        print(f"Reference files: {stats['reference']['files']}")

    if stats["structures"]["pdb_files"] > 0:
        print(f"Structure files: {stats['structures']['pdb_files']} PDB files")

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
