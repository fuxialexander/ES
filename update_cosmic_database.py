#!/usr/bin/env python
"""
Update COSMIC Database from v2015 to v103 (November 2024)

This script orchestrates the complete update of COSMIC mutation data for the
ES Score pipeline. It downloads, processes, and validates the new COSMIC v103
database.

Usage:
    python update_cosmic_database.py [--skip-download] [--skip-process]

Requirements:
    - COSMIC account credentials (set up as environment variables or in script)
    - pandas, requests

Files created:
    - /mnt/storage/ES/raw/Cosmic_GenomeScreensMutant_v103_GRCh38.tar
    - /mnt/storage/ES/raw/Cosmic_MutantCensus_v103_GRCh38.tar
    - CosmicMutantExport.missense.aachange.tsv (processed mutations)
    - rank_all_cosmic/mutations.txt (recurrence counts by gene/position)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path


# Configuration
RAW_DATA_DIR = Path("/mnt/storage/ES/raw")
PROJECT_DIR = Path(__file__).parent
COSMIC_API_BASE = "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"

# COSMIC files to download
COSMIC_FILES = [
    {
        "name": "GenomeScreensMutant",
        "path": "grch38/cosmic/v103/Cosmic_GenomeScreensMutant_Tsv_v103_GRCh38.tar",
        "description": "Genome-wide mutations from WGS/WES screens"
    },
    {
        "name": "MutantCensus",
        "path": "grch38/cosmic/v103/Cosmic_MutantCensus_Tsv_v103_GRCh38.tar",
        "description": "Cancer Gene Census mutations only"
    }
]


def get_auth_string() -> str:
    """Get base64 encoded auth string from environment or prompt."""
    auth = os.environ.get("COSMIC_AUTH")
    if not auth:
        email = os.environ.get("COSMIC_EMAIL")
        password = os.environ.get("COSMIC_PASSWORD")
        if email and password:
            auth = base64.b64encode(f"{email}:{password}".encode()).decode()
        else:
            print("COSMIC credentials not found in environment.")
            print("Set COSMIC_AUTH (base64 encoded) or COSMIC_EMAIL and COSMIC_PASSWORD")
            sys.exit(1)
    return auth


def download_cosmic_file(file_info: dict, auth: str, output_dir: Path) -> Path:
    """Download a COSMIC file using the scripted API."""
    import requests

    print(f"Getting download URL for {file_info['name']}...")

    # Get signed URL
    url = f"{COSMIC_API_BASE}?path={file_info['path']}&bucket=downloads"
    headers = {"Authorization": f"Basic {auth}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    download_url = resp.json()["url"]

    # Download file
    output_file = output_dir / Path(file_info["path"]).name
    print(f"Downloading to {output_file}...")

    subprocess.run([
        "curl", "-o", str(output_file), download_url
    ], check=True)

    # Extract tar
    print(f"Extracting {output_file}...")
    subprocess.run([
        "tar", "-xvf", str(output_file), "-C", str(output_dir)
    ], check=True)

    return output_file


def process_cosmic_data(input_file: Path, output_file: Path):
    """Process COSMIC data to the expected format."""
    from process_cosmic_v103 import process_cosmic_v103
    process_cosmic_v103(input_file, output_file)


def generate_mutations_file(cosmic_file: Path, output_file: Path, gene_list: Path = None):
    """Generate mutations.txt from COSMIC data."""
    from generate_mutations_from_cosmic import generate_mutations
    generate_mutations(cosmic_file, output_file, gene_list)


def main():
    parser = argparse.ArgumentParser(description="Update COSMIC database")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading files (use existing)")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip processing files")
    parser.add_argument("--census-only", action="store_true",
                        help="Only use Cancer Gene Census (smaller)")
    args = parser.parse_args()

    # Create output directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        auth = get_auth_string()

        # Download files
        for file_info in COSMIC_FILES:
            if args.census_only and file_info["name"] != "MutantCensus":
                continue
            download_cosmic_file(file_info, auth, RAW_DATA_DIR)

    if not args.skip_process:
        # Process the genome-wide file (or census if --census-only)
        if args.census_only:
            input_gz = RAW_DATA_DIR / "Cosmic_MutantCensus_v103_GRCh38.tsv.gz"
        else:
            input_gz = RAW_DATA_DIR / "Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz"

        output_tsv = PROJECT_DIR / "CosmicMutantExport.missense.aachange.tsv"
        print(f"Processing {input_gz}...")
        process_cosmic_data(input_gz, output_tsv)

        # Generate mutations file
        mutations_file = PROJECT_DIR / "rank_all_cosmic" / "mutations.txt"
        print(f"Generating {mutations_file}...")
        generate_mutations_file(output_tsv, mutations_file)

    print("\nCOSMIC database update complete!")
    print("\nNext steps:")
    print("1. Run plot.py to regenerate ES scores for all genes")
    print("2. Run plot_gs_rank.py to regenerate cosmic.feather")
    print("3. Verify results with benchmark scripts")


if __name__ == "__main__":
    main()
