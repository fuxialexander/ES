#!/usr/bin/env python3
"""
ClinVar Data Downloader

Downloads and preprocesses ClinVar data for pathogenic variant benchmark.

Data sources:
1. ClinVar VCF from NCBI FTP (weekly updates)
2. ClinVar variant_summary.txt for detailed annotations
3. Optional: ProteinGym clinical_substitutions for pre-processed data

Usage:
    python download_data.py --output_dir ./data
    python download_data.py --source vcf --output_dir ./data
    python download_data.py --source proteingym --output_dir ./data
"""

import os
import sys
import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ClinVar FTP URLs
CLINVAR_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar"

DATASETS = {
    "variant_summary": {
        "url": f"{CLINVAR_BASE}/tab_delimited/variant_summary.txt.gz",
        "filename": "variant_summary.txt.gz",
        "description": "ClinVar variant summary with all annotations"
    },
    "vcf_grch38": {
        "url": f"{CLINVAR_BASE}/vcf_GRCh38/clinvar.vcf.gz",
        "filename": "clinvar_grch38.vcf.gz",
        "description": "ClinVar VCF for GRCh38"
    },
    "vcf_grch37": {
        "url": f"{CLINVAR_BASE}/vcf_GRCh37/clinvar.vcf.gz",
        "filename": "clinvar_grch37.vcf.gz",
        "description": "ClinVar VCF for GRCh37"
    }
}

# ProteinGym clinical data URL
PROTEINGYM_CLINICAL_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
    "clinical_substitutions_labels.csv"
)


def download_progress_hook(count, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve"""
    if total_size > 0:
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


def download_clinvar_summary(output_dir: Path) -> bool:
    """Download ClinVar variant summary file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASETS["variant_summary"]
    gz_path = output_dir / dataset["filename"]

    success = download_file(
        dataset["url"],
        gz_path,
        dataset["description"]
    )

    if success:
        # Decompress
        txt_path = output_dir / "variant_summary.txt"
        print(f"Decompressing to: {txt_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(txt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Decompression complete")

    return success


def download_proteingym_clinical(output_dir: Path) -> bool:
    """Download ProteinGym clinical substitutions labels"""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "proteingym_clinical_substitutions.csv"

    return download_file(
        PROTEINGYM_CLINICAL_URL,
        output_path,
        "ProteinGym clinical substitutions (ClinVar pathogenic/benign)"
    )


def preprocess_variant_summary(
    input_path: Path,
    output_path: Path,
    min_review_stars: int = 1
) -> pd.DataFrame:
    """
    Preprocess ClinVar variant summary to extract pathogenic/benign missense variants.

    Args:
        input_path: Path to variant_summary.txt
        output_path: Path to save processed CSV
        min_review_stars: Minimum review status (1-4 stars)

    Returns:
        Processed DataFrame with pathogenic/benign missense variants
    """
    print(f"\nPreprocessing ClinVar variant summary...")
    print(f"Minimum review status: {min_review_stars} star(s)")

    # Read variant summary
    # Columns of interest:
    # - GeneSymbol
    # - ClinicalSignificance
    # - ReviewStatus
    # - Type (missense, etc.)
    # - ProteinChange
    # - PositionVCF

    # Read in chunks due to large file size
    chunks = []
    cols_to_use = [
        '#AlleleID', 'Type', 'Name', 'GeneID', 'GeneSymbol',
        'ClinicalSignificance', 'ClinSigSimple', 'ReviewStatus',
        'VariationID', 'PositionVCF', 'ReferenceAlleleVCF',
        'AlternateAlleleVCF', 'Assembly', 'Origin'
    ]

    chunksize = 100000
    for chunk in pd.read_csv(
        input_path,
        sep='\t',
        low_memory=False,
        chunksize=chunksize
    ):
        # Filter to GRCh38 assembly
        if 'Assembly' in chunk.columns:
            chunk = chunk[chunk['Assembly'] == 'GRCh38']

        # Filter to missense variants
        if 'Type' in chunk.columns:
            chunk = chunk[chunk['Type'].str.contains('Missense|missense', na=False)]

        # Filter by clinical significance
        if 'ClinicalSignificance' in chunk.columns:
            pathogenic_terms = [
                'Pathogenic', 'Likely pathogenic',
                'Pathogenic/Likely pathogenic'
            ]
            benign_terms = [
                'Benign', 'Likely benign',
                'Benign/Likely benign'
            ]
            all_terms = pathogenic_terms + benign_terms

            mask = chunk['ClinicalSignificance'].apply(
                lambda x: any(term in str(x) for term in all_terms)
            )
            chunk = chunk[mask]

        # Filter by review status
        if 'ReviewStatus' in chunk.columns and min_review_stars > 0:
            # Map review status to stars
            star_mapping = {
                'no assertion criteria provided': 0,
                'no assertion provided': 0,
                'criteria provided, single submitter': 1,
                'criteria provided, conflicting interpretations': 1,
                'criteria provided, multiple submitters, no conflicts': 2,
                'reviewed by expert panel': 3,
                'practice guideline': 4
            }
            chunk['Stars'] = chunk['ReviewStatus'].map(star_mapping).fillna(0)
            chunk = chunk[chunk['Stars'] >= min_review_stars]

        chunks.append(chunk)

    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)

    # Add binary label
    def get_label(clin_sig):
        clin_sig = str(clin_sig)
        if 'athogenic' in clin_sig and 'Benign' not in clin_sig:
            return 1  # Pathogenic
        elif 'enign' in clin_sig and 'Pathogenic' not in clin_sig:
            return 0  # Benign
        else:
            return -1  # Conflicting or uncertain

    df['Label'] = df['ClinicalSignificance'].apply(get_label)
    df = df[df['Label'] >= 0]  # Remove conflicting

    # Parse protein change to get position
    def parse_protein_change(name):
        """Extract amino acid position from Name field"""
        import re
        # Pattern: (p.X123Y)
        match = re.search(r'\(p\.([A-Za-z]+)(\d+)([A-Za-z\*]+)\)', str(name))
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        return None, None, None

    df['wt_aa'], df['position'], df['mt_aa'] = zip(
        *df['Name'].apply(parse_protein_change)
    )
    df = df[df['position'].notna()]
    df['position'] = df['position'].astype(int)

    # Create mutation string
    df['Mutation'] = df['wt_aa'] + df['position'].astype(str) + df['mt_aa']

    # Save processed data
    output_cols = [
        'GeneSymbol', 'Mutation', 'position', 'wt_aa', 'mt_aa',
        'Label', 'ClinicalSignificance', 'ReviewStatus',
        'VariationID', '#AlleleID'
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    df_out = df[output_cols].copy()
    df_out = df_out.drop_duplicates(subset=['GeneSymbol', 'Mutation'])

    df_out.to_csv(output_path, index=False)

    print(f"\nProcessed {len(df_out):,} variants:")
    print(f"  Pathogenic: {(df_out['Label'] == 1).sum():,}")
    print(f"  Benign: {(df_out['Label'] == 0).sum():,}")
    print(f"  Unique genes: {df_out['GeneSymbol'].nunique():,}")
    print(f"Saved to: {output_path}")

    return df_out


def verify_download(output_dir: Path) -> Dict[str, bool]:
    """Verify downloaded data"""
    results = {}

    # Check for processed ClinVar
    clinvar_path = output_dir / "clinvar_pathogenic_benign.csv"
    results['clinvar_processed'] = clinvar_path.exists()

    # Check for raw files
    summary_path = output_dir / "variant_summary.txt"
    results['variant_summary'] = summary_path.exists()

    # Check for ProteinGym clinical
    pg_path = output_dir / "proteingym_clinical_substitutions.csv"
    results['proteingym_clinical'] = pg_path.exists()

    return results


def print_verification(results: Dict[str, bool]):
    """Print verification results"""
    print("\n" + "="*60)
    print("Data Verification")
    print("="*60)

    for name, exists in results.items():
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")


def download_all(
    output_dir: Path,
    source: str = "summary",
    min_review_stars: int = 1
) -> bool:
    """
    Download and preprocess all ClinVar data.

    Args:
        output_dir: Output directory
        source: 'summary' (NCBI), 'proteingym' (pre-processed), or 'both'
        min_review_stars: Minimum review status for filtering

    Returns:
        Success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = True

    if source in ['summary', 'both']:
        # Download variant summary
        if download_clinvar_summary(output_dir):
            # Preprocess
            summary_path = output_dir / "variant_summary.txt"
            processed_path = output_dir / "clinvar_pathogenic_benign.csv"
            preprocess_variant_summary(summary_path, processed_path, min_review_stars)
        else:
            success = False

    if source in ['proteingym', 'both']:
        if not download_proteingym_clinical(output_dir):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download ClinVar data for pathogenic variant benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and preprocess from NCBI ClinVar
    python download_data.py --output_dir ./data --source summary

    # Download ProteinGym pre-processed clinical data
    python download_data.py --output_dir ./data --source proteingym

    # Download both sources
    python download_data.py --output_dir ./data --source both

    # Filter by minimum review status (1-4 stars)
    python download_data.py --output_dir ./data --min_stars 2
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=['summary', 'proteingym', 'both'],
        default='summary',
        help="Data source: 'summary' (NCBI ClinVar), 'proteingym', or 'both'"
    )
    parser.add_argument(
        "--min_stars",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Minimum review status (0-4 stars)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing data"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.verify:
        results = verify_download(output_dir)
        print_verification(results)
        return

    # Download
    success = download_all(
        output_dir,
        source=args.source,
        min_review_stars=args.min_stars
    )

    # Verify
    results = verify_download(output_dir)
    print_verification(results)

    if success:
        print("\nDownload complete!")
    else:
        print("\nSome downloads failed. Please check errors above.")


if __name__ == "__main__":
    main()
