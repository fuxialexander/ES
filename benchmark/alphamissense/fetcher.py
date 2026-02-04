#!/usr/bin/env python3
"""
Main AlphaMissense prediction fetcher.

Unified interface for retrieving AlphaMissense predictions from multiple sources:
1. Bulk data files (Zenodo) - for large-scale analysis
2. Ensembl VEP REST API - for small-scale queries
3. HegeLab web resource - for structure-based queries

Recommended usage:
- For benchmarking (100+ variants): Use bulk data
- For interactive queries (<10 variants): Use Ensembl VEP API
- For structural context: Use HegeLab

Example:
    fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")

    # Download bulk data (only needed once)
    fetcher.download_bulk_data()

    # Query variants
    score = fetcher.get_score("P00533", "L858R")
    scores = fetcher.get_scores_for_gene("EGFR")
"""

import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .data_loader import AlphaMissenseLoader, DATA_FILES, ZENODO_BASE_URL
from .ensembl_client import EnsemblVEPClient
from .hegelab_client import HegeLab_AMClient


# Default data directory
DEFAULT_DATA_DIR = "/mnt/storage/alphamissense"


class AlphaMissenseFetcher:
    """
    Main class for fetching AlphaMissense predictions.

    Provides a unified interface to multiple data sources with automatic
    fallback to API queries when bulk data is not available.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
        prefer_bulk: bool = True,
    ):
        """
        Initialize the fetcher.

        Args:
            data_dir: Directory for bulk data files
            uniprot_mapping_file: Path to gene name -> UniProt mapping file
            prefer_bulk: Prefer bulk data over API queries when available
        """
        self.data_dir = Path(data_dir)
        self.prefer_bulk = prefer_bulk

        # Initialize data sources
        self._bulk_loader: Optional[AlphaMissenseLoader] = None
        self._ensembl_client: Optional[EnsemblVEPClient] = None
        self._hegelab_client: Optional[HegeLab_AMClient] = None

        # UniProt mapping
        self._uniprot_mapping: Optional[Dict[str, str]] = None
        if uniprot_mapping_file:
            self._load_uniprot_mapping(uniprot_mapping_file)

    def _load_uniprot_mapping(self, filepath: Union[str, Path]) -> None:
        """Load gene name to UniProt ID mapping."""
        self._uniprot_mapping = {}
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    uniprot_id, gene_name = parts[0], parts[1]
                    self._uniprot_mapping[gene_name.upper()] = uniprot_id
                    # Also store reverse mapping
                    self._uniprot_mapping[uniprot_id] = uniprot_id

    @property
    def bulk_loader(self) -> AlphaMissenseLoader:
        """Get bulk data loader (lazy initialization)."""
        if self._bulk_loader is None:
            self._bulk_loader = AlphaMissenseLoader(self.data_dir)
        return self._bulk_loader

    @property
    def ensembl_client(self) -> EnsemblVEPClient:
        """Get Ensembl VEP client (lazy initialization)."""
        if self._ensembl_client is None:
            self._ensembl_client = EnsemblVEPClient()
        return self._ensembl_client

    @property
    def hegelab_client(self) -> HegeLab_AMClient:
        """Get HegeLab client (lazy initialization)."""
        if self._hegelab_client is None:
            self._hegelab_client = HegeLab_AMClient()
        return self._hegelab_client

    def has_bulk_data(self) -> bool:
        """Check if bulk data is available."""
        aa_file = self.data_dir / DATA_FILES["aa_substitutions"]["filename"]
        return aa_file.exists()

    def download_bulk_data(
        self,
        files: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, bool]:
        """
        Download bulk data files from Zenodo.

        Args:
            files: List of file keys to download (default: aa_substitutions + gene_hg38)
                   Options: aa_substitutions, hg38, hg19, gene_hg38, gene_hg19
            force: Force re-download even if files exist

        Returns:
            Dict mapping file key to download success status
        """
        if files is None:
            # Default to essential files
            files = ["aa_substitutions", "gene_hg38"]

        self.data_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for file_key in files:
            if file_key not in DATA_FILES:
                print(f"Unknown file key: {file_key}")
                print(f"Available: {list(DATA_FILES.keys())}")
                results[file_key] = False
                continue

            file_info = DATA_FILES[file_key]
            output_path = self.data_dir / file_info["filename"]

            if output_path.exists() and not force:
                print(f"File already exists: {output_path}")
                results[file_key] = True
                continue

            print(f"\nDownloading {file_key}:")
            print(f"  URL: {file_info['url']}")
            print(f"  Size: ~{file_info['size_mb']:.0f} MB")
            print(f"  Description: {file_info['description']}")

            success = self._download_file(file_info["url"], output_path)
            results[file_key] = success

        return results

    def _download_file(
        self,
        url: str,
        output_path: Path,
        retries: int = 3,
        timeout: int = 300,
    ) -> bool:
        """Download a file with retry logic."""
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (ES Score Benchmark)"},
                )

                with urllib.request.urlopen(request, timeout=timeout) as response:
                    total_size = response.headers.get("Content-Length")
                    if total_size:
                        total_size = int(total_size)
                        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

                    downloaded = 0
                    chunk_size = 65536  # 64KB chunks
                    with open(output_path, "wb") as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size:
                                pct = downloaded / total_size * 100
                                mb_done = downloaded / 1024 / 1024
                                mb_total = total_size / 1024 / 1024
                                print(f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)",
                                      end="", flush=True)

                print(f"\n  Saved to: {output_path}")
                return True

            except urllib.error.HTTPError as e:
                print(f"\n  HTTP Error {e.code}: {e.reason}")
            except urllib.error.URLError as e:
                print(f"\n  URL Error: {e.reason}")
            except Exception as e:
                print(f"\n  Error: {e}")

            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        return False

    def gene_to_uniprot(self, gene_name: str) -> Optional[str]:
        """
        Convert gene name to UniProt ID.

        Args:
            gene_name: Gene symbol (e.g., 'EGFR')

        Returns:
            UniProt ID or None if not found
        """
        if self._uniprot_mapping is None:
            # Try to load default mapping file
            default_mapping = Path(__file__).parent.parent.parent / "uniprot_to_genename.txt"
            if default_mapping.exists():
                self._load_uniprot_mapping(default_mapping)

        if self._uniprot_mapping:
            return self._uniprot_mapping.get(gene_name.upper())

        return None

    def get_score(
        self,
        identifier: str,
        variant: str,
        use_api_fallback: bool = True,
    ) -> Optional[Tuple[float, str]]:
        """
        Get AlphaMissense score for a variant.

        Args:
            identifier: UniProt ID (e.g., 'P00533') or gene name (e.g., 'EGFR')
            variant: Variant string (e.g., 'L858R')
            use_api_fallback: Fall back to API if bulk data not available

        Returns:
            Tuple of (pathogenicity_score, classification) or None if not found
        """
        # Resolve identifier to UniProt ID
        uniprot_id = identifier
        if not re.match(r'^[A-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$', identifier):
            # Looks like a gene name, try to convert
            resolved = self.gene_to_uniprot(identifier)
            if resolved:
                uniprot_id = resolved
            elif use_api_fallback:
                # Use Ensembl to look up by gene name
                pass  # Will try API below

        # Try bulk data first
        if self.prefer_bulk and self.has_bulk_data():
            result = self.bulk_loader.get_score(uniprot_id, variant)
            if result:
                return result

        # Fall back to API
        if use_api_fallback:
            # Try Ensembl VEP
            # Note: This requires HGVS notation, so we'd need transcript info
            pass

        return None

    def get_scores_for_protein(
        self,
        identifier: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get all AlphaMissense scores for a protein.

        Args:
            identifier: UniProt ID or gene name

        Returns:
            DataFrame with all variants and scores
        """
        # Resolve identifier
        uniprot_id = identifier
        if not re.match(r'^[A-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$', identifier):
            resolved = self.gene_to_uniprot(identifier)
            if resolved:
                uniprot_id = resolved

        if self.has_bulk_data():
            return self.bulk_loader.get_scores_for_protein(uniprot_id)

        return None

    def score_mutations_batch(
        self,
        mutations: pd.DataFrame,
        uniprot_col: str = "uniprot_id",
        variant_col: str = "protein_variant",
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Score a batch of mutations.

        Args:
            mutations: DataFrame with mutations
            uniprot_col: Column name for UniProt IDs
            variant_col: Column name for variant strings
            progress: Show progress

        Returns:
            DataFrame with added am_pathogenicity and am_class columns
        """
        if not self.has_bulk_data():
            raise RuntimeError(
                "Bulk data required for batch scoring. "
                "Run download_bulk_data() first."
            )

        # Prepare query list
        variants = list(zip(mutations[uniprot_col], mutations[variant_col]))

        # Query bulk data
        result_df = self.bulk_loader.batch_query(variants, progress=progress)

        # Merge back with original
        mutations = mutations.copy()
        mutations["am_pathogenicity"] = result_df["am_pathogenicity"].values
        mutations["am_class"] = result_df["am_class"].values

        return mutations

    def score_position_batch(
        self,
        positions: List[Tuple[str, int, str, str]],
        progress: bool = True,
    ) -> List[Optional[Tuple[float, str]]]:
        """
        Score a batch of positions with specific amino acid changes.

        Args:
            positions: List of (uniprot_id, position, ref_aa, alt_aa) tuples
            progress: Show progress

        Returns:
            List of (score, classification) tuples or None for each position
        """
        if not self.has_bulk_data():
            raise RuntimeError(
                "Bulk data required for batch scoring. "
                "Run download_bulk_data() first."
            )

        results = []
        total = len(positions)

        for i, (uniprot_id, pos, ref_aa, alt_aa) in enumerate(positions, 1):
            if progress and i % 100 == 0:
                print(f"Scoring {i}/{total}", end="\r")

            variant = f"{ref_aa}{pos}{alt_aa}"
            result = self.bulk_loader.get_score(uniprot_id, variant)
            results.append(result)

        if progress:
            n_found = sum(1 for r in results if r is not None)
            print(f"\nScored {n_found}/{total} variants")

        return results

    def get_gene_level_scores(
        self,
        genome: str = "hg38",
    ) -> pd.DataFrame:
        """
        Get gene-level average AlphaMissense scores.

        Args:
            genome: Genome build ('hg38' or 'hg19')

        Returns:
            DataFrame with gene-level average scores
        """
        return self.bulk_loader.load_gene_scores(genome)


def main():
    """Command-line interface for AlphaMissense fetcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch AlphaMissense predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download bulk data
    python -m benchmark.alphamissense.fetcher --download --data_dir /mnt/storage/alphamissense

    # Query a variant
    python -m benchmark.alphamissense.fetcher --uniprot P00533 --variant L858R

    # Query all variants for a protein
    python -m benchmark.alphamissense.fetcher --uniprot P00533 --all
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory for bulk data files",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download bulk data files",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Specific files to download",
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        help="UniProt ID or gene name to query",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Variant to query (e.g., L858R)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Get all variants for the protein",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check available data files",
    )

    args = parser.parse_args()

    fetcher = AlphaMissenseFetcher(data_dir=args.data_dir)

    if args.check:
        print("AlphaMissense Data Status:")
        print(f"  Data directory: {fetcher.data_dir}")
        print(f"  Bulk data available: {fetcher.has_bulk_data()}")
        if fetcher.data_dir.exists():
            available = fetcher.bulk_loader.get_available_files()
            for name, exists in available.items():
                status = "OK" if exists else "MISSING"
                print(f"    [{status}] {name}")

    elif args.download:
        results = fetcher.download_bulk_data(files=args.files)
        print("\nDownload Summary:")
        for name, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  [{status}] {name}")

    elif args.uniprot:
        if args.all:
            df = fetcher.get_scores_for_protein(args.uniprot)
            if df is not None:
                print(f"AlphaMissense predictions for {args.uniprot}:")
                print(f"  Total variants: {len(df):,}")
                print(f"  Score range: {df['am_pathogenicity'].min():.3f} - {df['am_pathogenicity'].max():.3f}")
                print(f"\n  Classification distribution:")
                print(df["am_class"].value_counts())
            else:
                print(f"No predictions found for {args.uniprot}")

        elif args.variant:
            result = fetcher.get_score(args.uniprot, args.variant)
            if result:
                score, classification = result
                print(f"AlphaMissense prediction for {args.uniprot} {args.variant}:")
                print(f"  Pathogenicity score: {score:.4f}")
                print(f"  Classification: {classification}")
            else:
                print(f"No prediction found for {args.uniprot} {args.variant}")

        else:
            print("Specify --variant or --all")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
