#!/usr/bin/env python3
"""
AlphaMissense bulk data loader.

Loads and queries pre-downloaded AlphaMissense prediction files from Zenodo.
Supports efficient querying of 216M+ amino acid substitutions.

Data files (from https://zenodo.org/records/10813168):
- AlphaMissense_aa_substitutions.tsv.gz: 216M protein variants by UniProt ID (1.2 GB)
- AlphaMissense_hg38.tsv.gz: 71M SNV missense variants with hg38 coordinates (643 MB)
- AlphaMissense_hg19.tsv.gz: 71M SNV missense variants with hg19 coordinates (622 MB)
- AlphaMissense_gene_hg38.tsv.gz: Gene-level average scores (254 KB)
"""

import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# File specifications
ZENODO_DOI = "10.5281/zenodo.10813168"
ZENODO_BASE_URL = "https://zenodo.org/records/10813168/files"

DATA_FILES = {
    "aa_substitutions": {
        "filename": "AlphaMissense_aa_substitutions.tsv.gz",
        "url": f"{ZENODO_BASE_URL}/AlphaMissense_aa_substitutions.tsv.gz",
        "size_mb": 1200,
        "description": "All possible amino acid substitutions (216M variants) by UniProt ID",
        "columns": ["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"],
    },
    "hg38": {
        "filename": "AlphaMissense_hg38.tsv.gz",
        "url": f"{ZENODO_BASE_URL}/AlphaMissense_hg38.tsv.gz",
        "size_mb": 643,
        "description": "SNV missense variants with hg38 genomic coordinates (71M variants)",
        "columns": ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
                    "transcript_id", "protein_variant", "am_pathogenicity", "am_class"],
    },
    "hg19": {
        "filename": "AlphaMissense_hg19.tsv.gz",
        "url": f"{ZENODO_BASE_URL}/AlphaMissense_hg19.tsv.gz",
        "size_mb": 622,
        "description": "SNV missense variants with hg19 genomic coordinates (71M variants)",
        "columns": ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
                    "transcript_id", "protein_variant", "am_pathogenicity", "am_class"],
    },
    "gene_hg38": {
        "filename": "AlphaMissense_gene_hg38.tsv.gz",
        "url": f"{ZENODO_BASE_URL}/AlphaMissense_gene_hg38.tsv.gz",
        "size_mb": 0.25,
        "description": "Gene-level average pathogenicity scores (hg38)",
        "columns": ["transcript_id", "uniprot_id", "mean_am_pathogenicity"],
    },
    "gene_hg19": {
        "filename": "AlphaMissense_gene_hg19.tsv.gz",
        "url": f"{ZENODO_BASE_URL}/AlphaMissense_gene_hg19.tsv.gz",
        "size_mb": 0.24,
        "description": "Gene-level average pathogenicity scores (hg19)",
        "columns": ["transcript_id", "uniprot_id", "mean_am_pathogenicity"],
    },
}

# Classification thresholds (from AlphaMissense paper)
PATHOGENICITY_THRESHOLDS = {
    "likely_benign": 0.34,      # am_pathogenicity < 0.34
    "likely_pathogenic": 0.564,  # am_pathogenicity > 0.564
    # ambiguous: 0.34 <= am_pathogenicity <= 0.564
}


class AlphaMissenseLoader:
    """
    Load and query AlphaMissense prediction data.

    For large-scale analysis, loads bulk data files from Zenodo.
    Supports efficient memory-mapped reading of compressed files.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing AlphaMissense data files
        """
        self.data_dir = Path(data_dir)
        self._aa_data: Optional[pd.DataFrame] = None
        self._hg38_data: Optional[pd.DataFrame] = None
        self._gene_data: Optional[pd.DataFrame] = None
        self._uniprot_index: Optional[Dict[str, pd.DataFrame]] = None

    def get_available_files(self) -> Dict[str, bool]:
        """Check which data files are available."""
        return {
            name: (self.data_dir / info["filename"]).exists()
            for name, info in DATA_FILES.items()
        }

    def load_aa_substitutions(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load amino acid substitutions data.

        This is the primary file for querying by UniProt ID and mutation.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            DataFrame with columns: uniprot_id, protein_variant, am_pathogenicity, am_class
        """
        if self._aa_data is not None and not force_reload:
            return self._aa_data

        filepath = self.data_dir / DATA_FILES["aa_substitutions"]["filename"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"AlphaMissense data not found at {filepath}. "
                f"Run AlphaMissenseFetcher.download_bulk_data() first."
            )

        print(f"Loading AlphaMissense amino acid substitutions from {filepath}...")
        print("  This may take a few minutes for 216M variants...")

        self._aa_data = pd.read_csv(
            filepath,
            sep="\t",
            compression="gzip",
            comment="#",  # Skip comment lines starting with #
            dtype={
                "uniprot_id": "category",
                "protein_variant": str,
                "am_pathogenicity": np.float32,
                "am_class": "category",
            },
        )

        print(f"  Loaded {len(self._aa_data):,} variants for "
              f"{self._aa_data['uniprot_id'].nunique():,} proteins")

        return self._aa_data

    def load_hg38_variants(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load hg38 genomic variant data.

        Use this for queries by genomic coordinates.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            DataFrame with genomic coordinates and pathogenicity scores
        """
        if self._hg38_data is not None and not force_reload:
            return self._hg38_data

        filepath = self.data_dir / DATA_FILES["hg38"]["filename"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"AlphaMissense hg38 data not found at {filepath}. "
                f"Run AlphaMissenseFetcher.download_bulk_data() first."
            )

        print(f"Loading AlphaMissense hg38 variants from {filepath}...")

        self._hg38_data = pd.read_csv(
            filepath,
            sep="\t",
            compression="gzip",
            comment="#",  # Skip comment lines starting with #
            dtype={
                "CHROM": "category",
                "POS": np.int32,
                "REF": "category",
                "ALT": "category",
                "uniprot_id": "category",
                "transcript_id": "category",
                "protein_variant": str,
                "am_pathogenicity": np.float32,
                "am_class": "category",
            },
        )

        print(f"  Loaded {len(self._hg38_data):,} variants")

        return self._hg38_data

    def load_gene_scores(self, genome: str = "hg38", force_reload: bool = False) -> pd.DataFrame:
        """
        Load gene-level average pathogenicity scores.

        Args:
            genome: Genome build ('hg38' or 'hg19')
            force_reload: Force reload even if already loaded

        Returns:
            DataFrame with gene-level average scores
        """
        if self._gene_data is not None and not force_reload:
            return self._gene_data

        key = f"gene_{genome}"
        filepath = self.data_dir / DATA_FILES[key]["filename"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"AlphaMissense gene data not found at {filepath}. "
                f"Run AlphaMissenseFetcher.download_bulk_data() first."
            )

        self._gene_data = pd.read_csv(
            filepath,
            sep="\t",
            compression="gzip",
            comment="#",  # Skip comment lines starting with #
        )

        return self._gene_data

    def build_uniprot_index(self) -> None:
        """
        Build an index for fast UniProt-based lookups.

        Groups variants by UniProt ID for efficient per-protein queries.
        """
        if self._aa_data is None:
            self.load_aa_substitutions()

        print("Building UniProt index...")
        self._uniprot_index = {
            uniprot_id: group
            for uniprot_id, group in self._aa_data.groupby("uniprot_id", observed=True)
        }
        print(f"  Indexed {len(self._uniprot_index):,} proteins")

    def get_score(
        self,
        uniprot_id: str,
        protein_variant: str,
    ) -> Optional[Tuple[float, str]]:
        """
        Get AlphaMissense score for a specific variant.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')
            protein_variant: Variant string (e.g., 'L858R')

        Returns:
            Tuple of (pathogenicity_score, classification) or None if not found
        """
        if self._aa_data is None:
            self.load_aa_substitutions()

        # Query directly
        mask = (self._aa_data["uniprot_id"] == uniprot_id) & \
               (self._aa_data["protein_variant"] == protein_variant)

        matches = self._aa_data[mask]
        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return (row["am_pathogenicity"], row["am_class"])

    def get_scores_for_protein(self, uniprot_id: str) -> Optional[pd.DataFrame]:
        """
        Get all AlphaMissense scores for a protein.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')

        Returns:
            DataFrame with all variants for this protein, or None if not found
        """
        if self._uniprot_index is not None:
            return self._uniprot_index.get(uniprot_id)

        if self._aa_data is None:
            self.load_aa_substitutions()

        matches = self._aa_data[self._aa_data["uniprot_id"] == uniprot_id]
        if len(matches) == 0:
            return None

        return matches.copy()

    def get_score_by_position(
        self,
        uniprot_id: str,
        position: int,
        alt_aa: str,
    ) -> Optional[Tuple[float, str]]:
        """
        Get AlphaMissense score by protein position and alternate amino acid.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')
            position: 1-based amino acid position
            alt_aa: Alternate amino acid (single letter)

        Returns:
            Tuple of (pathogenicity_score, classification) or None if not found
        """
        protein_df = self.get_scores_for_protein(uniprot_id)
        if protein_df is None:
            return None

        # Match variant string pattern (e.g., "L858R" where 858 is position and R is alt)
        matches = protein_df[
            protein_df["protein_variant"].str.contains(f"^[A-Z]{position}{alt_aa}$", regex=True)
        ]

        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return (row["am_pathogenicity"], row["am_class"])

    def get_score_by_genomic_position(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
    ) -> Optional[Tuple[float, str]]:
        """
        Get AlphaMissense score by hg38 genomic coordinates.

        Args:
            chrom: Chromosome (e.g., 'chr7')
            pos: 1-based genomic position
            ref: Reference nucleotide
            alt: Alternate nucleotide

        Returns:
            Tuple of (pathogenicity_score, classification) or None if not found
        """
        if self._hg38_data is None:
            self.load_hg38_variants()

        # Ensure chromosome format matches
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        mask = (
            (self._hg38_data["CHROM"] == chrom) &
            (self._hg38_data["POS"] == pos) &
            (self._hg38_data["REF"] == ref) &
            (self._hg38_data["ALT"] == alt)
        )

        matches = self._hg38_data[mask]
        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return (row["am_pathogenicity"], row["am_class"])

    def classify_score(self, score: float) -> str:
        """
        Classify a pathogenicity score.

        Args:
            score: AlphaMissense pathogenicity score (0-1)

        Returns:
            Classification: 'likely_benign', 'ambiguous', or 'likely_pathogenic'
        """
        if score < PATHOGENICITY_THRESHOLDS["likely_benign"]:
            return "likely_benign"
        elif score > PATHOGENICITY_THRESHOLDS["likely_pathogenic"]:
            return "likely_pathogenic"
        else:
            return "ambiguous"

    def batch_query(
        self,
        variants: List[Tuple[str, str]],
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Query multiple variants efficiently.

        Args:
            variants: List of (uniprot_id, protein_variant) tuples
            progress: Show progress bar

        Returns:
            DataFrame with query results
        """
        if self._aa_data is None:
            self.load_aa_substitutions()

        # Create query DataFrame
        query_df = pd.DataFrame(variants, columns=["uniprot_id", "protein_variant"])

        # Merge with data
        result = query_df.merge(
            self._aa_data[["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"]],
            on=["uniprot_id", "protein_variant"],
            how="left",
        )

        n_found = result["am_pathogenicity"].notna().sum()
        print(f"Found scores for {n_found:,} / {len(variants):,} variants")

        return result


def stream_query_variants(
    filepath: Union[str, Path],
    query_variants: List[Tuple[str, str]],
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """
    Query variants from a large file using streaming (memory-efficient).

    Args:
        filepath: Path to compressed TSV file
        query_variants: List of (uniprot_id, protein_variant) tuples
        chunk_size: Number of rows to read per chunk

    Returns:
        DataFrame with matching variants
    """
    query_set = set(query_variants)
    results = []

    with gzip.open(filepath, "rt") as f:
        reader = pd.read_csv(
            f,
            sep="\t",
            chunksize=chunk_size,
        )

        for chunk in reader:
            # Filter to matching variants
            mask = chunk.apply(
                lambda row: (row["uniprot_id"], row["protein_variant"]) in query_set,
                axis=1,
            )
            matches = chunk[mask]
            if len(matches) > 0:
                results.append(matches)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query AlphaMissense predictions")
    parser.add_argument("data_dir", type=str, help="Directory containing AlphaMissense data")
    parser.add_argument("--uniprot", type=str, help="UniProt ID to query")
    parser.add_argument("--variant", type=str, help="Variant to query (e.g., L858R)")
    parser.add_argument("--check", action="store_true", help="Check available files")

    args = parser.parse_args()

    loader = AlphaMissenseLoader(args.data_dir)

    if args.check:
        print("Available AlphaMissense data files:")
        for name, available in loader.get_available_files().items():
            status = "OK" if available else "MISSING"
            print(f"  [{status}] {name}: {DATA_FILES[name]['description']}")
    elif args.uniprot and args.variant:
        result = loader.get_score(args.uniprot, args.variant)
        if result:
            score, classification = result
            print(f"AlphaMissense prediction for {args.uniprot} {args.variant}:")
            print(f"  Pathogenicity score: {score:.4f}")
            print(f"  Classification: {classification}")
        else:
            print(f"No prediction found for {args.uniprot} {args.variant}")
    elif args.uniprot:
        df = loader.get_scores_for_protein(args.uniprot)
        if df is not None:
            print(f"AlphaMissense predictions for {args.uniprot}:")
            print(f"  Total variants: {len(df):,}")
            print(f"  Score range: {df['am_pathogenicity'].min():.3f} - {df['am_pathogenicity'].max():.3f}")
            print(f"\n  Class distribution:")
            print(df["am_class"].value_counts())
        else:
            print(f"No predictions found for {args.uniprot}")
