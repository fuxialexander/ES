#!/usr/bin/env python3
"""
ProteinGym Data Loader

Handles loading and parsing of ProteinGym benchmark datasets.
Provides utilities for accessing DMS assay data and reference files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DMSAssay:
    """Represents a single DMS assay dataset"""
    assay_id: str
    uniprot_id: str
    gene_name: str
    target_seq: str
    data: pd.DataFrame
    msa_depth: Optional[str] = None
    taxon: Optional[str] = None
    selection_type: Optional[str] = None


class ProteinGymLoader:
    """Load and manage ProteinGym benchmark datasets"""

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize ProteinGym loader.

        Args:
            data_dir: Path to directory containing downloaded ProteinGym data
        """
        self.data_dir = Path(data_dir)
        self._reference_df = None
        self._assay_cache = {}

    @property
    def substitutions_dir(self) -> Path:
        """Path to DMS substitutions directory"""
        return self.data_dir / "DMS_ProteinGym_substitutions"

    @property
    def indels_dir(self) -> Path:
        """Path to DMS indels directory"""
        return self.data_dir / "DMS_ProteinGym_indels"

    @property
    def reference_dir(self) -> Path:
        """Path to reference files directory"""
        return self.data_dir / "reference_files"

    def load_reference_file(self) -> pd.DataFrame:
        """
        Load the DMS reference file containing assay metadata.

        Returns:
            DataFrame with columns: DMS_id, UniProt_ID, target_seq, MSA_filename, etc.
        """
        if self._reference_df is not None:
            return self._reference_df

        # Try different possible locations for reference file
        possible_paths = [
            self.reference_dir / "DMS_substitutions.csv",
            self.data_dir / "DMS_substitutions.csv",
            self.reference_dir / "reference_files" / "DMS_substitutions.csv",
        ]

        for path in possible_paths:
            if path.exists():
                self._reference_df = pd.read_csv(path)
                return self._reference_df

        # If no reference file, create from available CSVs
        print("Reference file not found, building from available data...")
        return self._build_reference_from_data()

    def _build_reference_from_data(self) -> pd.DataFrame:
        """Build reference dataframe from available DMS CSV files"""
        if not self.substitutions_dir.exists():
            raise FileNotFoundError(
                f"Substitutions directory not found: {self.substitutions_dir}"
            )

        records = []
        for csv_file in self.substitutions_dir.glob("*.csv"):
            assay_id = csv_file.stem

            # Parse assay ID to extract metadata
            # Format: {GENE}_{AUTHOR}_{YEAR}_{...}
            parts = assay_id.split("_")
            gene_name = parts[0] if parts else ""

            # Load first row to get target sequence
            df = pd.read_csv(csv_file, nrows=1)
            target_seq = df.get("target_seq", [""])[0] if "target_seq" in df.columns else ""

            records.append({
                "DMS_id": assay_id,
                "gene_name": gene_name,
                "target_seq": target_seq,
                "csv_file": csv_file.name
            })

        self._reference_df = pd.DataFrame(records)
        return self._reference_df

    def list_assays(self, gene_filter: Optional[str] = None) -> List[str]:
        """
        List available DMS assay IDs.

        Args:
            gene_filter: Optional gene name to filter assays

        Returns:
            List of assay IDs
        """
        if self.substitutions_dir.exists():
            assays = [f.stem for f in self.substitutions_dir.glob("*.csv")]
        else:
            ref_df = self.load_reference_file()
            assays = ref_df["DMS_id"].tolist()

        if gene_filter:
            assays = [a for a in assays if a.upper().startswith(gene_filter.upper())]

        return sorted(assays)

    def load_assay(self, assay_id: str) -> DMSAssay:
        """
        Load a single DMS assay dataset.

        Args:
            assay_id: The DMS assay identifier

        Returns:
            DMSAssay object containing the data
        """
        if assay_id in self._assay_cache:
            return self._assay_cache[assay_id]

        # Find the CSV file
        csv_path = self.substitutions_dir / f"{assay_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Assay file not found: {csv_path}")

        # Load data
        df = pd.read_csv(csv_path)

        # Extract metadata
        parts = assay_id.split("_")
        gene_name = parts[0] if parts else ""

        # Get target sequence
        target_seq = df["target_seq"].iloc[0] if "target_seq" in df.columns else ""

        # Try to get UniProt ID from reference
        uniprot_id = ""
        ref_df = self.load_reference_file()
        if "UniProt_ID" in ref_df.columns:
            match = ref_df[ref_df["DMS_id"] == assay_id]
            if len(match) > 0:
                uniprot_id = match["UniProt_ID"].iloc[0]

        assay = DMSAssay(
            assay_id=assay_id,
            uniprot_id=uniprot_id,
            gene_name=gene_name,
            target_seq=target_seq,
            data=df
        )

        self._assay_cache[assay_id] = assay
        return assay

    def load_all_assays(self, max_assays: Optional[int] = None) -> Dict[str, DMSAssay]:
        """
        Load all available DMS assays.

        Args:
            max_assays: Maximum number of assays to load (for testing)

        Returns:
            Dictionary mapping assay_id to DMSAssay objects
        """
        assay_ids = self.list_assays()
        if max_assays:
            assay_ids = assay_ids[:max_assays]

        assays = {}
        for assay_id in assay_ids:
            try:
                assays[assay_id] = self.load_assay(assay_id)
            except Exception as e:
                print(f"Warning: Failed to load {assay_id}: {e}")

        return assays


def parse_mutation(mutation_str: str) -> List[Tuple[str, int, str]]:
    """
    Parse a ProteinGym mutation string into components.

    Args:
        mutation_str: Mutation string like "A1P" or "A1P:D2N" for multiple

    Returns:
        List of tuples: (wild_type_aa, position, mutant_aa)
    """
    mutations = []
    for mut in mutation_str.split(":"):
        if not mut:
            continue
        # Handle edge cases
        mut = mut.strip()
        if len(mut) < 3:
            continue

        wt = mut[0]
        mt = mut[-1]
        try:
            pos = int(mut[1:-1])
            mutations.append((wt, pos, mt))
        except ValueError:
            continue

    return mutations


def is_single_mutation(mutation_str: str) -> bool:
    """Check if mutation string represents a single amino acid substitution"""
    return ":" not in mutation_str


def get_mutation_positions(mutation_str: str) -> List[int]:
    """Extract all positions from a mutation string"""
    mutations = parse_mutation(mutation_str)
    return [pos for _, pos, _ in mutations]


def filter_single_mutations(df: pd.DataFrame, mutant_col: str = "mutant") -> pd.DataFrame:
    """Filter dataframe to only include single-point mutations"""
    return df[df[mutant_col].apply(is_single_mutation)].copy()


def get_assay_statistics(assay: DMSAssay) -> Dict:
    """
    Calculate statistics for a DMS assay.

    Returns:
        Dictionary with statistics about the assay
    """
    df = assay.data

    stats = {
        "assay_id": assay.assay_id,
        "gene_name": assay.gene_name,
        "uniprot_id": assay.uniprot_id,
        "n_variants": len(df),
        "seq_length": len(assay.target_seq) if assay.target_seq else None,
    }

    # Count single vs multi-mutations
    if "mutant" in df.columns:
        single_mask = df["mutant"].apply(is_single_mutation)
        stats["n_single_mutations"] = single_mask.sum()
        stats["n_multi_mutations"] = (~single_mask).sum()

    # DMS score statistics
    if "DMS_score" in df.columns:
        stats["dms_score_mean"] = df["DMS_score"].mean()
        stats["dms_score_std"] = df["DMS_score"].std()
        stats["dms_score_min"] = df["DMS_score"].min()
        stats["dms_score_max"] = df["DMS_score"].max()

    # Binary classification statistics
    if "DMS_score_bin" in df.columns:
        stats["n_fit"] = (df["DMS_score_bin"] == 1).sum()
        stats["n_not_fit"] = (df["DMS_score_bin"] == 0).sum()
        stats["fraction_fit"] = stats["n_fit"] / len(df)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProteinGym data loader utilities")
    parser.add_argument("data_dir", type=str, help="Path to ProteinGym data directory")
    parser.add_argument("--list", action="store_true", help="List available assays")
    parser.add_argument("--gene", type=str, help="Filter by gene name")
    parser.add_argument("--stats", type=str, help="Show statistics for specific assay")

    args = parser.parse_args()

    loader = ProteinGymLoader(args.data_dir)

    if args.list:
        assays = loader.list_assays(gene_filter=args.gene)
        print(f"\nFound {len(assays)} assays:")
        for assay_id in assays[:20]:
            print(f"  {assay_id}")
        if len(assays) > 20:
            print(f"  ... and {len(assays) - 20} more")

    if args.stats:
        assay = loader.load_assay(args.stats)
        stats = get_assay_statistics(assay)
        print(f"\nStatistics for {args.stats}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
