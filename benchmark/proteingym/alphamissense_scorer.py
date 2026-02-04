#!/usr/bin/env python3
"""
AlphaMissense Scorer for ProteinGym

Computes AlphaMissense pathogenicity scores for ProteinGym benchmark proteins.
Integrates with the AlphaMissense prediction fetcher module.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proteingym_loader import (
    DMSAssay,
    ProteinGymLoader,
    parse_mutation,
    is_single_mutation,
)
from benchmark.alphamissense import AlphaMissenseFetcher


class AlphaMissenseScorer:
    """
    Compute AlphaMissense scores for ProteinGym proteins.

    Uses the AlphaMissense bulk data to retrieve pathogenicity scores
    for protein variants.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "/mnt/storage/alphamissense",
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize AlphaMissense Scorer.

        Args:
            data_dir: Directory containing AlphaMissense bulk data files
            uniprot_mapping_file: Path to UniProt to gene name mapping
        """
        self.data_dir = Path(data_dir)

        # Use default mapping file if not provided
        if uniprot_mapping_file is None:
            uniprot_mapping_file = PROJECT_ROOT / "uniprot_to_genename.txt"

        self.uniprot_mapping_file = Path(uniprot_mapping_file) if uniprot_mapping_file else None

        # Initialize the AlphaMissense fetcher
        self._fetcher: Optional[AlphaMissenseFetcher] = None

        # Cache for protein scores
        self._protein_cache: Dict[str, pd.DataFrame] = {}

    @property
    def fetcher(self) -> AlphaMissenseFetcher:
        """Get AlphaMissense fetcher (lazy initialization)."""
        if self._fetcher is None:
            self._fetcher = AlphaMissenseFetcher(
                data_dir=self.data_dir,
                uniprot_mapping_file=self.uniprot_mapping_file,
            )
        return self._fetcher

    def has_bulk_data(self) -> bool:
        """Check if bulk data is available."""
        return self.fetcher.has_bulk_data()

    def ensure_data_available(self) -> bool:
        """
        Ensure AlphaMissense bulk data is available.

        Returns:
            True if data is available or was successfully downloaded
        """
        if self.has_bulk_data():
            return True

        print("AlphaMissense bulk data not found. Attempting to download...")
        results = self.fetcher.download_bulk_data()
        return results.get("aa_substitutions", False)

    def get_protein_scores(self, uniprot_id: str) -> Optional[pd.DataFrame]:
        """
        Get all AlphaMissense scores for a protein.

        Args:
            uniprot_id: UniProt accession ID

        Returns:
            DataFrame with all variants and scores, or None if not found
        """
        if uniprot_id in self._protein_cache:
            return self._protein_cache[uniprot_id]

        df = self.fetcher.get_scores_for_protein(uniprot_id)
        if df is not None:
            self._protein_cache[uniprot_id] = df

        return df

    def score_variant(
        self,
        uniprot_id: str,
        ref_aa: str,
        position: int,
        alt_aa: str,
    ) -> Optional[float]:
        """
        Get AlphaMissense score for a specific variant.

        Args:
            uniprot_id: UniProt accession ID
            ref_aa: Reference amino acid
            position: 1-based position
            alt_aa: Alternate amino acid

        Returns:
            Pathogenicity score (0-1), or None if not found
        """
        variant = f"{ref_aa}{position}{alt_aa}"
        result = self.fetcher.get_score(uniprot_id, variant)
        if result:
            return result[0]  # Return just the score, not the classification
        return None

    def score_mutations(
        self,
        assay: DMSAssay,
        single_only: bool = True,
    ) -> pd.DataFrame:
        """
        Score all mutations in a DMS assay.

        Args:
            assay: DMSAssay object
            single_only: Only score single-point mutations

        Returns:
            DataFrame with AlphaMissense scores for each mutation
        """
        # Determine UniProt ID: use assay.uniprot_id if available,
        # otherwise extract from assay_id (format: UNIPROT_SPECIES_Author_Year)
        uniprot_id = assay.uniprot_id
        if not uniprot_id:
            # Try to extract from assay_id
            parts = assay.assay_id.split("_")
            if parts:
                uniprot_id = parts[0]

        if not uniprot_id:
            return pd.DataFrame()

        # Get all scores for this protein
        protein_scores = self.get_protein_scores(uniprot_id)
        if protein_scores is None:
            # Try without isoform suffix
            base_id = uniprot_id.split("-")[0]
            protein_scores = self.get_protein_scores(base_id)
            if protein_scores is None:
                return pd.DataFrame()

        # Create a lookup dictionary for fast access
        score_lookup = {}
        for _, row in protein_scores.iterrows():
            variant = row["protein_variant"]
            score_lookup[variant] = row["am_pathogenicity"]

        results = []
        df = assay.data.copy()

        for idx, row in df.iterrows():
            mutant = row.get("mutant", "")

            # Filter multi-mutations if requested
            if single_only and not is_single_mutation(mutant):
                continue

            # Parse mutation positions
            mutations = parse_mutation(mutant)
            if not mutations:
                continue

            # Get AlphaMissense score for mutation position(s)
            mutation_scores = []
            for wt, pos, mt in mutations:
                variant_str = f"{wt}{pos}{mt}"
                if variant_str in score_lookup:
                    score = score_lookup[variant_str]
                    if not np.isnan(score):
                        mutation_scores.append(score)

            if not mutation_scores:
                continue

            # For multi-mutations, use mean of position scores
            am_score = np.mean(mutation_scores)

            results.append({
                "mutant": mutant,
                "am_score": am_score,
                "DMS_score": row.get("DMS_score"),
                "DMS_score_bin": row.get("DMS_score_bin"),
                "n_mutations": len(mutations),
                "positions": [m[1] for m in mutations],
            })

        return pd.DataFrame(results)

    def score_all_assays(
        self,
        loader: ProteinGymLoader,
        assay_ids: Optional[List[str]] = None,
        single_only: bool = True,
        min_variants: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Score all assays in a ProteinGym dataset.

        Args:
            loader: ProteinGymLoader instance
            assay_ids: List of assay IDs to score (default: all)
            single_only: Only score single-point mutations
            min_variants: Minimum variants required to include assay

        Returns:
            Dictionary mapping assay_id to scored DataFrame
        """
        if not self.ensure_data_available():
            raise RuntimeError(
                "AlphaMissense bulk data not available. "
                "Run AlphaMissenseFetcher.download_bulk_data() first."
            )

        if assay_ids is None:
            assay_ids = loader.list_assays()

        results = {}
        failed = []

        for assay_id in tqdm(assay_ids, desc="Scoring assays with AlphaMissense"):
            try:
                assay = loader.load_assay(assay_id)
                scored = self.score_mutations(assay, single_only=single_only)

                if len(scored) >= min_variants:
                    results[assay_id] = scored
                else:
                    failed.append((assay_id, f"Only {len(scored)} variants"))
            except Exception as e:
                failed.append((assay_id, str(e)))

        if failed:
            print(f"\nFailed to score {len(failed)} assays with AlphaMissense:")
            for assay_id, reason in failed[:5]:
                print(f"  {assay_id}: {reason}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        return results


def create_alphamissense_scorer(
    data_dir: Union[str, Path] = "/mnt/storage/alphamissense",
    **kwargs
) -> AlphaMissenseScorer:
    """
    Create an AlphaMissenseScorer with default configuration.

    Args:
        data_dir: Directory containing AlphaMissense bulk data
        **kwargs: Additional arguments for AlphaMissenseScorer

    Returns:
        Configured AlphaMissenseScorer instance
    """
    return AlphaMissenseScorer(
        data_dir=data_dir,
        uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AlphaMissense scorer for ProteinGym"
    )
    parser.add_argument("data_dir", type=str, help="ProteinGym data directory")
    parser.add_argument("--assay", type=str, help="Score specific assay")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument(
        "--am_data_dir",
        type=str,
        default="/mnt/storage/alphamissense",
        help="AlphaMissense data directory",
    )

    args = parser.parse_args()

    # Create scorer
    scorer = create_alphamissense_scorer(data_dir=args.am_data_dir)

    # Load data
    loader = ProteinGymLoader(args.data_dir)

    if args.assay:
        # Score single assay
        assay = loader.load_assay(args.assay)
        scored = scorer.score_mutations(assay)

        print(f"\nScored {len(scored)} variants for {args.assay}")
        print(scored.head(10))

        if args.output:
            scored.to_csv(args.output, index=False)
    else:
        # Score all assays
        results = scorer.score_all_assays(loader)
        print(f"\nScored {len(results)} assays")

        if args.output:
            # Combine all results
            all_results = pd.concat(
                [df.assign(assay_id=aid) for aid, df in results.items()],
                ignore_index=True
            )
            all_results.to_csv(args.output, index=False)
