#!/usr/bin/env python3
"""
ESM LLR Scorer for ProteinGym

Computes ESM (Evolutionary Scale Modeling) Log-Likelihood Ratio scores directly
for ProteinGym benchmark proteins, without the pLDDT gradient component.

This provides a baseline to understand the contribution of the evolutionary
signal alone vs the combined ES Score (evolutionary + structural).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proteingym_loader import (
    DMSAssay,
    ProteinGymLoader,
    parse_mutation,
    is_single_mutation,
)


class ESMLLRScorer:
    """
    Compute ESM LLR scores directly for ProteinGym proteins.

    ESM LLR = -mean(log-likelihood ratios for all possible mutations at position)

    Higher scores indicate positions where mutations are more deleterious
    according to the evolutionary model.
    """

    def __init__(
        self,
        esm_dir: Optional[Union[str, Path]] = None,
        esm_gene_dir: Optional[Union[str, Path]] = None,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize ESM LLR Scorer.

        Args:
            esm_dir: Directory containing ESM LLR files by UniProt ID
            esm_gene_dir: Directory containing ESM LLR files by gene name
            uniprot_mapping_file: Path to UniProt to gene name mapping
        """
        # Set default directories
        if esm_dir is None:
            esm_dir = PROJECT_ROOT / "esm1b_LLR"
        if esm_gene_dir is None:
            esm_gene_dir = PROJECT_ROOT / "esm_ALL_hotspot"

        self.esm_dir = Path(esm_dir) if esm_dir else None
        self.esm_gene_dir = Path(esm_gene_dir) if esm_gene_dir else None

        # Load UniProt mapping
        self.uniprot_to_gene = {}
        self.gene_to_uniprot = {}
        if uniprot_mapping_file:
            self._load_uniprot_mapping(uniprot_mapping_file)
        else:
            default_mapping = PROJECT_ROOT / "uniprot_to_genename.txt"
            if default_mapping.exists():
                self._load_uniprot_mapping(default_mapping)

        # Cache for ESM scores
        self._esm_cache: Dict[str, np.ndarray] = {}
        self._variant_cache: Dict[str, Dict[str, float]] = {}

    def _load_uniprot_mapping(self, filepath: Union[str, Path]) -> None:
        """Load UniProt to gene name mapping."""
        df = pd.read_csv(filepath, sep='\t')
        if 'From' in df.columns and 'To' in df.columns:
            self.uniprot_to_gene = df.set_index('From')['To'].to_dict()
            self.gene_to_uniprot = df.set_index('To')['From'].to_dict()
            # Add uppercase gene names for case-insensitive lookup
            for gene, uniprot in list(self.gene_to_uniprot.items()):
                self.gene_to_uniprot[gene.upper()] = uniprot

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        x_min = x.min()
        x_range = x.max() - x_min
        if x_range == 0:
            return np.zeros_like(x)
        return (x - x_min) / x_range

    def _load_esm_by_uniprot(self, uniprot_id: str) -> Optional[Dict[str, float]]:
        """
        Load ESM LLR scores by UniProt ID.

        Returns dict mapping variant string (e.g., 'A1G') to LLR score.
        """
        cache_key = f"uniprot_{uniprot_id}"
        if cache_key in self._variant_cache:
            return self._variant_cache[cache_key]

        if self.esm_dir is None or not self.esm_dir.exists():
            return None

        # Try different filename patterns
        patterns = [
            self.esm_dir / f"{uniprot_id}_LLR.csv",
            self.esm_dir / f"{uniprot_id}.csv",
        ]

        for path in patterns:
            if path.exists():
                try:
                    df = pd.read_csv(path, index_col=0)
                    # df has amino acids as rows, positions as columns
                    # Column names are like "M 1", "L 2", etc.
                    variant_scores = {}

                    for col in df.columns:
                        # Parse position from column name
                        parts = col.strip().split()
                        if len(parts) == 2:
                            ref_aa = parts[0]
                            try:
                                pos = int(parts[1])
                            except ValueError:
                                continue

                            # Get scores for all alternative amino acids
                            for alt_aa in df.index:
                                if alt_aa != ref_aa:
                                    score = df.loc[alt_aa, col]
                                    if not np.isnan(score):
                                        variant_key = f"{ref_aa}{pos}{alt_aa}"
                                        # Negative LLR means more deleterious
                                        variant_scores[variant_key] = -score

                    self._variant_cache[cache_key] = variant_scores
                    return variant_scores
                except Exception as e:
                    print(f"Warning: Failed to load ESM for {uniprot_id}: {e}")

        return None

    def _load_esm_by_gene(self, gene_name: str) -> Optional[Dict[str, float]]:
        """
        Load ESM LLR scores by gene name.

        Returns dict mapping variant string (e.g., 'A1G') to LLR score.
        """
        cache_key = f"gene_{gene_name}"
        if cache_key in self._variant_cache:
            return self._variant_cache[cache_key]

        if self.esm_gene_dir is None or not self.esm_gene_dir.exists():
            return None

        path = self.esm_gene_dir / f"{gene_name}.csv"
        if not path.exists():
            # Try uppercase
            path = self.esm_gene_dir / f"{gene_name.upper()}.csv"

        if path.exists():
            try:
                df = pd.read_csv(path)
                variant_scores = {}

                # Format: Index, Mutation, Gene, ESM1b_LLR, ...
                if 'Mutation' in df.columns and 'ESM1b_LLR' in df.columns:
                    for _, row in df.iterrows():
                        mutation = row['Mutation']
                        llr = row['ESM1b_LLR']
                        if not np.isnan(llr):
                            # Negative LLR means more deleterious
                            variant_scores[mutation] = -llr
                else:
                    # Alternative format with multiple LLR columns
                    for _, row in df.iterrows():
                        mutation = row['Mutation']
                        # Average across LLR columns
                        llr_cols = [c for c in df.columns if 'LLR' in c.upper() or c.startswith('ESM')]
                        if llr_cols:
                            llr = row[llr_cols].astype(float).mean()
                        else:
                            # Use columns after the first 3 (Index, Mutation, Gene)
                            llr = row.iloc[3:].astype(float).mean()
                        if not np.isnan(llr):
                            variant_scores[mutation] = -llr

                self._variant_cache[cache_key] = variant_scores
                return variant_scores
            except Exception as e:
                print(f"Warning: Failed to load ESM for gene {gene_name}: {e}")

        return None

    def get_variant_score(
        self,
        uniprot_id: str,
        gene_name: Optional[str],
        variant: str,
    ) -> Optional[float]:
        """
        Get ESM LLR score for a specific variant.

        Args:
            uniprot_id: UniProt accession ID
            gene_name: Gene name
            variant: Variant string (e.g., 'A123G')

        Returns:
            ESM LLR score or None if not found
        """
        # Try gene name first (more likely to have data)
        if gene_name:
            scores = self._load_esm_by_gene(gene_name)
            if scores and variant in scores:
                return scores[variant]

        # Try UniProt ID
        if uniprot_id:
            scores = self._load_esm_by_uniprot(uniprot_id)
            if scores and variant in scores:
                return scores[variant]

            # Try without isoform
            base_id = uniprot_id.split("-")[0]
            if base_id != uniprot_id:
                scores = self._load_esm_by_uniprot(base_id)
                if scores and variant in scores:
                    return scores[variant]

        return None

    def score_mutations(
        self,
        assay: DMSAssay,
        single_only: bool = True,
    ) -> pd.DataFrame:
        """
        Score all mutations in a DMS assay using ESM LLR.

        Args:
            assay: DMSAssay object
            single_only: Only score single-point mutations

        Returns:
            DataFrame with ESM LLR scores for each mutation
        """
        uniprot_id = assay.uniprot_id
        gene_name = assay.gene_name

        # Try to get variant scores
        variant_scores = None

        # First try gene name
        if gene_name:
            variant_scores = self._load_esm_by_gene(gene_name)

        # Then try UniProt ID
        if variant_scores is None and uniprot_id:
            variant_scores = self._load_esm_by_uniprot(uniprot_id)
            if variant_scores is None:
                base_id = uniprot_id.split("-")[0]
                variant_scores = self._load_esm_by_uniprot(base_id)

        if variant_scores is None:
            return pd.DataFrame()

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

            # Get ESM LLR score for mutation(s)
            mutation_scores = []
            for wt, pos, mt in mutations:
                variant_str = f"{wt}{pos}{mt}"
                if variant_str in variant_scores:
                    mutation_scores.append(variant_scores[variant_str])

            if not mutation_scores:
                continue

            # For multi-mutations, use mean of scores
            esm_score = np.mean(mutation_scores)

            results.append({
                "mutant": mutant,
                "es_score": esm_score,  # Use es_score for compatibility with evaluator
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
        if assay_ids is None:
            assay_ids = loader.list_assays()

        results = {}
        failed = []

        for assay_id in tqdm(assay_ids, desc="Scoring assays with ESM LLR"):
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
            print(f"\nFailed to score {len(failed)} assays with ESM LLR:")
            for assay_id, reason in failed[:5]:
                print(f"  {assay_id}: {reason}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        return results


def create_esm_llr_scorer(**kwargs) -> ESMLLRScorer:
    """
    Create an ESMLLRScorer with default configuration.

    Returns:
        Configured ESMLLRScorer instance
    """
    return ESMLLRScorer(
        esm_dir=PROJECT_ROOT / "esm1b_LLR",
        esm_gene_dir=PROJECT_ROOT / "esm_ALL_hotspot",
        uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESM LLR scorer for ProteinGym")
    parser.add_argument("data_dir", type=str, help="ProteinGym data directory")
    parser.add_argument("--assay", type=str, help="Score specific assay")
    parser.add_argument("--output", type=str, help="Output CSV file")

    args = parser.parse_args()

    # Create scorer
    scorer = create_esm_llr_scorer()

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
