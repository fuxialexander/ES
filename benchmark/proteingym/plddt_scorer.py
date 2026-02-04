#!/usr/bin/env python3
"""
pLDDT Scorer for ProteinGym

Computes pLDDT-based scores for ProteinGym benchmark proteins.
Uses only the structural gradient signal without evolutionary information.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
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


class PLDDTScorer:
    """
    Compute pLDDT-based scores for ProteinGym proteins.

    Uses only the structural gradient signal derived from AlphaFold pLDDT scores.
    This serves as a baseline to compare against the full ES Score.
    """

    def __init__(
        self,
        plddt_file: Union[str, Path],
        uniprot_mapping_file: Union[str, Path],
        smooth_kernel: int = 10,
        smooth_method: str = 'gaussian',
    ):
        """
        Initialize pLDDT Scorer.

        Args:
            plddt_file: Path to AlphaFold pLDDT scores file
            uniprot_mapping_file: Path to UniProt to gene name mapping
            smooth_kernel: Smoothing kernel size for pLDDT
            smooth_method: 'gaussian' or 'conv'
        """
        self.plddt_file = Path(plddt_file)
        self.uniprot_mapping_file = Path(uniprot_mapping_file)
        self.smooth_kernel = smooth_kernel
        self.smooth_method = smooth_method

        # Load mappings
        self._load_plddt()
        self._load_uniprot_mapping()

        # Cache for computed scores
        self._score_cache = {}

    def _load_plddt(self):
        """Load pLDDT scores from file"""
        self.plddt = {}
        with open(self.plddt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    uid, scores = parts[0], parts[1]
                    self.plddt[uid] = np.array(scores.split(",")).astype(float)

    def _load_uniprot_mapping(self):
        """Load UniProt to gene name mapping"""
        df = pd.read_csv(self.uniprot_mapping_file, sep='\t')
        # Handle both directions of mapping
        if 'From' in df.columns and 'To' in df.columns:
            self.uniprot_to_gene = df.set_index('From')['To'].to_dict()
            self.gene_to_uniprot = df.set_index('To')['From'].to_dict()
            # Also add uppercase gene names for case-insensitive lookup
            for gene, uniprot in list(self.gene_to_uniprot.items()):
                self.gene_to_uniprot[gene.upper()] = uniprot
        else:
            self.uniprot_to_gene = {}
            self.gene_to_uniprot = {}

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Apply smoothing to array"""
        if self.smooth_method == 'conv':
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            return np.convolve(arr, kernel, mode='same')
        else:  # gaussian
            return gaussian_filter1d(arr, sigma=self.smooth_kernel / 2)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalization"""
        x_min = x.min()
        x_range = x.max() - x_min
        if x_range == 0:
            return np.zeros_like(x)
        return (x - x_min) / x_range

    def _square_grad(self, f: np.ndarray) -> np.ndarray:
        """Compute normalized squared gradient"""
        grad = np.gradient(f) ** 2
        return self._normalize(grad)

    def compute_plddt_scores(
        self,
        uniprot_id: str,
        gene_name: Optional[str] = None,
        seq_length: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute pLDDT-based scores for a protein.

        Args:
            uniprot_id: UniProt accession ID
            gene_name: Optional gene name (used for UniProt fallback)
            seq_length: Expected sequence length (for validation)

        Returns:
            Array of pLDDT-based scores for each position, or None if computation fails
        """
        cache_key = f"{uniprot_id}_{gene_name}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Get pLDDT scores
        plddt = None

        # First, try direct UniProt ID lookup
        if uniprot_id and uniprot_id in self.plddt:
            plddt = self.plddt[uniprot_id]
        elif uniprot_id:
            # Try alternative UniProt ID formats
            alt_ids = [
                uniprot_id.split("-")[0],  # Remove isoform
                uniprot_id.split(".")[0],  # Remove version
            ]
            for alt_id in alt_ids:
                if alt_id in self.plddt:
                    plddt = self.plddt[alt_id]
                    break

        # If still no pLDDT, try gene name -> UniProt lookup
        if plddt is None and gene_name:
            gene_upper = gene_name.upper()
            if gene_upper in self.gene_to_uniprot:
                lookup_uniprot = self.gene_to_uniprot[gene_upper]
                if lookup_uniprot in self.plddt:
                    plddt = self.plddt[lookup_uniprot]

        if plddt is None:
            return None

        # Normalize pLDDT to 0-1
        plddt = plddt / 100.0 if plddt.max() > 1 else plddt

        # Compute gradient-based structural score
        smooth_plddt = self._smooth(plddt)
        grad = self._square_grad(smooth_plddt)

        # Clip to reduce outliers
        grad = np.clip(grad, np.quantile(grad, 0.2), np.quantile(grad, 0.8))
        plddt_scores = self._normalize(grad)

        self._score_cache[cache_key] = plddt_scores
        return plddt_scores

    def score_mutations(
        self,
        assay: DMSAssay,
        single_only: bool = True
    ) -> pd.DataFrame:
        """
        Score all mutations in a DMS assay using pLDDT-based scoring.

        Args:
            assay: DMSAssay object
            single_only: Only score single-point mutations

        Returns:
            DataFrame with pLDDT scores for each mutation
        """
        # Get protein-level pLDDT scores
        plddt_scores = self.compute_plddt_scores(
            assay.uniprot_id,
            gene_name=assay.gene_name,
            seq_length=len(assay.target_seq) if assay.target_seq else None
        )

        if plddt_scores is None:
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

            # Get pLDDT score for mutation position(s)
            mutation_scores = []
            for wt, pos, mt in mutations:
                if 0 < pos <= len(plddt_scores):
                    mutation_scores.append(plddt_scores[pos - 1])

            if not mutation_scores:
                continue

            # For multi-mutations, use mean of position scores
            plddt_score = np.mean(mutation_scores)

            results.append({
                "mutant": mutant,
                "plddt_score": plddt_score,
                "DMS_score": row.get("DMS_score"),
                "DMS_score_bin": row.get("DMS_score_bin"),
                "n_mutations": len(mutations),
                "positions": [m[1] for m in mutations]
            })

        return pd.DataFrame(results)

    def score_all_assays(
        self,
        loader: ProteinGymLoader,
        assay_ids: Optional[List[str]] = None,
        single_only: bool = True,
        min_variants: int = 10
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

        for assay_id in tqdm(assay_ids, desc="Scoring assays with pLDDT"):
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
            print(f"\nFailed to score {len(failed)} assays with pLDDT:")
            for assay_id, reason in failed[:5]:
                print(f"  {assay_id}: {reason}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        return results


def create_plddt_scorer_from_project(
    project_root: Optional[Path] = None,
    **kwargs
) -> PLDDTScorer:
    """
    Create a PLDDTScorer using the ES Score project's default files.

    Args:
        project_root: Path to ES Score project root
        **kwargs: Additional arguments for PLDDTScorer

    Returns:
        Configured PLDDTScorer instance
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    return PLDDTScorer(
        plddt_file=project_root / "plddt" / "9606.pLDDT.tdt",
        uniprot_mapping_file=project_root / "uniprot_to_genename.txt",
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="pLDDT-only scorer for ProteinGym")
    parser.add_argument("data_dir", type=str, help="ProteinGym data directory")
    parser.add_argument("--assay", type=str, help="Score specific assay")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument("--smooth_kernel", type=int, default=10)

    args = parser.parse_args()

    # Create scorer
    scorer = create_plddt_scorer_from_project(
        smooth_kernel=args.smooth_kernel,
    )

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
