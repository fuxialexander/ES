#!/usr/bin/env python3
"""
ES Score Calculator for ProteinGym

Computes ES (Evolutionary-Structural) scores for ProteinGym benchmark proteins.
Integrates with the ES Score project's scoring pipeline.
"""

import os
import sys
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Add parent directories to path for ES Score imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proteingym_loader import (
    DMSAssay,
    ProteinGymLoader,
    parse_mutation,
    is_single_mutation,
    get_mutation_positions
)


class ESScorer:
    """
    Compute ES scores for ProteinGym proteins.

    ES Score = normalized(gradient(pLDDT) * ESM_score)

    Where:
    - pLDDT: AlphaFold confidence scores
    - ESM_score: Evolutionary scores from ESM language model
    """

    def __init__(
        self,
        plddt_file: Union[str, Path],
        uniprot_mapping_file: Union[str, Path],
        esm_dir: Optional[Union[str, Path]] = None,
        structures_dir: Optional[Union[str, Path]] = None,
        smooth_kernel: int = 10,
        smooth_method: str = 'gaussian',
        interaction_threshold: int = 15,
        use_3d: bool = False
    ):
        """
        Initialize ES Scorer.

        Args:
            plddt_file: Path to AlphaFold pLDDT scores file
            uniprot_mapping_file: Path to UniProt to gene name mapping
            esm_dir: Directory containing ESM scores (optional)
            structures_dir: Directory containing AlphaFold structures
            smooth_kernel: Smoothing kernel size for pLDDT
            smooth_method: 'gaussian' or 'conv'
            interaction_threshold: Angstrom threshold for 3D interactions
            use_3d: Whether to use 3D spatial averaging
        """
        self.plddt_file = Path(plddt_file)
        self.uniprot_mapping_file = Path(uniprot_mapping_file)
        self.esm_dir = Path(esm_dir) if esm_dir else None
        self.structures_dir = Path(structures_dir) if structures_dir else None

        self.smooth_kernel = smooth_kernel
        self.smooth_method = smooth_method
        self.interaction_threshold = interaction_threshold
        self.use_3d = use_3d

        # Load mappings
        self._load_plddt()
        self._load_uniprot_mapping()

        # Cache for computed scores
        self._score_cache = {}
        self._esm_cache = {}

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

    def _load_esm_scores(self, uniprot_id: str) -> Optional[np.ndarray]:
        """Load ESM scores for a protein"""
        if uniprot_id in self._esm_cache:
            return self._esm_cache[uniprot_id]

        if self.esm_dir is None:
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
                    # Average ESM scores across all mutations at each position
                    esm_scores = -df.mean(axis=0).values
                    esm_scores = self._normalize(esm_scores)
                    self._esm_cache[uniprot_id] = esm_scores
                    return esm_scores
                except Exception as e:
                    print(f"Warning: Failed to load ESM for {uniprot_id}: {e}")

        return None

    def _load_esm_for_gene(self, gene_name: str) -> Optional[np.ndarray]:
        """Load ESM scores using gene name"""
        esm_gene_dir = PROJECT_ROOT / "esm_ALL_hotspot"
        path = esm_gene_dir / f"{gene_name}.csv"

        if path.exists():
            try:
                df = pd.read_csv(path)
                df['esm'] = df.iloc[:, 3:].astype(float).mean(axis=1)

                # Extract position from mutation string
                df['pos'] = df['Mutation'].apply(lambda x: int(x[1:-1]))

                # Aggregate by position
                pos_esm = df.groupby('pos')['esm'].mean()

                # Create full-length array
                max_pos = pos_esm.index.max()
                esm_scores = np.zeros(max_pos)
                for pos, score in pos_esm.items():
                    if 0 < pos <= max_pos:
                        esm_scores[pos - 1] = -score

                return self._normalize(esm_scores)
            except Exception as e:
                print(f"Warning: Failed to load ESM for gene {gene_name}: {e}")

        return None

    def compute_es_scores(
        self,
        uniprot_id: str,
        gene_name: Optional[str] = None,
        seq_length: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute ES scores for a protein.

        Args:
            uniprot_id: UniProt accession ID
            gene_name: Optional gene name (used for ESM lookup)
            seq_length: Expected sequence length (for validation)

        Returns:
            Array of ES scores for each position, or None if computation fails
        """
        cache_key = f"{uniprot_id}_{gene_name}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Get pLDDT scores
        if uniprot_id not in self.plddt:
            # Try alternative UniProt ID formats
            alt_ids = [
                uniprot_id.split("-")[0],  # Remove isoform
                uniprot_id.split(".")[0],  # Remove version
            ]
            plddt = None
            for alt_id in alt_ids:
                if alt_id in self.plddt:
                    plddt = self.plddt[alt_id]
                    break
            if plddt is None:
                return None
        else:
            plddt = self.plddt[uniprot_id]

        # Normalize pLDDT to 0-1
        plddt = plddt / 100.0 if plddt.max() > 1 else plddt

        # Compute gradient-based structural score
        smooth_plddt = self._smooth(plddt)
        grad = self._square_grad(smooth_plddt)

        # Clip to reduce outliers
        grad = np.clip(grad, np.quantile(grad, 0.2), np.quantile(grad, 0.8))
        grad = self._normalize(grad)

        # Get ESM evolutionary scores
        esm_scores = None
        if gene_name:
            esm_scores = self._load_esm_for_gene(gene_name)
        if esm_scores is None:
            esm_scores = self._load_esm_scores(uniprot_id)
        if esm_scores is None:
            # Use uniform ESM if not available
            esm_scores = np.ones_like(plddt)

        # Ensure matching lengths
        min_len = min(len(grad), len(esm_scores))
        grad = grad[:min_len]
        esm_scores = esm_scores[:min_len]

        # Compute ES score
        es_scores = grad * esm_scores
        es_scores = self._normalize(es_scores)

        self._score_cache[cache_key] = es_scores
        return es_scores

    def score_mutations(
        self,
        assay: DMSAssay,
        single_only: bool = True
    ) -> pd.DataFrame:
        """
        Score all mutations in a DMS assay.

        Args:
            assay: DMSAssay object
            single_only: Only score single-point mutations

        Returns:
            DataFrame with ES scores for each mutation
        """
        # Get protein-level ES scores
        es_scores = self.compute_es_scores(
            assay.uniprot_id,
            gene_name=assay.gene_name,
            seq_length=len(assay.target_seq) if assay.target_seq else None
        )

        if es_scores is None:
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

            # Get ES score for mutation position(s)
            mutation_scores = []
            for wt, pos, mt in mutations:
                if 0 < pos <= len(es_scores):
                    mutation_scores.append(es_scores[pos - 1])

            if not mutation_scores:
                continue

            # For multi-mutations, use mean of position scores
            es_score = np.mean(mutation_scores)

            results.append({
                "mutant": mutant,
                "es_score": es_score,
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

        for assay_id in tqdm(assay_ids, desc="Scoring assays"):
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
            print(f"\nFailed to score {len(failed)} assays:")
            for assay_id, reason in failed[:5]:
                print(f"  {assay_id}: {reason}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        return results


def create_scorer_from_project(
    project_root: Optional[Path] = None,
    **kwargs
) -> ESScorer:
    """
    Create an ESScorer using the ES Score project's default files.

    Args:
        project_root: Path to ES Score project root
        **kwargs: Additional arguments for ESScorer

    Returns:
        Configured ESScorer instance
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    return ESScorer(
        plddt_file=project_root / "plddt" / "9606.pLDDT.tdt",
        uniprot_mapping_file=project_root / "uniprot_to_genename.txt",
        esm_dir=project_root / "esm1b" / "content" / "ALL_hum_isoforms_ESM1b_LLR",
        structures_dir=project_root / "structures",
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ES Score calculator for ProteinGym")
    parser.add_argument("data_dir", type=str, help="ProteinGym data directory")
    parser.add_argument("--assay", type=str, help="Score specific assay")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument("--smooth_kernel", type=int, default=10)
    parser.add_argument("--use_3d", action="store_true")

    args = parser.parse_args()

    # Create scorer
    scorer = create_scorer_from_project(
        smooth_kernel=args.smooth_kernel,
        use_3d=args.use_3d
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
