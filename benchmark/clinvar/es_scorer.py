#!/usr/bin/env python3
"""
ES Scorer for ClinVar Benchmark

Computes ES scores for ClinVar pathogenic/benign variants.
Adapts the ES scoring algorithm for variant pathogenicity prediction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from clinvar_loader import ClinVarLoader, ClinVarData, parse_mutation


class ClinVarScorer:
    """
    Compute ES scores for ClinVar variants.

    ES Score = normalized(gradient(pLDDT) * ESM_score)

    Higher ES scores indicate positions more likely to be functionally important
    (pathogenic variants).
    """

    def __init__(
        self,
        plddt_file: Union[str, Path],
        uniprot_mapping_file: Union[str, Path],
        esm_dir: Optional[Union[str, Path]] = None,
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
            esm_dir: Directory containing ESM scores
            smooth_kernel: Smoothing kernel size for pLDDT
            smooth_method: 'gaussian' or 'conv'
            interaction_threshold: Angstrom threshold for 3D interactions
            use_3d: Whether to use 3D spatial averaging
        """
        self.plddt_file = Path(plddt_file)
        self.uniprot_mapping_file = Path(uniprot_mapping_file)
        self.esm_dir = Path(esm_dir) if esm_dir else None

        self.smooth_kernel = smooth_kernel
        self.smooth_method = smooth_method
        self.interaction_threshold = interaction_threshold
        self.use_3d = use_3d

        # Load mappings
        self._load_plddt()
        self._load_uniprot_mapping()

        # Cache for computed scores
        self._score_cache: Dict[str, np.ndarray] = {}
        self._esm_cache: Dict[str, np.ndarray] = {}

    def _load_plddt(self):
        """Load pLDDT scores from file"""
        self.plddt = {}
        with open(self.plddt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    uid, scores = parts[0], parts[1]
                    try:
                        self.plddt[uid] = np.array(scores.split(",")).astype(float)
                    except ValueError:
                        continue

    def _load_uniprot_mapping(self):
        """Load UniProt to gene name mapping"""
        df = pd.read_csv(self.uniprot_mapping_file, sep='\t')

        self.uniprot_to_gene = {}
        self.gene_to_uniprot = {}

        if 'From' in df.columns and 'To' in df.columns:
            self.uniprot_to_gene = df.set_index('From')['To'].to_dict()
            self.gene_to_uniprot = df.set_index('To')['From'].to_dict()

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Apply smoothing to array"""
        if self.smooth_method == 'conv':
            kernel = np.ones(self.smooth_kernel) / self.smooth_kernel
            return np.convolve(arr, kernel, mode='same')
        else:
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

        patterns = [
            self.esm_dir / f"{uniprot_id}_LLR.csv",
            self.esm_dir / f"{uniprot_id}.csv",
        ]

        for path in patterns:
            if path.exists():
                try:
                    df = pd.read_csv(path, index_col=0)
                    esm_scores = -df.mean(axis=0).values
                    esm_scores = self._normalize(esm_scores)
                    self._esm_cache[uniprot_id] = esm_scores
                    return esm_scores
                except Exception:
                    continue

        return None

    def _load_esm_for_gene(self, gene_name: str) -> Optional[np.ndarray]:
        """Load ESM scores using gene name"""
        esm_gene_dir = PROJECT_ROOT / "esm_ALL_hotspot"
        path = esm_gene_dir / f"{gene_name}.csv"

        if path.exists():
            try:
                df = pd.read_csv(path)
                df['esm'] = df.iloc[:, 3:].astype(float).mean(axis=1)
                df['pos'] = df['Mutation'].apply(lambda x: int(x[1:-1]))
                pos_esm = df.groupby('pos')['esm'].mean()

                max_pos = pos_esm.index.max()
                esm_scores = np.zeros(max_pos)
                for pos, score in pos_esm.items():
                    if 0 < pos <= max_pos:
                        esm_scores[pos - 1] = -score

                return self._normalize(esm_scores)
            except Exception:
                pass

        return None

    def compute_es_scores(
        self,
        gene_name: str,
        seq_length: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute ES scores for a gene.

        Args:
            gene_name: Gene name (e.g., EGFR, KRAS)
            seq_length: Expected sequence length

        Returns:
            Array of ES scores for each position, or None if computation fails
        """
        cache_key = gene_name
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Get UniProt ID for gene
        uniprot_id = self.gene_to_uniprot.get(gene_name)

        # Try to get pLDDT
        plddt = None
        if uniprot_id and uniprot_id in self.plddt:
            plddt = self.plddt[uniprot_id]
        else:
            # Try alternative lookups
            for uid, gname in self.uniprot_to_gene.items():
                if gname == gene_name and uid in self.plddt:
                    plddt = self.plddt[uid]
                    uniprot_id = uid
                    break

        if plddt is None:
            return None

        # Normalize pLDDT to 0-1
        plddt = plddt / 100.0 if plddt.max() > 1 else plddt

        # Compute gradient-based structural score
        smooth_plddt = self._smooth(plddt)
        grad = self._square_grad(smooth_plddt)

        # Clip to reduce outliers
        grad = np.clip(grad, np.quantile(grad, 0.2), np.quantile(grad, 0.8))
        grad = self._normalize(grad)

        # Get ESM scores
        esm_scores = self._load_esm_for_gene(gene_name)
        if esm_scores is None and uniprot_id:
            esm_scores = self._load_esm_scores(uniprot_id)
        if esm_scores is None:
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

    def score_mutation(
        self,
        gene: str,
        position: int
    ) -> Optional[float]:
        """
        Score a single mutation.

        Args:
            gene: Gene name
            position: Amino acid position (1-indexed)

        Returns:
            ES score or None if unavailable
        """
        es_scores = self.compute_es_scores(gene)
        if es_scores is None:
            return None

        if 0 < position <= len(es_scores):
            return float(es_scores[position - 1])

        return None

    def score_variants(
        self,
        clinvar_data: ClinVarData,
        gene_col: str = 'GeneSymbol',
        position_col: str = 'position',
        mutation_col: str = 'Mutation'
    ) -> pd.DataFrame:
        """
        Score all ClinVar variants.

        Args:
            clinvar_data: ClinVarData object
            gene_col: Column name for gene symbols
            position_col: Column name for positions
            mutation_col: Column name for mutation strings

        Returns:
            DataFrame with ES scores added
        """
        df = clinvar_data.variants.copy()

        es_scores = []
        genes_processed = set()
        genes_failed = set()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring variants"):
            gene = row.get(gene_col)
            if pd.isna(gene):
                es_scores.append(np.nan)
                continue

            gene = str(gene).upper()

            # Get position
            position = None
            if position_col in df.columns and pd.notna(row.get(position_col)):
                try:
                    position = int(row[position_col])
                except (ValueError, TypeError):
                    pass

            # Try to parse from mutation string if position not available
            if position is None and mutation_col in df.columns:
                _, position, _ = parse_mutation(row.get(mutation_col, ''))

            if position is None or position == 0:
                es_scores.append(np.nan)
                continue

            # Score the mutation
            score = self.score_mutation(gene, position)

            if score is not None:
                genes_processed.add(gene)
                es_scores.append(score)
            else:
                genes_failed.add(gene)
                es_scores.append(np.nan)

        df["ES_score"] = es_scores

        print(f"\nScoring complete:")
        print(f"  Genes successfully scored: {len(genes_processed)}")
        print(f"  Genes failed (no pLDDT data): {len(genes_failed)}")
        print(f"  Variants scored: {df['ES_score'].notna().sum():,} / {len(df):,}")

        return df

    def get_pathogenicity_predictions(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        method: str = "percentile",
        percentile: float = 75
    ) -> pd.Series:
        """
        Convert ES scores to binary pathogenicity predictions.

        Args:
            df: DataFrame with ES_score column
            threshold: Classification threshold (default: auto-select)
            method: 'percentile', 'absolute', or 'median'
            percentile: Percentile for threshold if method='percentile'

        Returns:
            Series of binary predictions (1 = pathogenic, 0 = benign)
        """
        scores = df["ES_score"].values

        if threshold is None:
            if method == "percentile":
                threshold = np.nanpercentile(scores, percentile)
            elif method == "median":
                threshold = np.nanmedian(scores)
            else:
                threshold = 0.5

        # High ES score = more likely pathogenic
        predictions = (scores > threshold).astype(int)
        predictions = pd.Series(predictions, index=df.index)
        predictions[df["ES_score"].isna()] = np.nan

        return predictions


def create_scorer_from_project(
    project_root: Optional[Path] = None,
    **kwargs
) -> ClinVarScorer:
    """
    Create a scorer using the ES Score project's default files.

    Args:
        project_root: Path to ES Score project root
        **kwargs: Additional arguments for scorer

    Returns:
        Configured ClinVarScorer instance
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    return ClinVarScorer(
        plddt_file=project_root / "plddt" / "9606.pLDDT.tdt",
        uniprot_mapping_file=project_root / "uniprot_to_genename.txt",
        esm_dir=project_root / "esm1b" / "content" / "ALL_hum_isoforms_ESM1b_LLR",
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score ClinVar variants with ES Score")
    parser.add_argument("data_dir", type=str, help="Directory with ClinVar data")
    parser.add_argument("--output", type=str, help="Output CSV file")
    parser.add_argument("--smooth_kernel", type=int, default=10)
    parser.add_argument("--cancer-only", action="store_true",
                        help="Filter to cancer genes only")

    args = parser.parse_args()

    # Load data
    loader = ClinVarLoader(args.data_dir)
    data = loader.load(cancer_genes_only=args.cancer_only)

    print(loader.summary())

    # Create scorer
    scorer = create_scorer_from_project(smooth_kernel=args.smooth_kernel)

    # Score variants
    scored_df = scorer.score_variants(data)

    print(f"\nScored {scored_df['ES_score'].notna().sum()} variants")

    if args.output:
        scored_df.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}")
    else:
        print("\nSample of scored variants:")
        print(scored_df[scored_df["ES_score"].notna()].head(10))
