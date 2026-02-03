#!/usr/bin/env python3
"""
ClinVar Data Loader

Loads and parses ClinVar pathogenic/benign variant data for benchmark evaluation.

Supports two data formats:
1. Preprocessed ClinVar from download_data.py
2. ProteinGym clinical_substitutions format

Provides utilities for filtering by:
- Review status (1-4 stars)
- Gene sets (cancer-relevant genes)
- Variant counts per gene
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClinVarVariant:
    """Represents a single ClinVar variant"""
    gene: str
    mutation: str  # e.g., "V600E"
    position: int
    wt_aa: str
    mt_aa: str
    label: int  # 1 = pathogenic, 0 = benign
    clinical_significance: str
    review_status: Optional[str] = None
    variation_id: Optional[str] = None


@dataclass
class ClinVarData:
    """Container for ClinVar benchmark data"""
    variants: pd.DataFrame
    n_pathogenic: int
    n_benign: int
    genes: List[str]
    source: str  # 'clinvar' or 'proteingym'

    def get_variants_for_gene(self, gene: str) -> pd.DataFrame:
        """Get all variants for a specific gene"""
        gene_col = self._get_gene_column()
        return self.variants[self.variants[gene_col] == gene].copy()

    def get_pathogenic_variants(self) -> pd.DataFrame:
        """Get all pathogenic variants"""
        label_col = self._get_label_column()
        return self.variants[self.variants[label_col] == 1].copy()

    def get_benign_variants(self) -> pd.DataFrame:
        """Get all benign variants"""
        label_col = self._get_label_column()
        return self.variants[self.variants[label_col] == 0].copy()

    def _get_gene_column(self) -> str:
        """Get the gene column name based on data format"""
        for col in ['GeneSymbol', 'gene', 'Gene', 'GENE']:
            if col in self.variants.columns:
                return col
        raise ValueError("No gene column found")

    def _get_label_column(self) -> str:
        """Get the label column name"""
        for col in ['Label', 'label', 'ClinVar_labels']:
            if col in self.variants.columns:
                return col
        raise ValueError("No label column found")


# Cancer-relevant genes (COSMIC Cancer Gene Census + common driver genes)
CANCER_GENES = [
    # Major oncogenes
    'EGFR', 'BRAF', 'KRAS', 'NRAS', 'PIK3CA', 'MET', 'ALK', 'RET', 'ROS1',
    'ERBB2', 'FGFR1', 'FGFR2', 'FGFR3', 'KIT', 'PDGFRA',
    # Tumor suppressors
    'TP53', 'APC', 'PTEN', 'BRCA1', 'BRCA2', 'RB1', 'NF1', 'NF2',
    'STK11', 'VHL', 'WT1', 'SMAD4', 'CDKN2A',
    # DNA repair
    'ATM', 'ATR', 'CHEK1', 'CHEK2', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
    # Signaling
    'JAK2', 'STAT3', 'MTOR', 'AKT1', 'MAP2K1', 'MAP2K2', 'RAF1',
    # Epigenetic regulators
    'DNMT3A', 'TET2', 'IDH1', 'IDH2', 'EZH2', 'ARID1A', 'ARID1B',
    # Transcription factors
    'MYC', 'MYCN', 'RUNX1', 'NOTCH1', 'NOTCH2', 'GATA3', 'FOXA1',
    # Others
    'ABL1', 'BCR', 'NPM1', 'FLT3', 'KMT2A', 'SETD2', 'SF3B1', 'SRSF2',
    'U2AF1', 'ASXL1', 'BCOR', 'CALR', 'CBL', 'CEBPA', 'CSF3R',
    'DDX3X', 'DNMT3B', 'EP300', 'CREBBP', 'FBXW7', 'GNAS',
    'HIST1H3B', 'HIST1H3C', 'KDM6A', 'KEAP1', 'LATS1', 'LATS2',
    'MAX', 'MEN1', 'NSD1', 'PTPN11', 'RAD21', 'SMARCA4', 'SMARCB1',
    'SMO', 'SPOP', 'STAG2', 'SUZ12', 'TERT', 'TSC1', 'TSC2',
]


class ClinVarLoader:
    """Load and manage ClinVar benchmark data"""

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize ClinVar loader.

        Args:
            data_dir: Path to directory containing ClinVar data files
        """
        self.data_dir = Path(data_dir)
        self._data: Optional[ClinVarData] = None

    @property
    def clinvar_file(self) -> Path:
        """Path to preprocessed ClinVar file"""
        return self.data_dir / "clinvar_pathogenic_benign.csv"

    @property
    def proteingym_file(self) -> Path:
        """Path to ProteinGym clinical file"""
        return self.data_dir / "proteingym_clinical_substitutions.csv"

    def load(
        self,
        source: str = 'auto',
        min_variants_per_gene: int = 0,
        cancer_genes_only: bool = False,
        gene_list: Optional[List[str]] = None
    ) -> ClinVarData:
        """
        Load ClinVar data.

        Args:
            source: 'clinvar', 'proteingym', or 'auto' (try both)
            min_variants_per_gene: Filter genes with fewer variants
            cancer_genes_only: Only include CANCER_GENES
            gene_list: Custom list of genes to include

        Returns:
            ClinVarData object
        """
        if source == 'auto':
            if self.clinvar_file.exists():
                source = 'clinvar'
            elif self.proteingym_file.exists():
                source = 'proteingym'
            else:
                raise FileNotFoundError(
                    f"No ClinVar data found in {self.data_dir}"
                )

        if source == 'clinvar':
            df = self._load_clinvar()
        else:
            df = self._load_proteingym()

        # Get gene and label columns
        gene_col = self._find_column(df, ['GeneSymbol', 'gene', 'Gene'])
        label_col = self._find_column(df, ['Label', 'label', 'ClinVar_labels'])

        # Apply filters
        if cancer_genes_only:
            df = df[df[gene_col].isin(CANCER_GENES)]

        if gene_list:
            df = df[df[gene_col].isin(gene_list)]

        if min_variants_per_gene > 0:
            gene_counts = df[gene_col].value_counts()
            valid_genes = gene_counts[gene_counts >= min_variants_per_gene].index
            df = df[df[gene_col].isin(valid_genes)]

        self._data = ClinVarData(
            variants=df,
            n_pathogenic=int((df[label_col] == 1).sum()),
            n_benign=int((df[label_col] == 0).sum()),
            genes=df[gene_col].unique().tolist(),
            source=source
        )

        return self._data

    def _load_clinvar(self) -> pd.DataFrame:
        """Load preprocessed ClinVar data"""
        if not self.clinvar_file.exists():
            raise FileNotFoundError(f"ClinVar file not found: {self.clinvar_file}")

        df = pd.read_csv(self.clinvar_file)

        # Ensure required columns exist
        required = ['GeneSymbol', 'Mutation', 'position', 'Label']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    def _load_proteingym(self) -> pd.DataFrame:
        """Load ProteinGym clinical data"""
        if not self.proteingym_file.exists():
            raise FileNotFoundError(
                f"ProteinGym clinical file not found: {self.proteingym_file}"
            )

        df = pd.read_csv(self.proteingym_file)

        # ProteinGym format has different column names
        # Standardize to our format
        col_mapping = {
            'UniProt_ID': 'UniProt_ID',
            'gene_name': 'GeneSymbol',
            'mutant': 'Mutation',
            'ClinVar_labels': 'Label'
        }

        for old, new in col_mapping.items():
            if old in df.columns and new != old:
                df[new] = df[old]

        # Parse position from mutation string
        if 'Mutation' in df.columns and 'position' not in df.columns:
            df['wt_aa'], df['position'], df['mt_aa'] = zip(
                *df['Mutation'].apply(self._parse_mutation)
            )

        return df

    @staticmethod
    def _parse_mutation(mutation_str: str) -> Tuple[str, int, str]:
        """Parse mutation string like 'V600E' into (wt_aa, position, mt_aa)"""
        if pd.isna(mutation_str) or not mutation_str:
            return '', 0, ''

        mutation_str = str(mutation_str)

        # Handle multi-mutations (take first)
        if ':' in mutation_str:
            mutation_str = mutation_str.split(':')[0]

        # Pattern: letter(s) + number + letter(s)
        match = re.match(r'^([A-Za-z]+)(\d+)([A-Za-z\*]+)$', mutation_str)
        if match:
            return match.group(1).upper(), int(match.group(2)), match.group(3).upper()

        return '', 0, ''

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        """Find first matching column from candidates"""
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"No matching column found from: {candidates}")

    @property
    def data(self) -> ClinVarData:
        """Get loaded data"""
        if self._data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._data

    def get_gene_statistics(self) -> pd.DataFrame:
        """Get per-gene statistics"""
        df = self.data.variants
        gene_col = self.data._get_gene_column()
        label_col = self.data._get_label_column()

        stats = df.groupby(gene_col).agg({
            label_col: ['count', 'sum', 'mean']
        }).reset_index()

        stats.columns = ['Gene', 'Total', 'Pathogenic', 'PathogenicFrac']
        stats['Benign'] = stats['Total'] - stats['Pathogenic']
        stats = stats.sort_values('Total', ascending=False)

        return stats

    def summary(self) -> str:
        """Get summary of loaded data"""
        if self._data is None:
            return "No data loaded"

        lines = [
            "ClinVar Benchmark Data Summary",
            "="*50,
            f"Source: {self._data.source}",
            f"Total variants: {len(self._data.variants):,}",
            f"  Pathogenic: {self._data.n_pathogenic:,}",
            f"  Benign: {self._data.n_benign:,}",
            f"  Class ratio: {self._data.n_pathogenic / self._data.n_benign:.2f}:1",
            f"Unique genes: {len(self._data.genes):,}",
        ]

        # Top genes by variant count
        gene_col = self._data._get_gene_column()
        top_genes = self._data.variants[gene_col].value_counts().head(10)
        lines.append("\nTop 10 genes by variant count:")
        for gene, count in top_genes.items():
            lines.append(f"  {gene}: {count:,}")

        return "\n".join(lines)


def parse_mutation(mutation_str: str) -> Tuple[str, int, str]:
    """
    Parse a mutation string into components.

    Args:
        mutation_str: Mutation string like "V600E" or "A1P:D2N"

    Returns:
        Tuple of (wild_type_aa, position, mutant_aa) for first mutation
    """
    return ClinVarLoader._parse_mutation(mutation_str)


def is_single_mutation(mutation_str: str) -> bool:
    """Check if mutation string represents a single substitution"""
    return ':' not in str(mutation_str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClinVar data loader")
    parser.add_argument("data_dir", type=str, help="Directory with ClinVar data")
    parser.add_argument("--cancer-only", action="store_true",
                        help="Filter to cancer genes only")
    parser.add_argument("--min-variants", type=int, default=0,
                        help="Minimum variants per gene")
    parser.add_argument("--stats", action="store_true",
                        help="Show per-gene statistics")

    args = parser.parse_args()

    loader = ClinVarLoader(args.data_dir)
    data = loader.load(
        cancer_genes_only=args.cancer_only,
        min_variants_per_gene=args.min_variants
    )

    print(loader.summary())

    if args.stats:
        print("\n\nPer-gene statistics:")
        print(loader.get_gene_statistics().head(20).to_string())
