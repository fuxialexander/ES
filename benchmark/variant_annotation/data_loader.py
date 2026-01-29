#!/usr/bin/env python3
"""
Data Loader for Variant Annotation Benchmark

Loads and processes MSK-IMPACT NSCLC clinical and mutation data.

Data includes:
- Clinical outcomes (survival status and duration)
- Missense mutations with VEP annotations from 13 predictors
- OncoKB and ClinVar classifications
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# VEP predictor columns in the mutation data
VEP_PREDICTORS = [
    "SIFT",
    "PolyPhen",
    "CADD",
    "REVEL",
    "MutationAssessor",
    "AlphaMissense",
    "VEST",
    "MetaLR",
    "MetaSVM",
    "FATHMM",
    "PROVEAN",
    "LRT",
    "MutationTaster"
]

# Columns with binary pathogenicity calls
PATHOGENICITY_COLS = [f"{vep}_Pathogenic" for vep in VEP_PREDICTORS]

# Columns with rescue reclassifications
RESCUE_COLS = [f"{vep}_Rescue" for vep in VEP_PREDICTORS]


@dataclass
class MutationData:
    """Container for mutation data."""
    data: pd.DataFrame
    n_patients: int
    n_mutations: int
    genes: List[str]
    vep_predictors: List[str]
    has_oncokb: bool
    has_clinvar: bool

    def get_mutations_for_gene(self, gene: str) -> pd.DataFrame:
        """Get all mutations for a specific gene."""
        return self.data[self.data["Hugo_Symbol"] == gene].copy()

    def get_pathogenic_mutations(self, predictor: str) -> pd.DataFrame:
        """Get mutations classified as pathogenic by a predictor."""
        col = f"{predictor}_Pathogenic"
        if col not in self.data.columns:
            raise ValueError(f"Predictor {predictor} not found")
        return self.data[self.data[col] == 1].copy()


@dataclass
class ClinicalData:
    """Container for clinical data."""
    data: pd.DataFrame
    n_patients: int
    n_events: int
    median_survival: float
    has_ancestry: bool
    has_smoking: bool

    def get_survival_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get survival times and event indicators."""
        times = self.data["OS_DURATION"].values
        events = self.data["OS_STATUS"].values
        return times, events


def parse_mutation_string(mutation_str: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a mutation string like 'V600E' into (wildtype, position, mutant).

    Args:
        mutation_str: Mutation string (e.g., 'V600E', 'p.V600E')

    Returns:
        Tuple of (wildtype_aa, position, mutant_aa) or (None, None, None) if parsing fails
    """
    if pd.isna(mutation_str) or not mutation_str:
        return None, None, None

    # Remove 'p.' prefix if present
    mutation_str = str(mutation_str).replace("p.", "")

    # Try to match pattern: letter(s) + number + letter(s)
    match = re.match(r'^([A-Za-z]+)(\d+)([A-Za-z\*]+)$', mutation_str)
    if match:
        wt = match.group(1).upper()
        pos = int(match.group(2))
        mt = match.group(3).upper()
        return wt, pos, mt

    return None, None, None


def load_clinical_data(filepath: Union[str, Path]) -> ClinicalData:
    """
    Load clinical data from CSV file.

    Args:
        filepath: Path to clinical CSV file

    Returns:
        ClinicalData object
    """
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required_cols = ["PATIENT_ID", "OS_STATUS", "OS_DURATION"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert OS_STATUS to binary if needed
    if df["OS_STATUS"].dtype == object:
        # Handle strings like "1:DECEASED" or "0:LIVING"
        df["OS_STATUS"] = df["OS_STATUS"].apply(
            lambda x: 1 if "DECEASED" in str(x) or str(x).startswith("1") else 0
        )

    # Calculate summary statistics
    n_patients = len(df)
    n_events = df["OS_STATUS"].sum()
    median_survival = df["OS_DURATION"].median()

    # Check for optional columns
    has_ancestry = "ancestry_label" in df.columns or "ADMIXTURE" in df.columns
    has_smoking = "SMOKING" in df.columns or "Smoking_History_Prediction_Aggregated" in df.columns

    return ClinicalData(
        data=df,
        n_patients=n_patients,
        n_events=n_events,
        median_survival=median_survival,
        has_ancestry=has_ancestry,
        has_smoking=has_smoking
    )


def load_mutation_data(filepath: Union[str, Path]) -> MutationData:
    """
    Load mutation data from CSV file.

    Args:
        filepath: Path to mutation CSV file

    Returns:
        MutationData object
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Identify available columns
    available_veps = []
    for vep in VEP_PREDICTORS:
        # Check for score column or pathogenic column
        score_col = vep
        path_col = f"{vep}_Pathogenic"
        if score_col in df.columns or path_col in df.columns:
            available_veps.append(vep)

    # Check for OncoKB and ClinVar
    has_oncokb = "oncogenic" in df.columns or "OncoKB" in df.columns
    has_clinvar = "ClinVar" in df.columns or "clinvar_clnsig" in df.columns

    # Get unique genes
    gene_col = None
    for col in ["Hugo_Symbol", "gene", "Gene", "GENE"]:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        genes = []
    else:
        # Standardize column name
        if gene_col != "Hugo_Symbol":
            df["Hugo_Symbol"] = df[gene_col]
        genes = df["Hugo_Symbol"].dropna().unique().tolist()

    # Get patient count
    patient_col = None
    for col in ["PATIENT_ID", "Tumor_Sample_Barcode", "SAMPLE_ID"]:
        if col in df.columns:
            patient_col = col
            break

    n_patients = df[patient_col].nunique() if patient_col else 0

    return MutationData(
        data=df,
        n_patients=n_patients,
        n_mutations=len(df),
        genes=genes,
        vep_predictors=available_veps,
        has_oncokb=has_oncokb,
        has_clinvar=has_clinvar
    )


class VariantAnnotationLoader:
    """
    Main loader class for the variant annotation benchmark.

    Handles loading and merging clinical and mutation data.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)

        # File paths
        self.clinical_file = self.data_dir / "msk_impact_nsclc_clinical.csv"
        self.mutations_file = self.data_dir / "msk_impact_nsclc_missense_mutations.csv"
        self.gam_file = self.data_dir / "msk_impact_gam.csv"

        # Loaded data
        self._clinical: Optional[ClinicalData] = None
        self._mutations: Optional[MutationData] = None
        self._gam: Optional[pd.DataFrame] = None
        self._merged: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load all data files."""
        if self.clinical_file.exists():
            self._clinical = load_clinical_data(self.clinical_file)
            print(f"Loaded clinical data: {self._clinical.n_patients} patients, "
                  f"{self._clinical.n_events} events")

        if self.mutations_file.exists():
            self._mutations = load_mutation_data(self.mutations_file)
            print(f"Loaded mutation data: {self._mutations.n_mutations} mutations, "
                  f"{len(self._mutations.genes)} genes")
            print(f"Available VEP predictors: {self._mutations.vep_predictors}")

        if self.gam_file.exists():
            self._gam = pd.read_csv(self.gam_file)
            print(f"Loaded gene alteration matrix: {self._gam.shape}")

    @property
    def clinical(self) -> ClinicalData:
        """Get clinical data."""
        if self._clinical is None:
            self._clinical = load_clinical_data(self.clinical_file)
        return self._clinical

    @property
    def mutations(self) -> MutationData:
        """Get mutation data."""
        if self._mutations is None:
            self._mutations = load_mutation_data(self.mutations_file)
        return self._mutations

    @property
    def gam(self) -> pd.DataFrame:
        """Get gene alteration matrix."""
        if self._gam is None:
            self._gam = pd.read_csv(self.gam_file)
        return self._gam

    def get_merged_data(self) -> pd.DataFrame:
        """
        Get merged clinical and mutation data.

        Returns:
            DataFrame with mutations joined to clinical data
        """
        if self._merged is not None:
            return self._merged

        clinical_df = self.clinical.data
        mutation_df = self.mutations.data

        # Find common ID column
        clinical_id = "PATIENT_ID" if "PATIENT_ID" in clinical_df.columns else None
        mutation_id = None
        for col in ["PATIENT_ID", "Tumor_Sample_Barcode", "SAMPLE_ID"]:
            if col in mutation_df.columns:
                mutation_id = col
                break

        if clinical_id is None or mutation_id is None:
            raise ValueError("Cannot find matching ID columns for merge")

        # Merge
        self._merged = mutation_df.merge(
            clinical_df[["PATIENT_ID", "OS_STATUS", "OS_DURATION"]],
            left_on=mutation_id,
            right_on="PATIENT_ID",
            how="left"
        )

        return self._merged

    def get_gene_mutations(self, gene: str) -> pd.DataFrame:
        """Get all mutations for a specific gene with clinical data."""
        merged = self.get_merged_data()
        return merged[merged["Hugo_Symbol"] == gene].copy()

    def get_available_predictors(self) -> List[str]:
        """Get list of available VEP predictors."""
        return self.mutations.vep_predictors

    def get_predictor_calls(self, predictor: str) -> pd.DataFrame:
        """
        Get pathogenicity calls for a specific predictor.

        Args:
            predictor: Name of the VEP predictor

        Returns:
            DataFrame with patient, mutation, and pathogenicity call
        """
        df = self.mutations.data.copy()

        # Find the pathogenic column
        path_col = f"{predictor}_Pathogenic"
        rescue_col = f"{predictor}_Rescue"

        if path_col not in df.columns and rescue_col not in df.columns:
            # Try to derive from score column
            if predictor in df.columns:
                # Apply threshold (different for each predictor)
                thresholds = {
                    "SIFT": 0.05,  # Below threshold is deleterious
                    "PolyPhen": 0.85,  # Above threshold is damaging
                    "CADD": 20,  # Above threshold is deleterious
                    "REVEL": 0.5,  # Above threshold is pathogenic
                    "AlphaMissense": 0.564,  # Above threshold is pathogenic
                    "VEST": 0.5,  # Above threshold is pathogenic
                    "MetaLR": 0.5,
                    "MetaSVM": 0,
                    "FATHMM": -1.5,  # Below threshold is deleterious
                }
                if predictor in thresholds:
                    threshold = thresholds[predictor]
                    if predictor in ["SIFT", "FATHMM"]:
                        df[path_col] = (df[predictor] < threshold).astype(int)
                    else:
                        df[path_col] = (df[predictor] > threshold).astype(int)

        return df

    def summary(self) -> str:
        """Get a summary of loaded data."""
        lines = ["Variant Annotation Benchmark Data Summary", "="*50]

        if self._clinical is not None:
            lines.append(f"\nClinical Data:")
            lines.append(f"  Patients: {self._clinical.n_patients:,}")
            lines.append(f"  Events (deaths): {self._clinical.n_events:,}")
            lines.append(f"  Median survival: {self._clinical.median_survival:.1f} days")

        if self._mutations is not None:
            lines.append(f"\nMutation Data:")
            lines.append(f"  Total mutations: {self._mutations.n_mutations:,}")
            lines.append(f"  Unique genes: {len(self._mutations.genes)}")
            lines.append(f"  VEP predictors: {len(self._mutations.vep_predictors)}")
            lines.append(f"    {', '.join(self._mutations.vep_predictors)}")
            lines.append(f"  OncoKB available: {self._mutations.has_oncokb}")
            lines.append(f"  ClinVar available: {self._mutations.has_clinvar}")

        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and explore variant annotation data")
    parser.add_argument("data_dir", type=str, help="Directory containing data files")
    parser.add_argument("--gene", type=str, help="Show mutations for specific gene")

    args = parser.parse_args()

    loader = VariantAnnotationLoader(args.data_dir)
    loader.load()

    print(loader.summary())

    if args.gene:
        print(f"\n\nMutations for {args.gene}:")
        gene_df = loader.get_gene_mutations(args.gene)
        print(f"  Total: {len(gene_df)}")
        if len(gene_df) > 0:
            print(gene_df.head(10))
