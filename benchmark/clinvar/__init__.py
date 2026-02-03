#!/usr/bin/env python3
"""
ClinVar Pathogenic Variant Benchmark

Benchmark ES Score against ClinVar pathogenic/benign variant classifications.
This provides a more curated and validated benchmark than the Bailey et al. analysis,
as suggested by Reviewer 2.

Key differences from vanilla ClinVar:
- Only uses variants with confident pathogenic or benign classifications
- Excludes variants of uncertain significance (VUS)
- Focuses on missense variants for protein-level analysis
- Optional filtering by review status (1-4 stars)

Based on ProteinGym's clinical_substitutions benchmark methodology.
Reference: https://proteingym.org/
"""

from .clinvar_loader import ClinVarLoader, ClinVarVariant, ClinVarData
from .es_scorer import ClinVarScorer, create_scorer_from_project
from .evaluate import (
    evaluate_classification,
    ClassificationResult,
    BenchmarkResults,
    print_summary,
    plot_roc_curves
)

__all__ = [
    'ClinVarLoader',
    'ClinVarVariant',
    'ClinVarData',
    'ClinVarScorer',
    'create_scorer_from_project',
    'evaluate_classification',
    'ClassificationResult',
    'BenchmarkResults',
    'print_summary',
    'plot_roc_curves'
]
