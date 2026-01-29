"""
Variant Annotation Benchmark

Evaluates ES Score against real-world clinical outcomes from the MSK-IMPACT NSCLC cohort.
Based on the variant-annotation repository: https://github.com/clinical-data-mining/variant-annotation

Reference: Nature Communications (2025) - Validating machine learning cancer driver mutation
predictions against real-world clinical data.
"""

from .data_loader import (
    VariantAnnotationLoader,
    MutationData,
    ClinicalData,
    load_mutation_data,
    load_clinical_data
)
from .es_scorer import VariantAnnotationScorer, score_mutations
from .evaluate import (
    SurvivalEvaluator,
    EvaluationResult,
    BenchmarkResults,
    evaluate_predictor,
    compare_predictors
)

__all__ = [
    "VariantAnnotationLoader",
    "MutationData",
    "ClinicalData",
    "load_mutation_data",
    "load_clinical_data",
    "VariantAnnotationScorer",
    "score_mutations",
    "SurvivalEvaluator",
    "EvaluationResult",
    "BenchmarkResults",
    "evaluate_predictor",
    "compare_predictors"
]
