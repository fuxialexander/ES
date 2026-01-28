"""
ProteinGym Benchmark Module

Automated evaluation of ES scores against the ProteinGym DMS benchmark.

Usage:
    from benchmark.proteingym import ProteinGymLoader, ESScorer, evaluate_benchmark

    # Load data
    loader = ProteinGymLoader("./data")

    # Score variants
    scorer = ESScorer(plddt_file, uniprot_mapping)
    scored = scorer.score_all_assays(loader)

    # Evaluate
    results = evaluate_benchmark(scored)
"""

from .proteingym_loader import (
    ProteinGymLoader,
    DMSAssay,
    parse_mutation,
    is_single_mutation,
    get_mutation_positions,
    filter_single_mutations,
    get_assay_statistics
)

from .es_scorer import (
    ESScorer,
    create_scorer_from_project
)

from .evaluate import (
    EvaluationResult,
    BenchmarkResults,
    evaluate_assay,
    evaluate_benchmark,
    results_to_dataframe,
    plot_benchmark_results,
    print_summary,
    compute_spearman,
    compute_auc,
    compute_mcc,
    compute_ndcg,
    compute_top_k_recall
)

__all__ = [
    # Loader
    "ProteinGymLoader",
    "DMSAssay",
    "parse_mutation",
    "is_single_mutation",
    "get_mutation_positions",
    "filter_single_mutations",
    "get_assay_statistics",
    # Scorer
    "ESScorer",
    "create_scorer_from_project",
    # Evaluation
    "EvaluationResult",
    "BenchmarkResults",
    "evaluate_assay",
    "evaluate_benchmark",
    "results_to_dataframe",
    "plot_benchmark_results",
    "print_summary",
    "compute_spearman",
    "compute_auc",
    "compute_mcc",
    "compute_ndcg",
    "compute_top_k_recall",
]

__version__ = "1.0.0"
