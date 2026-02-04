#!/usr/bin/env python3
"""
ClinVar Pathogenic Variant Benchmark Runner

Automated pipeline for evaluating ES Score against ClinVar pathogenic/benign variants.

This benchmark uses ClinVar variants as a more validated benchmark than the
Bailey et al. analysis, as suggested by Reviewer 2.

Key features:
- Downloads and preprocesses ClinVar variant data
- Scores variants with ES Score algorithm
- Evaluates classification performance (AUC-ROC, AUC-PR, MCC, etc.)
- Compares with ProteinGym clinical benchmark methodology

Usage:
    # Full pipeline (download, score, evaluate)
    python run_benchmark.py --full

    # Just download data
    python run_benchmark.py --download

    # Score and evaluate (data already downloaded)
    python run_benchmark.py --data_dir ./data --output_dir ./results

    # Cancer genes only
    python run_benchmark.py --full --cancer-genes-only

    # Compare with ProteinGym clinical data
    python run_benchmark.py --full --source proteingym
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from download_data import download_all, verify_download, print_verification
from clinvar_loader import ClinVarLoader, CANCER_GENES
from es_scorer import ClinVarScorer, create_scorer_from_project
from evaluate import (
    evaluate_classification,
    BenchmarkResults,
    print_summary,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_score_distributions
)


def run_download(
    output_dir: Path,
    source: str = 'summary',
    min_review_stars: int = 1
) -> bool:
    """Download ClinVar data"""
    print("\n" + "="*60)
    print("Step 1: Downloading ClinVar Data")
    print("="*60)

    success = download_all(output_dir, source=source, min_review_stars=min_review_stars)

    # Verify
    results = verify_download(output_dir)
    print_verification(results)

    return success


def run_scoring(
    data_dir: Path,
    output_dir: Path,
    smooth_kernel: int = 10,
    use_3d: bool = False,
    cancer_genes_only: bool = False,
    min_variants_per_gene: int = 0,
    source: str = 'auto'
) -> pd.DataFrame:
    """Score ClinVar variants with ES Score"""
    print("\n" + "="*60)
    print("Step 2: Computing ES Scores")
    print("="*60)

    # Load data
    loader = ClinVarLoader(data_dir)
    data = loader.load(
        source=source,
        cancer_genes_only=cancer_genes_only,
        min_variants_per_gene=min_variants_per_gene
    )

    print(loader.summary())

    # Create scorer
    try:
        scorer = create_scorer_from_project(
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )
    except Exception as e:
        print(f"Error creating scorer: {e}")
        print("Attempting with minimal configuration...")
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        scorer = ClinVarScorer(
            plddt_file=PROJECT_ROOT / "plddt" / "9606.pLDDT.tdt",
            uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )

    # Score variants
    scored_df = scorer.score_variants(data)

    # Save scored data
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_variants.csv"
    scored_df.to_csv(scored_path, index=False)
    print(f"\nSaved scored variants to: {scored_path}")

    return scored_df


def run_evaluation(
    output_dir: Path,
    scored_df: Optional[pd.DataFrame] = None,
    label_col: str = 'Label',
    gene_col: str = 'GeneSymbol'
) -> BenchmarkResults:
    """Evaluate ES Score against ClinVar labels"""
    print("\n" + "="*60)
    print("Step 3: Evaluating Classification Performance")
    print("="*60)

    # Load scored data if not provided
    if scored_df is None:
        scored_path = output_dir / "scored_variants.csv"
        if scored_path.exists():
            scored_df = pd.read_csv(scored_path)
        else:
            raise ValueError("No scored data found. Run scoring first.")

    # Prepare predictor columns
    predictor_cols = ['ES_score']
    predictor_names = ['ES Score']

    # Add other available predictors if present
    other_predictors = {
        'EVE_scores': 'EVE',
        'esm_score': 'ESM',
        'esm': 'ESM',
        'AlphaMissense_score': 'AlphaMissense',
        'REVEL_score': 'REVEL',
        'CADD_score': 'CADD'
    }

    for col, name in other_predictors.items():
        if col in scored_df.columns:
            predictor_cols.append(col)
            predictor_names.append(name)

    print(f"\nEvaluating {len(predictor_cols)} predictors:")
    for name in predictor_names:
        print(f"  - {name}")

    # Run evaluation
    results = evaluate_classification(
        scored_df,
        predictor_cols,
        predictor_names,
        label_col=label_col,
        gene_col=gene_col
    )

    # Print summary
    print_summary(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed results CSV
    results_df = results.to_dataframe()
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_variants_total": results.n_variants_total,
        "n_genes_total": results.n_genes_total,
        "best_by_auc_roc": results.best_by_auc_roc,
        "best_by_auc_pr": results.best_by_auc_pr,
        "best_by_mcc": results.best_by_mcc,
        "predictors": {
            name: {
                "n_variants": res.n_variants,
                "n_pathogenic": res.n_pathogenic,
                "n_benign": res.n_benign,
                "auc_roc": res.auc_roc,
                "auc_pr": res.auc_pr,
                "mcc": res.mcc
            }
            for name, res in results.predictor_results.items()
        }
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Visualizations
    try:
        plot_roc_curves(
            scored_df, predictor_cols, predictor_names,
            label_col=label_col,
            output_path=str(output_dir / "roc_curves.png")
        )
        plot_precision_recall_curves(
            scored_df, predictor_cols, predictor_names,
            label_col=label_col,
            output_path=str(output_dir / "pr_curves.png")
        )
        if 'ES_score' in scored_df.columns:
            plot_score_distributions(
                scored_df, 'ES_score', label_col,
                output_path=str(output_dir / "es_score_distribution.png")
            )
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")

    return results


def compare_with_proteingym(
    results: BenchmarkResults,
    proteingym_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare ES Score results with ProteinGym clinical benchmark.

    Note: ProteinGym reports average per-protein AUC-ROC, which may differ
    from our variant-level AUC.

    Args:
        results: Our benchmark results
        proteingym_path: Path to ProteinGym results file

    Returns:
        Comparison DataFrame
    """
    print("\n" + "="*60)
    print("Comparison with ProteinGym Clinical Benchmark")
    print("="*60)

    comparisons = []

    # Our results
    for name, res in results.predictor_results.items():
        comparisons.append({
            "Method": name,
            "Source": "This study",
            "AUC-ROC": res.auc_roc,
            "AUC-PR": res.auc_pr,
            "N Variants": res.n_variants
        })

    # Known ProteinGym clinical benchmark values (from leaderboard)
    # These are approximate reference values for comparison
    proteingym_baselines = {
        "ESM-1v (zero-shot)": 0.73,
        "EVE": 0.78,
        "Tranception L": 0.79,
        "VESPA": 0.76,
        "AlphaMissense": 0.82,
        "GEMME": 0.74,
    }

    for method, auc in proteingym_baselines.items():
        comparisons.append({
            "Method": f"{method} (ProteinGym ref)",
            "Source": "ProteinGym clinical",
            "AUC-ROC": auc,
            "AUC-PR": None,
            "N Variants": "~500K"
        })

    comparison_df = pd.DataFrame(comparisons)
    comparison_df = comparison_df.sort_values("AUC-ROC", ascending=False)

    print("\nMethod Comparison (sorted by AUC-ROC):")
    print(comparison_df.to_string(index=False))

    return comparison_df


def run_full_pipeline(
    output_dir: Path,
    smooth_kernel: int = 10,
    use_3d: bool = False,
    skip_download: bool = False,
    cancer_genes_only: bool = False,
    min_variants_per_gene: int = 0,
    source: str = 'summary',
    min_review_stars: int = 1
):
    """Run the complete benchmark pipeline"""
    print("\n" + "="*60)
    print("ClinVar Pathogenic Variant Benchmark Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    # Step 1: Download
    if not skip_download:
        success = run_download(data_dir, source=source, min_review_stars=min_review_stars)
        if not success:
            print("Download failed. Please check errors above.")
            return None

    # Step 2: Score
    scored_df = run_scoring(
        data_dir,
        results_dir,
        smooth_kernel=smooth_kernel,
        use_3d=use_3d,
        cancer_genes_only=cancer_genes_only,
        min_variants_per_gene=min_variants_per_gene,
        source='auto'
    )

    if scored_df is None or len(scored_df) == 0:
        print("Scoring failed. No variants could be scored.")
        return None

    # Step 3: Evaluate
    results = run_evaluation(results_dir, scored_df=scored_df)

    # Step 4: Compare with ProteinGym
    comparison = compare_with_proteingym(results)
    comparison.to_csv(results_dir / "proteingym_comparison.csv", index=False)

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print(f"Results saved to: {results_dir}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ClinVar pathogenic variant benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with NCBI ClinVar data
    python run_benchmark.py --full --output_dir ./clinvar_benchmark

    # Use ProteinGym pre-processed clinical data
    python run_benchmark.py --full --source proteingym

    # Cancer genes only (COSMIC Cancer Gene Census)
    python run_benchmark.py --full --cancer-genes-only

    # Filter by minimum review status (2+ stars)
    python run_benchmark.py --full --min-stars 2

    # Just download data
    python run_benchmark.py --download --output_dir ./clinvar_benchmark

    # Score and evaluate existing data
    python run_benchmark.py --data_dir ./clinvar_benchmark/data --output_dir ./clinvar_benchmark/results

Notes:
    This benchmark implements Reviewer 2's suggestion to use ClinVar pathogenic
    variants as a more validated benchmark than the Bailey et al. analysis.

    Data sources:
    - 'summary': NCBI ClinVar variant_summary.txt (comprehensive, updated weekly)
    - 'proteingym': ProteinGym clinical_substitutions (curated, ~500K variants)

    Key differences from vanilla ClinVar:
    - Only pathogenic and benign variants (no VUS)
    - Optional filtering by review status (1-4 stars)
    - Focuses on missense variants for protein-level analysis
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (download, score, evaluate)"
    )
    mode_group.add_argument(
        "--download",
        action="store_true",
        help="Only download ClinVar data"
    )
    mode_group.add_argument(
        "--score",
        action="store_true",
        help="Only score variants (requires downloaded data)"
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Only evaluate (requires scored data)"
    )

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./clinvar_benchmark",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory with downloaded data (for --score/--evaluate)"
    )
    parser.add_argument(
        "--scored_file",
        type=str,
        help="CSV file with scored variants (for --evaluate)"
    )

    # Data options
    parser.add_argument(
        "--source",
        type=str,
        choices=['summary', 'proteingym', 'both'],
        default='summary',
        help="Data source: 'summary' (NCBI ClinVar), 'proteingym', or 'both'"
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Minimum review status (0-4 stars)"
    )
    parser.add_argument(
        "--cancer-genes-only",
        action="store_true",
        help="Filter to COSMIC Cancer Gene Census genes"
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=0,
        help="Minimum variants per gene"
    )

    # ES Score options
    parser.add_argument(
        "--smooth_kernel",
        type=int,
        default=10,
        help="Smoothing kernel size for pLDDT"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D spatial averaging in ES score"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download step in --full mode"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.full:
        run_full_pipeline(
            output_dir,
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d,
            skip_download=args.skip_download,
            cancer_genes_only=args.cancer_genes_only,
            min_variants_per_gene=args.min_variants,
            source=args.source,
            min_review_stars=args.min_stars
        )

    elif args.download:
        data_dir = output_dir / "data"
        run_download(data_dir, source=args.source, min_review_stars=args.min_stars)

    elif args.score:
        data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
        results_dir = output_dir / "results"
        run_scoring(
            data_dir,
            results_dir,
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d,
            cancer_genes_only=args.cancer_genes_only,
            min_variants_per_gene=args.min_variants
        )

    elif args.evaluate:
        results_dir = output_dir / "results"

        scored_df = None
        if args.scored_file:
            scored_df = pd.read_csv(args.scored_file)

        run_evaluation(results_dir, scored_df=scored_df)


if __name__ == "__main__":
    main()
