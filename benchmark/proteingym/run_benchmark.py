#!/usr/bin/env python3
"""
ProteinGym Benchmark Runner

Automated pipeline for evaluating ES scores against ProteinGym DMS benchmarks.
Supports comparison with AlphaMissense predictions.

Usage:
    # Full pipeline (download, score, evaluate)
    python run_benchmark.py --full

    # Full pipeline with AlphaMissense comparison
    python run_benchmark.py --full --include_alphamissense

    # Just download data
    python run_benchmark.py --download

    # Score and evaluate (data already downloaded)
    python run_benchmark.py --data_dir ./data --output_dir ./results

    # Quick test with limited assays
    python run_benchmark.py --full --max_assays 10
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

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from download_data import download_dataset, DATASETS
from proteingym_loader import ProteinGymLoader, get_assay_statistics
from es_scorer import ESScorer, create_scorer_from_project
from alphamissense_scorer import AlphaMissenseScorer, create_alphamissense_scorer
from evaluate import (
    evaluate_benchmark,
    results_to_dataframe,
    plot_benchmark_results,
    print_summary,
    BenchmarkResults
)


def run_download(output_dir: Path, datasets: List[str] = None) -> bool:
    """Download ProteinGym data"""
    print("\n" + "="*60)
    print("Step 1: Downloading ProteinGym Data")
    print("="*60)

    if datasets is None:
        datasets = ["substitutions", "reference"]

    for dataset in datasets:
        success = download_dataset(dataset, output_dir)
        if not success:
            print(f"Failed to download {dataset}")
            return False

    return True


def run_scoring(
    data_dir: Path,
    output_dir: Path,
    max_assays: Optional[int] = None,
    single_only: bool = True,
    smooth_kernel: int = 10,
    use_3d: bool = False
) -> Dict[str, pd.DataFrame]:
    """Score all assays with ES scores"""
    print("\n" + "="*60)
    print("Step 2: Computing ES Scores")
    print("="*60)

    # Create loader
    loader = ProteinGymLoader(data_dir)
    assay_ids = loader.list_assays()

    if max_assays:
        assay_ids = assay_ids[:max_assays]

    print(f"Found {len(assay_ids)} assays to process")

    # Create scorer
    try:
        scorer = create_scorer_from_project(
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )
    except Exception as e:
        print(f"Error creating scorer: {e}")
        print("Trying with minimal configuration...")
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        scorer = ESScorer(
            plddt_file=PROJECT_ROOT / "plddt" / "9606.pLDDT.tdt",
            uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )

    # Score all assays
    scored_assays = scorer.score_all_assays(
        loader,
        assay_ids=assay_ids,
        single_only=single_only
    )

    print(f"Successfully scored {len(scored_assays)} assays")

    # Save intermediate results
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_variants.csv"

    all_scored = pd.concat(
        [df.assign(assay_id=aid) for aid, df in scored_assays.items()],
        ignore_index=True
    )
    all_scored.to_csv(scored_path, index=False)
    print(f"Saved scored variants to: {scored_path}")

    return scored_assays


def run_alphamissense_scoring(
    data_dir: Path,
    output_dir: Path,
    max_assays: Optional[int] = None,
    single_only: bool = True,
    am_data_dir: str = "/mnt/storage/alphamissense",
) -> Dict[str, pd.DataFrame]:
    """Score all assays with AlphaMissense predictions"""
    print("\n" + "="*60)
    print("Computing AlphaMissense Scores")
    print("="*60)

    # Create loader
    loader = ProteinGymLoader(data_dir)
    assay_ids = loader.list_assays()

    if max_assays:
        assay_ids = assay_ids[:max_assays]

    print(f"Found {len(assay_ids)} assays to process")

    # Create AlphaMissense scorer
    try:
        scorer = create_alphamissense_scorer(data_dir=am_data_dir)
    except Exception as e:
        print(f"Error creating AlphaMissense scorer: {e}")
        print("Trying with default configuration...")
        scorer = AlphaMissenseScorer(data_dir=am_data_dir)

    # Check if bulk data is available
    if not scorer.has_bulk_data():
        print("AlphaMissense bulk data not found.")
        print(f"Please download it to: {am_data_dir}")
        print("Run: python -m benchmark.alphamissense.fetcher --download --data_dir " + am_data_dir)
        return {}

    # Score all assays
    scored_assays = scorer.score_all_assays(
        loader,
        assay_ids=assay_ids,
        single_only=single_only
    )

    print(f"Successfully scored {len(scored_assays)} assays with AlphaMissense")

    # Save intermediate results
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "alphamissense_scored_variants.csv"

    if scored_assays:
        all_scored = pd.concat(
            [df.assign(assay_id=aid) for aid, df in scored_assays.items()],
            ignore_index=True
        )
        all_scored.to_csv(scored_path, index=False)
        print(f"Saved AlphaMissense scored variants to: {scored_path}")

    return scored_assays


def run_evaluation(
    scored_assays: Dict[str, pd.DataFrame],
    output_dir: Path,
    method_name: str = "ES Score"
) -> BenchmarkResults:
    """Evaluate ES scores against DMS ground truth"""
    print("\n" + "="*60)
    print("Step 3: Evaluating Performance")
    print("="*60)

    # Run evaluation
    results = evaluate_benchmark(scored_assays, method_name=method_name)

    # Print summary
    print_summary(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed results CSV
    results_df = results_to_dataframe(results)
    results_df.to_csv(output_dir / "detailed_results.csv", index=False)

    # Summary JSON
    summary = {
        "method": results.method_name,
        "timestamp": datetime.now().isoformat(),
        "n_assays_evaluated": results.n_assays_evaluated,
        "n_assays_failed": results.n_assays_failed,
        "mean_spearman": results.mean_spearman,
        "std_spearman": results.std_spearman,
        "mean_auc": results.mean_auc,
        "std_auc": results.std_auc,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Visualization
    plot_benchmark_results(results, output_dir / "benchmark_results.png")

    return results


def compare_with_baselines(
    results: BenchmarkResults,
    alphamissense_results: Optional[BenchmarkResults] = None,
    baseline_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare ES Score results with ProteinGym baselines and AlphaMissense.

    Args:
        results: ES Score benchmark results
        alphamissense_results: AlphaMissense benchmark results (if computed)
        baseline_dir: Directory containing baseline results from ProteinGym

    Returns:
        Comparison DataFrame
    """
    print("\n" + "="*60)
    print("Comparison with Baselines")
    print("="*60)

    comparisons = [{
        "Method": results.method_name,
        "Mean Spearman": results.mean_spearman,
        "Std Spearman": results.std_spearman,
        "Mean AUC": results.mean_auc,
        "N Assays": results.n_assays_evaluated
    }]

    # Add AlphaMissense results if computed
    if alphamissense_results is not None:
        comparisons.append({
            "Method": alphamissense_results.method_name,
            "Mean Spearman": alphamissense_results.mean_spearman,
            "Std Spearman": alphamissense_results.std_spearman,
            "Mean AUC": alphamissense_results.mean_auc,
            "N Assays": alphamissense_results.n_assays_evaluated
        })

    # Known baseline values from ProteinGym leaderboard (approximate)
    # These are reference values for comparison
    known_baselines = {
        "ESM-1v (zero-shot)": {"spearman": 0.42, "auc": None},
        "EVE": {"spearman": 0.45, "auc": None},
        "Tranception L": {"spearman": 0.46, "auc": None},
        "MSA Transformer": {"spearman": 0.43, "auc": None},
        "VESPA": {"spearman": 0.44, "auc": None},
        "AlphaMissense": {"spearman": 0.48, "auc": None},  # Reference from ProteinGym
    }

    for method, metrics in known_baselines.items():
        # Skip AlphaMissense reference if we computed it
        if method == "AlphaMissense" and alphamissense_results is not None:
            continue
        comparisons.append({
            "Method": method + " (reference)",
            "Mean Spearman": metrics["spearman"],
            "Std Spearman": None,
            "Mean AUC": metrics["auc"],
            "N Assays": "217"  # ProteinGym full benchmark
        })

    comparison_df = pd.DataFrame(comparisons)
    comparison_df = comparison_df.sort_values("Mean Spearman", ascending=False)

    print("\nMethod Comparison (sorted by Spearman):")
    print(comparison_df.to_string(index=False))

    return comparison_df


def run_full_pipeline(
    output_dir: Path,
    max_assays: Optional[int] = None,
    skip_download: bool = False,
    smooth_kernel: int = 10,
    use_3d: bool = False,
    include_alphamissense: bool = False,
    am_data_dir: str = "/mnt/storage/alphamissense",
):
    """Run the complete benchmark pipeline"""
    print("\n" + "="*60)
    print("ProteinGym Benchmark Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    if include_alphamissense:
        print("Including AlphaMissense comparison")
    print("="*60)

    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    # Step 1: Download
    if not skip_download:
        success = run_download(data_dir)
        if not success:
            print("Download failed. Exiting.")
            return None

    # Step 2: Score with ES Score
    scored_assays = run_scoring(
        data_dir,
        results_dir,
        max_assays=max_assays,
        smooth_kernel=smooth_kernel,
        use_3d=use_3d
    )

    if not scored_assays:
        print("No assays were scored. Exiting.")
        return None

    # Step 3: Evaluate ES Score
    results = run_evaluation(scored_assays, results_dir)

    # Step 4: AlphaMissense scoring and evaluation (optional)
    am_results = None
    if include_alphamissense:
        am_scored_assays = run_alphamissense_scoring(
            data_dir,
            results_dir,
            max_assays=max_assays,
            am_data_dir=am_data_dir,
        )

        if am_scored_assays:
            am_results_dir = results_dir / "alphamissense"
            am_results = run_evaluation(
                am_scored_assays,
                am_results_dir,
                method_name="AlphaMissense",
            )

    # Step 5: Compare
    comparison = compare_with_baselines(results, alphamissense_results=am_results)
    comparison.to_csv(results_dir / "comparison.csv", index=False)

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print(f"Results saved to: {results_dir}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ES Score benchmark against ProteinGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_benchmark.py --full --output_dir ./proteingym_benchmark

    # Quick test with 10 assays
    python run_benchmark.py --full --max_assays 10

    # Just download data
    python run_benchmark.py --download --output_dir ./proteingym_benchmark

    # Evaluate pre-scored data
    python run_benchmark.py --evaluate --scored_file ./results/scored_variants.csv
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
        help="Only download ProteinGym data"
    )
    mode_group.add_argument(
        "--score",
        action="store_true",
        help="Only compute ES scores (requires downloaded data)"
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Only evaluate pre-scored data"
    )

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./proteingym_benchmark",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory with downloaded ProteinGym data (for --score)"
    )
    parser.add_argument(
        "--scored_file",
        type=str,
        help="CSV file with scored variants (for --evaluate)"
    )

    # Options
    parser.add_argument(
        "--max_assays",
        type=int,
        help="Maximum number of assays to process (for testing)"
    )
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
        "--single_only",
        action="store_true",
        default=True,
        help="Only evaluate single-point mutations"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download step in --full mode"
    )
    parser.add_argument(
        "--include_alphamissense",
        action="store_true",
        help="Include AlphaMissense comparison in benchmark"
    )
    parser.add_argument(
        "--am_data_dir",
        type=str,
        default="/mnt/storage/alphamissense",
        help="Directory containing AlphaMissense bulk data"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.full:
        run_full_pipeline(
            output_dir,
            max_assays=args.max_assays,
            skip_download=args.skip_download,
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d,
            include_alphamissense=args.include_alphamissense,
            am_data_dir=args.am_data_dir,
        )

    elif args.download:
        data_dir = output_dir / "data"
        run_download(data_dir)

    elif args.score:
        data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
        results_dir = output_dir / "results"
        run_scoring(
            data_dir,
            results_dir,
            max_assays=args.max_assays,
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d
        )

    elif args.evaluate:
        if not args.scored_file:
            print("Error: --scored_file required for --evaluate mode")
            return

        scored_df = pd.read_csv(args.scored_file)
        if "assay_id" in scored_df.columns:
            scored_assays = {aid: group for aid, group in scored_df.groupby("assay_id")}
        else:
            scored_assays = {"unknown": scored_df}

        results_dir = output_dir / "results"
        run_evaluation(scored_assays, results_dir)


if __name__ == "__main__":
    main()
