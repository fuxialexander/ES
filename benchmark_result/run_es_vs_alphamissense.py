#!/usr/bin/env python3
"""
ES Score vs AlphaMissense Benchmark on ProteinGym

This script runs a preliminary benchmark comparing ES Score with AlphaMissense
predictions on the ProteinGym DMS dataset.

Results are saved to the benchmark_result folder.

Usage:
    # Run full benchmark (may take time)
    python run_es_vs_alphamissense.py --full

    # Quick test with limited assays
    python run_es_vs_alphamissense.py --max_assays 50

    # Skip download if data already exists
    python run_es_vs_alphamissense.py --skip_download --max_assays 50
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark" / "proteingym"))

from benchmark.proteingym.run_benchmark import (
    run_download,
    run_evaluation,
    compare_with_baselines,
)
from benchmark.proteingym.proteingym_loader import ProteinGymLoader
from benchmark.proteingym.es_scorer import ESScorer, create_scorer_from_project
from benchmark.proteingym.alphamissense_scorer import AlphaMissenseScorer, create_alphamissense_scorer


def create_comparison_visualization(
    es_results_df: pd.DataFrame,
    am_results_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Create visualization comparing ES Score and AlphaMissense performance.

    Args:
        es_results_df: DataFrame with ES Score results per assay
        am_results_df: DataFrame with AlphaMissense results per assay
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping visualizations")
        return

    # Merge results on assay_id
    merged = es_results_df.merge(
        am_results_df,
        on="assay_id",
        suffixes=("_es", "_am"),
    )

    if len(merged) == 0:
        print("No overlapping assays for comparison visualization")
        return

    # Use the correct column names (spearman_rho instead of spearman)
    spearman_es_col = "spearman_rho_es"
    spearman_am_col = "spearman_rho_am"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot: ES Score vs AlphaMissense Spearman correlation
    ax1 = axes[0, 0]
    ax1.scatter(
        merged[spearman_es_col],
        merged[spearman_am_col],
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    # Add diagonal line
    lims = [
        min(merged[spearman_es_col].min(), merged[spearman_am_col].min()) - 0.05,
        max(merged[spearman_es_col].max(), merged[spearman_am_col].max()) + 0.05,
    ]
    ax1.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax1.set_xlabel("ES Score Spearman ρ")
    ax1.set_ylabel("AlphaMissense Spearman ρ")
    ax1.set_title("Per-Assay Spearman Correlation Comparison")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    # Add stats annotation
    es_wins = (merged[spearman_es_col] > merged[spearman_am_col]).sum()
    am_wins = (merged[spearman_am_col] > merged[spearman_es_col]).sum()
    ax1.text(
        0.05, 0.95,
        f"ES wins: {es_wins}\nAM wins: {am_wins}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 2. Distribution of Spearman correlations
    ax2 = axes[0, 1]
    data_to_plot = pd.DataFrame({
        "Spearman ρ": pd.concat([merged[spearman_es_col], merged[spearman_am_col]]),
        "Method": ["ES Score"] * len(merged) + ["AlphaMissense"] * len(merged),
    })
    sns.boxplot(data=data_to_plot, x="Method", y="Spearman ρ", ax=ax2)
    ax2.set_title("Distribution of Spearman Correlations")

    # Add mean annotations
    es_mean = merged[spearman_es_col].mean()
    am_mean = merged[spearman_am_col].mean()
    ax2.text(0, es_mean + 0.02, f"μ={es_mean:.3f}", ha="center", fontsize=9)
    ax2.text(1, am_mean + 0.02, f"μ={am_mean:.3f}", ha="center", fontsize=9)

    # 3. Histogram of performance difference
    ax3 = axes[1, 0]
    diff = merged[spearman_es_col] - merged[spearman_am_col]
    ax3.hist(diff, bins=30, edgecolor="black", alpha=0.7)
    ax3.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax3.axvline(diff.mean(), color="blue", linestyle="-", linewidth=2, label=f"Mean diff: {diff.mean():.3f}")
    ax3.set_xlabel("ES Score - AlphaMissense (Spearman ρ)")
    ax3.set_ylabel("Number of Assays")
    ax3.set_title("Performance Difference Distribution")
    ax3.legend()

    # 4. Comparison by assay type/protein family (if available)
    ax4 = axes[1, 1]
    # Top 10 best and worst performing assays for ES Score relative to AM
    merged_sorted = merged.sort_values(spearman_es_col, ascending=False).head(20)
    x = range(len(merged_sorted))
    width = 0.35
    ax4.bar([i - width/2 for i in x], merged_sorted[spearman_es_col], width, label="ES Score", color="steelblue")
    ax4.bar([i + width/2 for i in x], merged_sorted[spearman_am_col], width, label="AlphaMissense", color="coral")
    ax4.set_xlabel("Assay (sorted by ES Score performance)")
    ax4.set_ylabel("Spearman ρ")
    ax4.set_title("Top 20 Assays by ES Score Performance")
    ax4.legend()
    ax4.set_xticks([])  # Too many labels

    plt.tight_layout()
    plt.savefig(output_dir / "es_vs_alphamissense_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison visualization to: {output_dir / 'es_vs_alphamissense_comparison.png'}")


def run_scoring_human_only(
    data_dir: Path,
    output_dir: Path,
    max_assays: int = None,
    smooth_kernel: int = 10,
    use_3d: bool = False,
):
    """
    Score human-only assays with ES Score.

    The pLDDT data file only contains human proteins, so we filter for
    assays with 'HUMAN' in their ID.
    """
    print("\n" + "=" * 60)
    print("Computing ES Scores (Human proteins only)")
    print("=" * 60)

    # Create loader and filter for human assays
    loader = ProteinGymLoader(data_dir)
    all_assay_ids = loader.list_assays()
    human_assay_ids = [a for a in all_assay_ids if "HUMAN" in a]

    if max_assays:
        human_assay_ids = human_assay_ids[:max_assays]

    print(f"Total assays: {len(all_assay_ids)}")
    print(f"Human assays: {len(human_assay_ids)}")

    # Create scorer
    try:
        scorer = create_scorer_from_project(
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )
    except Exception as e:
        print(f"Error creating scorer: {e}")
        print("Trying with minimal configuration...")
        scorer = ESScorer(
            plddt_file=PROJECT_ROOT / "plddt" / "9606.pLDDT.tdt",
            uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )

    # Score human assays
    scored_assays = scorer.score_all_assays(
        loader,
        assay_ids=human_assay_ids,
        single_only=True
    )

    print(f"Successfully scored {len(scored_assays)} assays")

    # Save intermediate results
    output_dir.mkdir(parents=True, exist_ok=True)
    if scored_assays:
        all_scored = pd.concat(
            [df.assign(assay_id=aid) for aid, df in scored_assays.items()],
            ignore_index=True
        )
        scored_path = output_dir / "scored_variants.csv"
        all_scored.to_csv(scored_path, index=False)
        print(f"Saved scored variants to: {scored_path}")

    return scored_assays


def run_alphamissense_scoring_human_only(
    data_dir: Path,
    output_dir: Path,
    max_assays: int = None,
    am_data_dir: str = "/mnt/storage/alphamissense",
):
    """
    Score human-only assays with AlphaMissense.

    AlphaMissense data is also primarily for human proteins.
    """
    print("\n" + "=" * 60)
    print("Computing AlphaMissense Scores (Human proteins only)")
    print("=" * 60)

    # Create loader and filter for human assays
    loader = ProteinGymLoader(data_dir)
    all_assay_ids = loader.list_assays()
    human_assay_ids = [a for a in all_assay_ids if "HUMAN" in a]

    if max_assays:
        human_assay_ids = human_assay_ids[:max_assays]

    print(f"Scoring {len(human_assay_ids)} human assays with AlphaMissense")

    # Create AlphaMissense scorer
    try:
        scorer = create_alphamissense_scorer(data_dir=am_data_dir)
    except Exception as e:
        print(f"Error creating AlphaMissense scorer: {e}")
        scorer = AlphaMissenseScorer(data_dir=am_data_dir)

    # Check if bulk data is available
    if not scorer.has_bulk_data():
        print("AlphaMissense bulk data not found.")
        print(f"Please download it to: {am_data_dir}")
        return {}

    # Score all human assays
    scored_assays = scorer.score_all_assays(
        loader,
        assay_ids=human_assay_ids,
        single_only=True
    )

    print(f"Successfully scored {len(scored_assays)} assays with AlphaMissense")

    # Save intermediate results
    output_dir.mkdir(parents=True, exist_ok=True)
    if scored_assays:
        all_scored = pd.concat(
            [df.assign(assay_id=aid) for aid, df in scored_assays.items()],
            ignore_index=True
        )
        scored_path = output_dir / "alphamissense_scored_variants.csv"
        all_scored.to_csv(scored_path, index=False)
        print(f"Saved AlphaMissense scored variants to: {scored_path}")

    return scored_assays


def run_benchmark(
    output_dir: Path,
    max_assays: int = None,
    skip_download: bool = False,
    am_data_dir: str = "/mnt/storage/alphamissense",
    human_only: bool = True,
):
    """
    Run ES Score vs AlphaMissense benchmark.

    Args:
        output_dir: Directory to save all results
        max_assays: Maximum number of assays to process (for testing)
        skip_download: Skip data download step
        am_data_dir: Directory containing AlphaMissense bulk data
        human_only: Only benchmark on human proteins (default True, required for ES Score)
    """
    print("\n" + "=" * 70)
    print("ES Score vs AlphaMissense Benchmark on ProteinGym")
    print(f"Started: {datetime.now().isoformat()}")
    if human_only:
        print("Mode: Human proteins only (required for ES Score pLDDT data)")
    print("=" * 70)

    # Setup directories
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download ProteinGym data
    if not skip_download:
        print("\n[Step 1/5] Downloading ProteinGym data...")
        success = run_download(data_dir)
        if not success:
            print("ERROR: Failed to download data")
            return None
    else:
        print("\n[Step 1/5] Skipping download (--skip_download)")

    # Step 2: Compute ES Scores (human only)
    print("\n[Step 2/5] Computing ES Scores...")
    es_scored_assays = run_scoring_human_only(
        data_dir,
        results_dir,
        max_assays=max_assays,
        smooth_kernel=10,
        use_3d=False,
    )

    if not es_scored_assays:
        print("ERROR: No assays were scored with ES Score")
        return None

    # Step 3: Compute AlphaMissense Scores (human only)
    print("\n[Step 3/5] Computing AlphaMissense Scores...")
    am_scored_assays = run_alphamissense_scoring_human_only(
        data_dir,
        results_dir,
        max_assays=max_assays,
        am_data_dir=am_data_dir,
    )

    if not am_scored_assays:
        print("ERROR: No assays were scored with AlphaMissense")
        return None

    # Step 4: Evaluate both methods
    print("\n[Step 4/5] Evaluating performance...")

    es_results = run_evaluation(es_scored_assays, results_dir / "es_score", method_name="ES Score")
    am_results = run_evaluation(am_scored_assays, results_dir / "alphamissense", method_name="AlphaMissense")

    # Step 5: Generate comparison
    print("\n[Step 5/5] Generating comparison...")

    # Compare methods
    comparison_df = compare_with_baselines(es_results, alphamissense_results=am_results)
    comparison_df.to_csv(results_dir / "method_comparison.csv", index=False)

    # Load detailed results for visualization
    es_results_df = pd.read_csv(results_dir / "es_score" / "detailed_results.csv")
    am_results_df = pd.read_csv(results_dir / "alphamissense" / "detailed_results.csv")

    # Create visualizations
    create_comparison_visualization(es_results_df, am_results_df, results_dir)

    # Generate summary report
    summary = {
        "benchmark_info": {
            "date": datetime.now().isoformat(),
            "max_assays": max_assays,
            "am_data_dir": am_data_dir,
        },
        "es_score": {
            "n_assays": es_results.n_assays_evaluated,
            "mean_spearman": es_results.mean_spearman,
            "std_spearman": es_results.std_spearman,
            "mean_auc": es_results.mean_auc,
            "std_auc": es_results.std_auc,
        },
        "alphamissense": {
            "n_assays": am_results.n_assays_evaluated,
            "mean_spearman": am_results.mean_spearman,
            "std_spearman": am_results.std_spearman,
            "mean_auc": am_results.mean_auc,
            "std_auc": am_results.std_auc,
        },
        "comparison": {
            "spearman_difference": es_results.mean_spearman - am_results.mean_spearman,
            "auc_difference": es_results.mean_auc - am_results.mean_auc if (es_results.mean_auc and am_results.mean_auc) else None,
        },
    }

    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nES Score:")
    print(f"  - Assays evaluated: {es_results.n_assays_evaluated}")
    print(f"  - Mean Spearman ρ: {es_results.mean_spearman:.4f} ± {es_results.std_spearman:.4f}")
    if es_results.mean_auc:
        print(f"  - Mean AUC: {es_results.mean_auc:.4f} ± {es_results.std_auc:.4f}")

    print(f"\nAlphaMissense:")
    print(f"  - Assays evaluated: {am_results.n_assays_evaluated}")
    print(f"  - Mean Spearman ρ: {am_results.mean_spearman:.4f} ± {am_results.std_spearman:.4f}")
    if am_results.mean_auc:
        print(f"  - Mean AUC: {am_results.mean_auc:.4f} ± {am_results.std_auc:.4f}")

    print(f"\nDifference (ES - AM):")
    diff = es_results.mean_spearman - am_results.mean_spearman
    winner = "ES Score" if diff > 0 else "AlphaMissense"
    print(f"  - Spearman ρ: {diff:+.4f} ({winner} performs better)")

    print(f"\nResults saved to: {results_dir}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run ES Score vs AlphaMissense benchmark on ProteinGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="Output directory for results (default: benchmark_result/)",
    )
    parser.add_argument(
        "--max_assays",
        type=int,
        default=None,
        help="Maximum number of assays to process (default: all)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip data download if already present",
    )
    parser.add_argument(
        "--am_data_dir",
        type=str,
        default="/mnt/storage/alphamissense",
        help="Directory containing AlphaMissense bulk data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (all assays)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_assays = args.max_assays
    if args.full:
        max_assays = None

    run_benchmark(
        output_dir=output_dir,
        max_assays=max_assays,
        skip_download=args.skip_download,
        am_data_dir=args.am_data_dir,
    )


if __name__ == "__main__":
    main()
