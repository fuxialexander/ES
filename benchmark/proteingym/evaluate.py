#!/usr/bin/env python3
"""
ProteinGym Evaluation Pipeline

Evaluates variant effect prediction scores against ProteinGym DMS benchmarks
using standard metrics:
- Spearman correlation
- NDCG (Normalized Discounted Cumulative Gain)
- AUC (Area Under ROC Curve)
- MCC (Matthews Correlation Coefficient)
- Top-K Recall

Supports both ES Score and AlphaMissense prediction methods.

Note on score direction:
- ES Score and AlphaMissense: higher values indicate more pathogenic/deleterious
- DMS Score: higher values typically indicate better fitness
- Therefore, a NEGATIVE Spearman correlation is expected for pathogenicity predictors
- AUC > 0.5 indicates good discrimination of fit vs non-fit variants
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    auc,
    ndcg_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Results for a single assay evaluation"""
    assay_id: str
    n_variants: int
    spearman_rho: float
    spearman_pvalue: float
    pearson_r: Optional[float] = None
    auc_roc: Optional[float] = None
    mcc: Optional[float] = None
    ndcg: Optional[float] = None
    top_k_recall: Dict[int, float] = field(default_factory=dict)
    gene_name: Optional[str] = None
    uniprot_id: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    method_name: str
    assay_results: List[EvaluationResult]
    mean_spearman: float
    std_spearman: float
    mean_auc: Optional[float] = None
    std_auc: Optional[float] = None
    n_assays_evaluated: int = 0
    n_assays_failed: int = 0


def compute_spearman(
    predictions: np.ndarray,
    targets: np.ndarray,
    remove_nan: bool = True
) -> Tuple[float, float]:
    """
    Compute Spearman correlation coefficient.

    Args:
        predictions: Predicted scores
        targets: Ground truth DMS scores
        remove_nan: Remove NaN values before computation

    Returns:
        Tuple of (correlation, p-value)
    """
    if remove_nan:
        mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[mask]
        targets = targets[mask]

    if len(predictions) < 3:
        return np.nan, np.nan

    return spearmanr(predictions, targets)


def compute_auc(
    predictions: np.ndarray,
    binary_labels: np.ndarray,
    remove_nan: bool = True
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        predictions: Predicted scores (higher = more deleterious)
        binary_labels: Binary fitness labels (1 = fit, 0 = not fit)
        remove_nan: Remove NaN values

    Returns:
        AUC score
    """
    if remove_nan:
        mask = ~(np.isnan(predictions) | np.isnan(binary_labels))
        predictions = predictions[mask]
        binary_labels = binary_labels[mask]

    if len(np.unique(binary_labels)) < 2:
        return np.nan

    # For ES score: higher score = more likely deleterious = lower fitness
    # So we predict NOT fit when ES score is high
    # This means AUC should be computed with predictions as-is for predicting "not fit"
    # or inverted for predicting "fit"
    try:
        # Invert to predict "not fit" (deleterious)
        return roc_auc_score(1 - binary_labels, predictions)
    except:
        return np.nan


def compute_mcc(
    predictions: np.ndarray,
    binary_labels: np.ndarray,
    threshold: Optional[float] = None
) -> float:
    """
    Compute Matthews Correlation Coefficient.

    Args:
        predictions: Predicted scores
        binary_labels: Binary fitness labels
        threshold: Threshold for binarizing predictions (default: median)

    Returns:
        MCC score
    """
    mask = ~(np.isnan(predictions) | np.isnan(binary_labels))
    predictions = predictions[mask]
    binary_labels = binary_labels[mask]

    if len(np.unique(binary_labels)) < 2:
        return np.nan

    if threshold is None:
        threshold = np.median(predictions)

    pred_binary = (predictions > threshold).astype(int)

    # Invert prediction for comparison (high ES = deleterious = not fit)
    pred_binary = 1 - pred_binary

    try:
        return matthews_corrcoef(binary_labels, pred_binary)
    except:
        return np.nan


def compute_ndcg(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores (relevance)
        k: Top-k for NDCG (None = all)

    Returns:
        NDCG score
    """
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]

    if len(predictions) < 2:
        return np.nan

    # Reshape for sklearn
    predictions = predictions.reshape(1, -1)
    targets = targets.reshape(1, -1)

    # Normalize targets to positive
    targets = targets - targets.min() + 1e-6

    try:
        return ndcg_score(targets, -predictions, k=k)  # Negate predictions
    except:
        return np.nan


def compute_top_k_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_values: List[int] = [10, 20, 50, 100]
) -> Dict[int, float]:
    """
    Compute Top-K recall for identifying top variants.

    Args:
        predictions: Predicted scores
        targets: Ground truth DMS scores
        k_values: List of K values to compute

    Returns:
        Dictionary mapping K to recall value
    """
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]

    n = len(predictions)
    results = {}

    for k in k_values:
        if k >= n:
            continue

        # Top-K by prediction (lowest ES score = most fit)
        pred_top_k = set(np.argsort(predictions)[:k])

        # Top-K by ground truth (highest DMS score = most fit)
        true_top_k = set(np.argsort(targets)[-k:])

        # Recall = intersection / K
        recall = len(pred_top_k & true_top_k) / k
        results[k] = recall

    return results


def evaluate_assay(
    scored_df: pd.DataFrame,
    assay_id: str,
    gene_name: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    score_col: str = "es_score",
    dms_col: str = "DMS_score",
    binary_col: str = "DMS_score_bin"
) -> EvaluationResult:
    """
    Evaluate predictions for a single assay.

    Args:
        scored_df: DataFrame with ES scores and DMS scores
        assay_id: Assay identifier
        gene_name: Gene name
        uniprot_id: UniProt ID
        score_col: Column name for predictions
        dms_col: Column name for continuous DMS scores
        binary_col: Column name for binary fitness labels

    Returns:
        EvaluationResult with all metrics
    """
    predictions = scored_df[score_col].values
    targets = scored_df[dms_col].values if dms_col in scored_df.columns else None
    binary_labels = scored_df[binary_col].values if binary_col in scored_df.columns else None

    # Spearman correlation (primary metric)
    if targets is not None:
        rho, pval = compute_spearman(predictions, targets)
        pearson, _ = pearsonr(
            predictions[~np.isnan(predictions) & ~np.isnan(targets)],
            targets[~np.isnan(predictions) & ~np.isnan(targets)]
        ) if len(predictions) > 2 else (np.nan, np.nan)
    else:
        rho, pval, pearson = np.nan, np.nan, np.nan

    # Binary classification metrics
    auc_val = compute_auc(predictions, binary_labels) if binary_labels is not None else None
    mcc_val = compute_mcc(predictions, binary_labels) if binary_labels is not None else None

    # NDCG
    ndcg_val = compute_ndcg(predictions, targets) if targets is not None else None

    # Top-K recall
    top_k = compute_top_k_recall(predictions, targets) if targets is not None else {}

    return EvaluationResult(
        assay_id=assay_id,
        n_variants=len(scored_df),
        spearman_rho=rho,
        spearman_pvalue=pval,
        pearson_r=pearson,
        auc_roc=auc_val,
        mcc=mcc_val,
        ndcg=ndcg_val,
        top_k_recall=top_k,
        gene_name=gene_name,
        uniprot_id=uniprot_id
    )


def evaluate_benchmark(
    scored_assays: Dict[str, pd.DataFrame],
    method_name: str = "ES Score",
    min_variants: int = 10,
    score_col: Optional[str] = None,
) -> BenchmarkResults:
    """
    Evaluate all assays and aggregate results.

    Args:
        scored_assays: Dictionary mapping assay_id to scored DataFrame
        method_name: Name of the method being evaluated
        min_variants: Minimum variants required for evaluation
        score_col: Column name for predictions (auto-detected if None)

    Returns:
        BenchmarkResults with aggregated metrics
    """
    # Auto-detect score column if not specified
    if score_col is None:
        # Check first assay to determine score column
        if scored_assays:
            sample_df = list(scored_assays.values())[0]
            if "am_score" in sample_df.columns:
                score_col = "am_score"
            elif "plddt_score" in sample_df.columns:
                score_col = "plddt_score"
            else:
                score_col = "es_score"
        else:
            score_col = "es_score"

    assay_results = []
    failed = 0

    for assay_id, df in tqdm(scored_assays.items(), desc="Evaluating"):
        if len(df) < min_variants:
            failed += 1
            continue

        try:
            result = evaluate_assay(df, assay_id, score_col=score_col)
            if not np.isnan(result.spearman_rho):
                assay_results.append(result)
            else:
                failed += 1
        except Exception as e:
            print(f"Warning: Failed to evaluate {assay_id}: {e}")
            failed += 1

    if not assay_results:
        return BenchmarkResults(
            method_name=method_name,
            assay_results=[],
            mean_spearman=np.nan,
            std_spearman=np.nan,
            n_assays_evaluated=0,
            n_assays_failed=failed
        )

    # Aggregate metrics
    spearman_values = [r.spearman_rho for r in assay_results]
    auc_values = [r.auc_roc for r in assay_results if r.auc_roc is not None and not np.isnan(r.auc_roc)]

    return BenchmarkResults(
        method_name=method_name,
        assay_results=assay_results,
        mean_spearman=np.mean(spearman_values),
        std_spearman=np.std(spearman_values),
        mean_auc=np.mean(auc_values) if auc_values else None,
        std_auc=np.std(auc_values) if auc_values else None,
        n_assays_evaluated=len(assay_results),
        n_assays_failed=failed
    )


def results_to_dataframe(results: BenchmarkResults) -> pd.DataFrame:
    """Convert benchmark results to a DataFrame"""
    records = []
    for r in results.assay_results:
        record = {
            "assay_id": r.assay_id,
            "gene_name": r.gene_name,
            "uniprot_id": r.uniprot_id,
            "n_variants": r.n_variants,
            "spearman_rho": r.spearman_rho,
            "spearman_pvalue": r.spearman_pvalue,
            "pearson_r": r.pearson_r,
            "auc_roc": r.auc_roc,
            "mcc": r.mcc,
            "ndcg": r.ndcg,
        }
        # Add top-k recall
        for k, v in r.top_k_recall.items():
            record[f"top_{k}_recall"] = v
        records.append(record)

    return pd.DataFrame(records)


def plot_benchmark_results(
    results: BenchmarkResults,
    output_path: Optional[str] = None,
    compare_with: Optional[Dict[str, BenchmarkResults]] = None
):
    """
    Generate visualization of benchmark results.

    Args:
        results: BenchmarkResults to visualize
        output_path: Path to save figure
        compare_with: Dictionary of other methods' results for comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Histogram of Spearman correlations
    ax = axes[0, 0]
    spearman_values = [r.spearman_rho for r in results.assay_results]
    ax.hist(spearman_values, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results.mean_spearman, color='red', linestyle='--',
               label=f'Mean: {results.mean_spearman:.3f}')
    ax.set_xlabel('Spearman Correlation')
    ax.set_ylabel('Count')
    ax.set_title(f'{results.method_name} - Spearman Distribution')
    ax.legend()

    # 2. Scatter plot: variants vs correlation
    ax = axes[0, 1]
    n_variants = [r.n_variants for r in results.assay_results]
    ax.scatter(n_variants, spearman_values, alpha=0.5)
    ax.set_xlabel('Number of Variants')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Correlation vs Dataset Size')

    # 3. AUC distribution
    ax = axes[1, 0]
    auc_values = [r.auc_roc for r in results.assay_results if r.auc_roc is not None]
    if auc_values:
        ax.hist(auc_values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(auc_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(auc_values):.3f}')
        ax.set_xlabel('AUC-ROC')
        ax.set_ylabel('Count')
        ax.set_title('AUC Distribution')
        ax.legend()

    # 4. Top-K recall
    ax = axes[1, 1]
    k_values = [10, 20, 50, 100]
    mean_recalls = []
    for k in k_values:
        recalls = [r.top_k_recall.get(k, np.nan) for r in results.assay_results]
        recalls = [r for r in recalls if not np.isnan(r)]
        mean_recalls.append(np.mean(recalls) if recalls else np.nan)

    ax.bar(range(len(k_values)), mean_recalls)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylabel('Mean Recall')
    ax.set_title('Top-K Recall')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(results: BenchmarkResults):
    """Print a summary of benchmark results"""
    print("\n" + "="*60)
    print(f"Benchmark Results: {results.method_name}")
    print("="*60)
    print(f"Assays evaluated: {results.n_assays_evaluated}")
    print(f"Assays failed: {results.n_assays_failed}")
    print(f"\nSpearman Correlation:")
    print(f"  Mean: {results.mean_spearman:.4f} ± {results.std_spearman:.4f}")

    if results.mean_auc is not None:
        print(f"\nAUC-ROC:")
        print(f"  Mean: {results.mean_auc:.4f} ± {results.std_auc:.4f}")

    # Top performers
    print(f"\nTop 5 Assays by Spearman:")
    sorted_results = sorted(results.assay_results, key=lambda x: x.spearman_rho, reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r.assay_id}: {r.spearman_rho:.4f} (n={r.n_variants})")

    # Bottom performers
    print(f"\nBottom 5 Assays by Spearman:")
    for r in sorted_results[-5:]:
        print(f"  {r.assay_id}: {r.spearman_rho:.4f} (n={r.n_variants})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ProteinGym benchmark")
    parser.add_argument("scored_file", type=str, help="CSV file with scored variants")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--method_name", type=str, default="ES Score")

    args = parser.parse_args()

    # Load scored data
    df = pd.read_csv(args.scored_file)

    # Group by assay
    if "assay_id" in df.columns:
        scored_assays = {aid: group for aid, group in df.groupby("assay_id")}
    else:
        scored_assays = {"unknown": df}

    # Evaluate
    results = evaluate_benchmark(scored_assays, method_name=args.method_name)

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = results_to_dataframe(results)
        results_df.to_csv(output_dir / "detailed_results.csv", index=False)

        # Save plot
        plot_benchmark_results(results, output_dir / "benchmark_results.png")

        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"Method: {results.method_name}\n")
            f.write(f"Assays Evaluated: {results.n_assays_evaluated}\n")
            f.write(f"Mean Spearman: {results.mean_spearman:.4f} ± {results.std_spearman:.4f}\n")
            if results.mean_auc:
                f.write(f"Mean AUC: {results.mean_auc:.4f} ± {results.std_auc:.4f}\n")
