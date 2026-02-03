#!/usr/bin/env python3
"""
ClinVar Benchmark Evaluation

Evaluates ES Score and other predictors for pathogenic/benign variant classification.

Metrics:
- AUC-ROC (Area Under ROC Curve)
- AUC-PR (Area Under Precision-Recall Curve)
- Sensitivity (recall for pathogenic)
- Specificity (recall for benign)
- MCC (Matthews Correlation Coefficient)
- F1 Score

Supports comparison with:
- EVE model scores
- ESM model scores
- AlphaMissense
- Other VEP predictors
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    f1_score as sklearn_f1
)
import matplotlib.pyplot as plt
import warnings


@dataclass
class ClassificationResult:
    """Results for a single predictor evaluation"""
    predictor_name: str
    n_variants: int
    n_pathogenic: int
    n_benign: int

    # Classification metrics
    auc_roc: float
    auc_pr: float
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    mcc: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None

    # Additional stats
    threshold_optimal: Optional[float] = None
    mann_whitney_pvalue: Optional[float] = None

    # Per-gene statistics
    n_genes: int = 0
    mean_auc_per_gene: Optional[float] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    predictor_results: Dict[str, ClassificationResult]
    best_by_auc_roc: str = ""
    best_by_auc_pr: str = ""
    best_by_mcc: str = ""
    n_variants_total: int = 0
    n_genes_total: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        records = []
        for name, result in self.predictor_results.items():
            records.append({
                "Predictor": name,
                "N Variants": result.n_variants,
                "N Pathogenic": result.n_pathogenic,
                "N Benign": result.n_benign,
                "AUC-ROC": result.auc_roc,
                "AUC-PR": result.auc_pr,
                "Sensitivity": result.sensitivity,
                "Specificity": result.specificity,
                "MCC": result.mcc,
                "F1 Score": result.f1_score,
                "Accuracy": result.accuracy,
                "N Genes": result.n_genes,
                "Mean AUC/Gene": result.mean_auc_per_gene,
                "MW p-value": result.mann_whitney_pvalue
            })
        return pd.DataFrame(records)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels (1 = pathogenic, 0 = benign)
        y_pred: Predicted scores (higher = more likely pathogenic)
        threshold: Classification threshold (default: optimal from ROC)

    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return {}

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except Exception:
        metrics['auc_roc'] = np.nan

    # AUC-PR
    try:
        metrics['auc_pr'] = average_precision_score(y_true, y_pred)
    except Exception:
        metrics['auc_pr'] = np.nan

    # Find optimal threshold if not provided
    if threshold is None:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            # Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]
            metrics['threshold_optimal'] = threshold
        except Exception:
            threshold = np.median(y_pred)

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

        # F1 Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = metrics['sensitivity']
        if precision + recall > 0:
            metrics['f1_score'] = 2 * precision * recall / (precision + recall)
    except Exception:
        pass

    # MCC
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)
    except Exception:
        metrics['mcc'] = np.nan

    # Mann-Whitney U test
    try:
        pathogenic_scores = y_pred[y_true == 1]
        benign_scores = y_pred[y_true == 0]
        _, pvalue = mannwhitneyu(pathogenic_scores, benign_scores, alternative='greater')
        metrics['mann_whitney_pvalue'] = pvalue
    except Exception:
        metrics['mann_whitney_pvalue'] = np.nan

    return metrics


def compute_per_gene_auc(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    gene_col: str,
    min_variants: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-gene AUC and return mean.

    Args:
        df: DataFrame with scores, labels, and gene info
        score_col: Column with prediction scores
        label_col: Column with true labels
        gene_col: Column with gene names
        min_variants: Minimum variants per gene

    Returns:
        Tuple of (mean_auc, dict of per-gene AUCs)
    """
    per_gene_auc = {}

    for gene, group in df.groupby(gene_col):
        if len(group) < min_variants:
            continue

        y_true = group[label_col].values
        y_pred = group[score_col].values

        # Need both classes
        if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
            continue

        try:
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() >= min_variants:
                auc = roc_auc_score(y_true[mask], y_pred[mask])
                per_gene_auc[gene] = auc
        except Exception:
            continue

    mean_auc = np.mean(list(per_gene_auc.values())) if per_gene_auc else np.nan

    return mean_auc, per_gene_auc


def evaluate_predictor(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    predictor_name: str,
    gene_col: Optional[str] = None,
    threshold: Optional[float] = None
) -> ClassificationResult:
    """
    Evaluate a single predictor.

    Args:
        df: DataFrame with scores and labels
        score_col: Column with prediction scores
        label_col: Column with true labels (1 = pathogenic, 0 = benign)
        predictor_name: Name of the predictor
        gene_col: Column with gene names (for per-gene analysis)
        threshold: Optional classification threshold

    Returns:
        ClassificationResult with all metrics
    """
    # Filter to valid data
    valid_mask = df[score_col].notna() & df[label_col].notna()
    data = df[valid_mask].copy()

    y_true = data[label_col].values.astype(int)
    y_pred = data[score_col].values.astype(float)

    n_variants = len(data)
    n_pathogenic = int((y_true == 1).sum())
    n_benign = int((y_true == 0).sum())

    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred, threshold)

    # Per-gene analysis
    n_genes = 0
    mean_auc_per_gene = None
    if gene_col and gene_col in data.columns:
        n_genes = data[gene_col].nunique()
        mean_auc_per_gene, _ = compute_per_gene_auc(
            data, score_col, label_col, gene_col
        )

    return ClassificationResult(
        predictor_name=predictor_name,
        n_variants=n_variants,
        n_pathogenic=n_pathogenic,
        n_benign=n_benign,
        auc_roc=metrics.get('auc_roc', np.nan),
        auc_pr=metrics.get('auc_pr', np.nan),
        sensitivity=metrics.get('sensitivity'),
        specificity=metrics.get('specificity'),
        mcc=metrics.get('mcc'),
        f1_score=metrics.get('f1_score'),
        accuracy=metrics.get('accuracy'),
        threshold_optimal=metrics.get('threshold_optimal'),
        mann_whitney_pvalue=metrics.get('mann_whitney_pvalue'),
        n_genes=n_genes,
        mean_auc_per_gene=mean_auc_per_gene
    )


def evaluate_classification(
    df: pd.DataFrame,
    predictor_cols: List[str],
    predictor_names: Optional[List[str]] = None,
    label_col: str = 'Label',
    gene_col: Optional[str] = 'GeneSymbol'
) -> BenchmarkResults:
    """
    Evaluate multiple predictors.

    Args:
        df: DataFrame with scores and labels
        predictor_cols: List of columns with prediction scores
        predictor_names: Optional list of display names
        label_col: Column with true labels
        gene_col: Column with gene names

    Returns:
        BenchmarkResults with all predictor evaluations
    """
    if predictor_names is None:
        predictor_names = predictor_cols

    results = {}
    for col, name in zip(predictor_cols, predictor_names):
        if col not in df.columns:
            print(f"Warning: Column {col} not found, skipping")
            continue

        print(f"Evaluating {name}...")
        result = evaluate_predictor(
            df, col, label_col, name, gene_col=gene_col
        )
        results[name] = result

    # Find best predictors
    auc_roc_values = {k: v.auc_roc for k, v in results.items()
                      if not np.isnan(v.auc_roc)}
    auc_pr_values = {k: v.auc_pr for k, v in results.items()
                     if not np.isnan(v.auc_pr)}
    mcc_values = {k: v.mcc for k, v in results.items()
                  if v.mcc is not None and not np.isnan(v.mcc)}

    best_auc_roc = max(auc_roc_values.items(), key=lambda x: x[1])[0] if auc_roc_values else ""
    best_auc_pr = max(auc_pr_values.items(), key=lambda x: x[1])[0] if auc_pr_values else ""
    best_mcc = max(mcc_values.items(), key=lambda x: x[1])[0] if mcc_values else ""

    # Total counts
    n_variants_total = df[label_col].notna().sum()
    n_genes_total = df[gene_col].nunique() if gene_col and gene_col in df.columns else 0

    return BenchmarkResults(
        predictor_results=results,
        best_by_auc_roc=best_auc_roc,
        best_by_auc_pr=best_auc_pr,
        best_by_mcc=best_mcc,
        n_variants_total=n_variants_total,
        n_genes_total=n_genes_total
    )


def plot_roc_curves(
    df: pd.DataFrame,
    predictor_cols: List[str],
    predictor_names: Optional[List[str]] = None,
    label_col: str = 'Label',
    output_path: Optional[str] = None,
    title: str = "ROC Curves - ClinVar Pathogenic vs Benign"
):
    """
    Plot ROC curves for multiple predictors.

    Args:
        df: DataFrame with scores and labels
        predictor_cols: List of columns with prediction scores
        predictor_names: Optional list of display names
        label_col: Column with true labels
        output_path: Path to save figure
        title: Plot title
    """
    if predictor_names is None:
        predictor_names = predictor_cols

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictor_cols)))

    for col, name, color in zip(predictor_cols, predictor_names, colors):
        if col not in df.columns:
            continue

        valid_mask = df[col].notna() & df[label_col].notna()
        y_true = df.loc[valid_mask, label_col].values.astype(int)
        y_pred = df.loc[valid_mask, col].values.astype(float)

        if len(np.unique(y_true)) < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {auc:.3f})')
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curves(
    df: pd.DataFrame,
    predictor_cols: List[str],
    predictor_names: Optional[List[str]] = None,
    label_col: str = 'Label',
    output_path: Optional[str] = None,
    title: str = "Precision-Recall Curves - ClinVar Pathogenic vs Benign"
):
    """Plot precision-recall curves for multiple predictors"""
    if predictor_names is None:
        predictor_names = predictor_cols

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictor_cols)))

    # Baseline (proportion of pathogenic)
    valid_mask = df[label_col].notna()
    baseline = df.loc[valid_mask, label_col].mean()

    for col, name, color in zip(predictor_cols, predictor_names, colors):
        if col not in df.columns:
            continue

        valid_mask = df[col].notna() & df[label_col].notna()
        y_true = df.loc[valid_mask, label_col].values.astype(int)
        y_pred = df.loc[valid_mask, col].values.astype(float)

        if len(np.unique(y_true)) < 2:
            continue

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            ax.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP = {ap:.3f})')
        except Exception:
            continue

    ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
               label=f'Baseline ({baseline:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curves to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_score_distributions(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = 'Label',
    predictor_name: str = 'ES Score',
    output_path: Optional[str] = None
):
    """Plot score distributions for pathogenic vs benign variants"""
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_mask = df[score_col].notna() & df[label_col].notna()
    data = df[valid_mask]

    pathogenic = data[data[label_col] == 1][score_col].values
    benign = data[data[label_col] == 0][score_col].values

    bins = np.linspace(
        min(pathogenic.min(), benign.min()),
        max(pathogenic.max(), benign.max()),
        50
    )

    ax.hist(benign, bins=bins, alpha=0.6, label=f'Benign (n={len(benign):,})',
            color='blue', density=True)
    ax.hist(pathogenic, bins=bins, alpha=0.6, label=f'Pathogenic (n={len(pathogenic):,})',
            color='red', density=True)

    # Add median lines
    ax.axvline(np.median(benign), color='blue', linestyle='--', lw=2,
               label=f'Benign median: {np.median(benign):.3f}')
    ax.axvline(np.median(pathogenic), color='red', linestyle='--', lw=2,
               label=f'Pathogenic median: {np.median(pathogenic):.3f}')

    # Mann-Whitney test
    _, pvalue = mannwhitneyu(pathogenic, benign, alternative='greater')

    ax.set_xlabel(f'{predictor_name} Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{predictor_name} Score Distribution\n(Mann-Whitney p = {pvalue:.2e})',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(results: BenchmarkResults):
    """Print formatted summary of benchmark results"""
    print("\n" + "="*80)
    print("ClinVar Pathogenic Variant Benchmark Results")
    print("="*80)
    print(f"Total variants evaluated: {results.n_variants_total:,}")
    print(f"Total genes: {results.n_genes_total:,}")
    print(f"\nBest by AUC-ROC: {results.best_by_auc_roc}")
    print(f"Best by AUC-PR: {results.best_by_auc_pr}")
    print(f"Best by MCC: {results.best_by_mcc}")

    print("\n" + "-"*80)
    print(f"{'Predictor':<25} {'N':>8} {'AUC-ROC':>8} {'AUC-PR':>8} "
          f"{'Sens':>6} {'Spec':>6} {'MCC':>6}")
    print("-"*80)

    # Sort by AUC-ROC
    df = results.to_dataframe()
    df = df.sort_values("AUC-ROC", ascending=False)

    for _, row in df.iterrows():
        auc_roc = f"{row['AUC-ROC']:.3f}" if pd.notna(row['AUC-ROC']) else "N/A"
        auc_pr = f"{row['AUC-PR']:.3f}" if pd.notna(row['AUC-PR']) else "N/A"
        sens = f"{row['Sensitivity']:.2f}" if pd.notna(row['Sensitivity']) else "N/A"
        spec = f"{row['Specificity']:.2f}" if pd.notna(row['Specificity']) else "N/A"
        mcc = f"{row['MCC']:.3f}" if pd.notna(row['MCC']) else "N/A"

        print(f"{row['Predictor']:<25} {row['N Variants']:>8,} {auc_roc:>8} "
              f"{auc_pr:>8} {sens:>6} {spec:>6} {mcc:>6}")

    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ClinVar benchmark")
    parser.add_argument("scored_file", type=str, help="CSV with scored variants")
    parser.add_argument("--predictors", type=str, nargs="+",
                        help="Columns with prediction scores")
    parser.add_argument("--label_col", type=str, default="Label",
                        help="Column with true labels")
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.scored_file)

    # Default predictors
    if args.predictors is None:
        predictors = ['ES_score']
        # Look for other score columns
        for col in df.columns:
            if 'score' in col.lower() and col != 'ES_score':
                predictors.append(col)
    else:
        predictors = args.predictors

    # Evaluate
    results = evaluate_classification(
        df, predictors, label_col=args.label_col
    )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results.to_dataframe().to_csv(output_dir / "evaluation_results.csv", index=False)
        plot_roc_curves(df, predictors, label_col=args.label_col,
                        output_path=str(output_dir / "roc_curves.png"))
        plot_precision_recall_curves(df, predictors, label_col=args.label_col,
                                     output_path=str(output_dir / "pr_curves.png"))

        if 'ES_score' in predictors:
            plot_score_distributions(df, 'ES_score', args.label_col,
                                     output_path=str(output_dir / "es_score_distribution.png"))
