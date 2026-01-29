#!/usr/bin/env python3
"""
Evaluation Module for Variant Annotation Benchmark

Evaluates VEP predictors using real-world clinical outcomes:
1. Survival analysis (Cox proportional hazards models)
2. Hazard ratio comparisons (pathogenic vs benign mutations)
3. ROC curves for OncoKB classification
4. Comparison across multiple predictors

Based on methodology from:
Nature Communications (2025) - Validating ML cancer driver mutation predictions
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import warnings

# Optional: lifelines for survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    warnings.warn("lifelines not installed. Survival analysis will be limited.")


@dataclass
class EvaluationResult:
    """Results for a single predictor evaluation."""
    predictor_name: str
    n_mutations: int
    n_pathogenic: int
    n_benign: int

    # Survival metrics
    hazard_ratio: Optional[float] = None
    hazard_ratio_ci_lower: Optional[float] = None
    hazard_ratio_ci_upper: Optional[float] = None
    hazard_ratio_pvalue: Optional[float] = None

    # Log-rank test
    logrank_pvalue: Optional[float] = None

    # OncoKB classification metrics
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1_score: Optional[float] = None

    # Additional stats
    median_survival_pathogenic: Optional[float] = None
    median_survival_benign: Optional[float] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results for all predictors."""
    predictor_results: Dict[str, EvaluationResult]
    best_by_hr: str = ""
    best_by_auc: str = ""
    n_patients: int = 0
    n_events: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easy comparison."""
        records = []
        for name, result in self.predictor_results.items():
            records.append({
                "Predictor": name,
                "N Mutations": result.n_mutations,
                "N Pathogenic": result.n_pathogenic,
                "N Benign": result.n_benign,
                "Hazard Ratio": result.hazard_ratio,
                "HR 95% CI Lower": result.hazard_ratio_ci_lower,
                "HR 95% CI Upper": result.hazard_ratio_ci_upper,
                "HR p-value": result.hazard_ratio_pvalue,
                "Log-rank p-value": result.logrank_pvalue,
                "AUC-ROC": result.auc_roc,
                "AUC-PR": result.auc_pr,
                "Sensitivity": result.sensitivity,
                "Specificity": result.specificity,
                "F1 Score": result.f1_score,
                "Median Survival (Path)": result.median_survival_pathogenic,
                "Median Survival (Benign)": result.median_survival_benign
            })
        return pd.DataFrame(records)


class SurvivalEvaluator:
    """
    Evaluates variant effect predictors using survival outcomes.

    Methods:
    1. Cox proportional hazards regression
    2. Kaplan-Meier curves
    3. Log-rank tests
    4. Hazard ratio comparisons
    """

    def __init__(
        self,
        duration_col: str = "OS_DURATION",
        event_col: str = "OS_STATUS"
    ):
        """
        Initialize evaluator.

        Args:
            duration_col: Column name for survival duration
            event_col: Column name for event indicator (1 = event occurred)
        """
        self.duration_col = duration_col
        self.event_col = event_col

    def compute_hazard_ratio(
        self,
        df: pd.DataFrame,
        pathogenic_col: str,
        covariates: Optional[List[str]] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute hazard ratio using Cox proportional hazards.

        Args:
            df: DataFrame with survival and pathogenicity data
            pathogenic_col: Column with pathogenicity calls (1 = pathogenic)
            covariates: Optional list of covariate column names

        Returns:
            Tuple of (hazard_ratio, ci_lower, ci_upper, pvalue)
        """
        if not HAS_LIFELINES:
            return np.nan, np.nan, np.nan, np.nan

        # Prepare data
        cols = [self.duration_col, self.event_col, pathogenic_col]
        if covariates:
            cols.extend(covariates)

        data = df[cols].dropna()

        if len(data) < 10 or data[pathogenic_col].nunique() < 2:
            return np.nan, np.nan, np.nan, np.nan

        try:
            cph = CoxPHFitter()
            cph.fit(data, duration_col=self.duration_col, event_col=self.event_col)

            hr = np.exp(cph.params_[pathogenic_col])
            ci = np.exp(cph.confidence_intervals_.loc[pathogenic_col])
            pvalue = cph.summary.loc[pathogenic_col, 'p']

            return hr, ci.iloc[0], ci.iloc[1], pvalue
        except Exception as e:
            warnings.warn(f"Cox PH failed: {e}")
            return np.nan, np.nan, np.nan, np.nan

    def compute_logrank(
        self,
        df: pd.DataFrame,
        pathogenic_col: str
    ) -> float:
        """
        Compute log-rank test p-value.

        Args:
            df: DataFrame with survival and pathogenicity data
            pathogenic_col: Column with pathogenicity calls

        Returns:
            p-value from log-rank test
        """
        if not HAS_LIFELINES:
            return np.nan

        data = df[[self.duration_col, self.event_col, pathogenic_col]].dropna()

        if len(data) < 10:
            return np.nan

        pathogenic = data[data[pathogenic_col] == 1]
        benign = data[data[pathogenic_col] == 0]

        if len(pathogenic) < 5 or len(benign) < 5:
            return np.nan

        try:
            result = logrank_test(
                pathogenic[self.duration_col],
                benign[self.duration_col],
                pathogenic[self.event_col],
                benign[self.event_col]
            )
            return result.p_value
        except Exception:
            return np.nan

    def compute_median_survival(
        self,
        df: pd.DataFrame,
        pathogenic_col: str
    ) -> Tuple[float, float]:
        """
        Compute median survival times for pathogenic and benign groups.

        Args:
            df: DataFrame with survival and pathogenicity data
            pathogenic_col: Column with pathogenicity calls

        Returns:
            Tuple of (median_pathogenic, median_benign)
        """
        data = df[[self.duration_col, self.event_col, pathogenic_col]].dropna()

        pathogenic = data[data[pathogenic_col] == 1]
        benign = data[data[pathogenic_col] == 0]

        if not HAS_LIFELINES:
            # Simple median (may be censored)
            med_path = pathogenic[self.duration_col].median() if len(pathogenic) > 0 else np.nan
            med_benign = benign[self.duration_col].median() if len(benign) > 0 else np.nan
            return med_path, med_benign

        # Use Kaplan-Meier for proper median estimation
        try:
            kmf = KaplanMeierFitter()

            kmf.fit(pathogenic[self.duration_col], pathogenic[self.event_col])
            med_path = kmf.median_survival_time_

            kmf.fit(benign[self.duration_col], benign[self.event_col])
            med_benign = kmf.median_survival_time_

            return med_path, med_benign
        except Exception:
            return np.nan, np.nan

    def plot_kaplan_meier(
        self,
        df: pd.DataFrame,
        pathogenic_col: str,
        predictor_name: str,
        output_path: Optional[str] = None
    ):
        """
        Plot Kaplan-Meier survival curves.

        Args:
            df: DataFrame with survival and pathogenicity data
            pathogenic_col: Column with pathogenicity calls
            predictor_name: Name of the predictor
            output_path: Path to save the figure
        """
        if not HAS_LIFELINES:
            print("lifelines required for KM plots")
            return

        data = df[[self.duration_col, self.event_col, pathogenic_col]].dropna()

        pathogenic = data[data[pathogenic_col] == 1]
        benign = data[data[pathogenic_col] == 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()

        # Pathogenic group
        kmf.fit(pathogenic[self.duration_col], pathogenic[self.event_col],
                label=f"Pathogenic (n={len(pathogenic)})")
        kmf.plot(ax=ax, ci_show=True, color='red')

        # Benign group
        kmf.fit(benign[self.duration_col], benign[self.event_col],
                label=f"Benign (n={len(benign)})")
        kmf.plot(ax=ax, ci_show=True, color='blue')

        # Log-rank p-value
        pvalue = self.compute_logrank(df, pathogenic_col)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"Kaplan-Meier Curves - {predictor_name}\n(Log-rank p = {pvalue:.4f})")
        ax.legend(loc='lower left')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved KM plot to: {output_path}")
        else:
            plt.show()

        plt.close()


def evaluate_predictor(
    df: pd.DataFrame,
    pathogenic_col: str,
    predictor_name: str,
    oncokb_col: Optional[str] = None,
    evaluator: Optional[SurvivalEvaluator] = None
) -> EvaluationResult:
    """
    Evaluate a single predictor.

    Args:
        df: DataFrame with mutations and outcomes
        pathogenic_col: Column with pathogenicity calls (1 = pathogenic)
        predictor_name: Name of the predictor
        oncokb_col: Column with OncoKB ground truth (for AUC)
        evaluator: SurvivalEvaluator instance

    Returns:
        EvaluationResult with all metrics
    """
    if evaluator is None:
        evaluator = SurvivalEvaluator()

    # Filter to valid pathogenicity calls
    valid_mask = df[pathogenic_col].notna()
    data = df[valid_mask].copy()

    n_mutations = len(data)
    n_pathogenic = int((data[pathogenic_col] == 1).sum())
    n_benign = int((data[pathogenic_col] == 0).sum())

    result = EvaluationResult(
        predictor_name=predictor_name,
        n_mutations=n_mutations,
        n_pathogenic=n_pathogenic,
        n_benign=n_benign
    )

    # Survival analysis
    if evaluator.duration_col in data.columns and evaluator.event_col in data.columns:
        hr, ci_lo, ci_hi, pval = evaluator.compute_hazard_ratio(data, pathogenic_col)
        result.hazard_ratio = hr
        result.hazard_ratio_ci_lower = ci_lo
        result.hazard_ratio_ci_upper = ci_hi
        result.hazard_ratio_pvalue = pval

        result.logrank_pvalue = evaluator.compute_logrank(data, pathogenic_col)

        med_path, med_benign = evaluator.compute_median_survival(data, pathogenic_col)
        result.median_survival_pathogenic = med_path
        result.median_survival_benign = med_benign

    # OncoKB classification metrics
    if oncokb_col and oncokb_col in data.columns:
        # Filter to mutations with OncoKB annotations
        onco_mask = data[oncokb_col].notna()
        onco_data = data[onco_mask]

        if len(onco_data) > 10:
            y_true = onco_data[oncokb_col].values
            y_pred = onco_data[pathogenic_col].values

            # AUC-ROC (for continuous predictions)
            try:
                result.auc_roc = roc_auc_score(y_true, y_pred)
            except Exception:
                pass

            # AUC-PR
            try:
                result.auc_pr = average_precision_score(y_true, y_pred)
            except Exception:
                pass

            # Binary metrics
            if np.array_equal(y_pred, y_pred.astype(int)):
                y_pred_bin = y_pred.astype(int)
                y_true_bin = y_true.astype(int)

                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(
                    y_true_bin, y_pred_bin, labels=[0, 1]
                ).ravel()

                result.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                result.specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = result.sensitivity
                if precision + recall > 0:
                    result.f1_score = 2 * precision * recall / (precision + recall)

    return result


def compare_predictors(
    df: pd.DataFrame,
    predictor_cols: List[str],
    predictor_names: Optional[List[str]] = None,
    oncokb_col: Optional[str] = None,
    evaluator: Optional[SurvivalEvaluator] = None
) -> BenchmarkResults:
    """
    Compare multiple predictors.

    Args:
        df: DataFrame with mutations and outcomes
        predictor_cols: List of columns with pathogenicity calls
        predictor_names: Optional list of display names
        oncokb_col: Column with OncoKB ground truth
        evaluator: SurvivalEvaluator instance

    Returns:
        BenchmarkResults with all predictor evaluations
    """
    if evaluator is None:
        evaluator = SurvivalEvaluator()

    if predictor_names is None:
        predictor_names = predictor_cols

    results = {}
    for col, name in zip(predictor_cols, predictor_names):
        if col not in df.columns:
            print(f"Warning: Column {col} not found, skipping")
            continue

        print(f"Evaluating {name}...")
        result = evaluate_predictor(
            df, col, name,
            oncokb_col=oncokb_col,
            evaluator=evaluator
        )
        results[name] = result

    # Find best predictors
    hr_values = {k: v.hazard_ratio for k, v in results.items()
                 if v.hazard_ratio is not None and not np.isnan(v.hazard_ratio)}
    auc_values = {k: v.auc_roc for k, v in results.items()
                  if v.auc_roc is not None and not np.isnan(v.auc_roc)}

    best_hr = max(hr_values.items(), key=lambda x: x[1])[0] if hr_values else ""
    best_auc = max(auc_values.items(), key=lambda x: x[1])[0] if auc_values else ""

    # Count patients and events
    n_patients = df["PATIENT_ID"].nunique() if "PATIENT_ID" in df.columns else len(df)
    n_events = int(df[evaluator.event_col].sum()) if evaluator.event_col in df.columns else 0

    return BenchmarkResults(
        predictor_results=results,
        best_by_hr=best_hr,
        best_by_auc=best_auc,
        n_patients=n_patients,
        n_events=n_events
    )


def plot_comparison(
    results: BenchmarkResults,
    metric: str = "hazard_ratio",
    output_path: Optional[str] = None
):
    """
    Plot comparison of predictors.

    Args:
        results: BenchmarkResults from compare_predictors
        metric: 'hazard_ratio' or 'auc_roc'
        output_path: Path to save figure
    """
    df = results.to_dataframe()
    df = df.sort_values(f"{'Hazard Ratio' if metric == 'hazard_ratio' else 'AUC-ROC'}",
                        ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    if metric == "hazard_ratio":
        # Forest plot style
        y_pos = range(len(df))
        ax.barh(y_pos, df["Hazard Ratio"], color='steelblue', alpha=0.7)

        # Add confidence intervals
        for i, (_, row) in enumerate(df.iterrows()):
            if pd.notna(row["HR 95% CI Lower"]) and pd.notna(row["HR 95% CI Upper"]):
                ax.plot([row["HR 95% CI Lower"], row["HR 95% CI Upper"]],
                        [i, i], 'k-', linewidth=2)
                ax.plot([row["HR 95% CI Lower"]], [i], 'k|', markersize=10)
                ax.plot([row["HR 95% CI Upper"]], [i], 'k|', markersize=10)

        ax.axvline(1.0, color='red', linestyle='--', label='HR = 1')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Predictor"])
        ax.set_xlabel("Hazard Ratio (95% CI)")
        ax.set_title("Predictor Comparison - Hazard Ratios\n(Higher = Better at predicting poor survival)")

    else:  # AUC-ROC
        y_pos = range(len(df))
        ax.barh(y_pos, df["AUC-ROC"], color='steelblue', alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', label='Random')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Predictor"])
        ax.set_xlabel("AUC-ROC")
        ax.set_title("Predictor Comparison - AUC-ROC for OncoKB Classification")
        ax.set_xlim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(results: BenchmarkResults):
    """Print formatted summary of benchmark results."""
    print("\n" + "="*70)
    print("Variant Annotation Benchmark Results")
    print("="*70)
    print(f"Patients: {results.n_patients:,}")
    print(f"Events (deaths): {results.n_events:,}")
    print(f"\nBest by Hazard Ratio: {results.best_by_hr}")
    print(f"Best by AUC-ROC: {results.best_by_auc}")

    print("\n" + "-"*70)
    print(f"{'Predictor':<20} {'N Mut':>8} {'HR':>8} {'HR p-val':>10} {'AUC':>8}")
    print("-"*70)

    # Sort by hazard ratio
    df = results.to_dataframe()
    df = df.sort_values("Hazard Ratio", ascending=False)

    for _, row in df.iterrows():
        hr_str = f"{row['Hazard Ratio']:.3f}" if pd.notna(row['Hazard Ratio']) else "N/A"
        pval_str = f"{row['HR p-value']:.1e}" if pd.notna(row['HR p-value']) else "N/A"
        auc_str = f"{row['AUC-ROC']:.3f}" if pd.notna(row['AUC-ROC']) else "N/A"

        print(f"{row['Predictor']:<20} {row['N Mutations']:>8,} {hr_str:>8} "
              f"{pval_str:>10} {auc_str:>8}")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictors")
    parser.add_argument("data_file", type=str, help="CSV with scored mutations")
    parser.add_argument("--predictors", type=str, nargs="+",
                        help="Columns with pathogenicity calls")
    parser.add_argument("--oncokb", type=str, help="OncoKB column for AUC")
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    df = pd.read_csv(args.data_file)

    # Default predictors
    if args.predictors is None:
        # Look for standard VEP columns
        predictors = [col for col in df.columns if col.endswith("_Pathogenic")]
        if "ES_score_Pathogenic" not in predictors and "ES_score" in df.columns:
            # Add ES score
            df["ES_score_Pathogenic"] = (df["ES_score"] > df["ES_score"].median()).astype(int)
            predictors.append("ES_score_Pathogenic")
    else:
        predictors = args.predictors

    results = compare_predictors(df, predictors, oncokb_col=args.oncokb)

    print_summary(results)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results.to_dataframe().to_csv(output_dir / "predictor_comparison.csv", index=False)
        plot_comparison(results, "hazard_ratio", output_dir / "hazard_ratio_comparison.png")
        if results.best_by_auc:
            plot_comparison(results, "auc_roc", output_dir / "auc_roc_comparison.png")
