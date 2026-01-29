#!/usr/bin/env python3
"""
Variant Annotation Benchmark Runner

Evaluates ES Score against other VEP predictors using real-world clinical
outcomes from the MSK-IMPACT NSCLC cohort.

Based on: https://github.com/clinical-data-mining/variant-annotation
Reference: Nature Communications (2025)

Usage:
    # Full pipeline (download, score, evaluate)
    python run_benchmark.py --full

    # Just download data
    python run_benchmark.py --download

    # Score and evaluate (data already downloaded)
    python run_benchmark.py --data_dir ./data --output_dir ./results

    # Quick test
    python run_benchmark.py --full --quick
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
from data_loader import VariantAnnotationLoader, VEP_PREDICTORS
from es_scorer import VariantAnnotationScorer, create_scorer_from_project
from evaluate import (
    SurvivalEvaluator,
    BenchmarkResults,
    evaluate_predictor,
    compare_predictors,
    plot_comparison,
    print_summary
)


def run_download(output_dir: Path, datasets: Optional[List[str]] = None) -> bool:
    """Download variant annotation data."""
    print("\n" + "="*60)
    print("Step 1: Downloading MSK-IMPACT NSCLC Data")
    print("="*60)

    if datasets is None:
        datasets = ["clinical", "mutations"]

    results = download_all(output_dir, datasets)

    # Verify
    verification = verify_download(output_dir)
    print_verification(verification)

    return all(results.values())


def run_scoring(
    data_dir: Path,
    output_dir: Path,
    smooth_kernel: int = 10,
    use_3d: bool = False
) -> pd.DataFrame:
    """Score mutations with ES Score."""
    print("\n" + "="*60)
    print("Step 2: Computing ES Scores")
    print("="*60)

    # Load data
    loader = VariantAnnotationLoader(data_dir)
    loader.load()

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
        scorer = VariantAnnotationScorer(
            plddt_file=PROJECT_ROOT / "plddt" / "9606.pLDDT.tdt",
            uniprot_mapping_file=PROJECT_ROOT / "uniprot_to_genename.txt",
            smooth_kernel=smooth_kernel,
            use_3d=use_3d
        )

    # Score mutations
    scored_df = scorer.score_mutations(loader.mutations)

    # Add pathogenicity calls based on ES score
    # Use multiple thresholds for comparison
    for pct in [50, 75, 90]:
        threshold = np.nanpercentile(scored_df["ES_score"], pct)
        col_name = f"ES_score_top{100-pct}pct"
        scored_df[col_name] = (scored_df["ES_score"] > threshold).astype(float)
        scored_df.loc[scored_df["ES_score"].isna(), col_name] = np.nan

    # Save scored data
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_mutations.csv"
    scored_df.to_csv(scored_path, index=False)
    print(f"\nSaved scored mutations to: {scored_path}")

    return scored_df


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
    scored_df: Optional[pd.DataFrame] = None,
    include_veps: bool = True
) -> BenchmarkResults:
    """Evaluate ES Score against other predictors."""
    print("\n" + "="*60)
    print("Step 3: Evaluating Predictors")
    print("="*60)

    # Load data if not provided
    if scored_df is None:
        scored_path = output_dir / "scored_mutations.csv"
        if scored_path.exists():
            scored_df = pd.read_csv(scored_path)
        else:
            raise ValueError("No scored data found. Run scoring first.")

    # Load clinical data for merging
    loader = VariantAnnotationLoader(data_dir)
    clinical = loader.clinical

    # Merge with clinical data
    if "OS_STATUS" not in scored_df.columns:
        scored_df = scored_df.merge(
            clinical.data[["PATIENT_ID", "OS_STATUS", "OS_DURATION"]],
            on="PATIENT_ID",
            how="left"
        )

    # Prepare predictor columns
    predictor_cols = []
    predictor_names = []

    # ES Score variants
    es_cols = [col for col in scored_df.columns if col.startswith("ES_score_top")]
    for col in es_cols:
        predictor_cols.append(col)
        predictor_names.append(f"ES Score ({col.replace('ES_score_', '')})")

    # Standard ES score with median threshold
    if "ES_score" in scored_df.columns:
        threshold = scored_df["ES_score"].median()
        scored_df["ES_score_median"] = (scored_df["ES_score"] > threshold).astype(float)
        scored_df.loc[scored_df["ES_score"].isna(), "ES_score_median"] = np.nan
        predictor_cols.append("ES_score_median")
        predictor_names.append("ES Score (median)")

    # Include other VEP predictors if available
    if include_veps:
        for vep in VEP_PREDICTORS:
            for suffix in ["_Pathogenic", "_Rescue", ""]:
                col = f"{vep}{suffix}"
                if col in scored_df.columns:
                    # Check if it's a binary column or needs conversion
                    if scored_df[col].dtype == object:
                        # Text labels - convert
                        scored_df[f"{col}_binary"] = scored_df[col].apply(
                            lambda x: 1 if str(x).lower() in ['pathogenic', 'deleterious', 'damaging', 'high', 1, '1'] else 0
                        )
                        predictor_cols.append(f"{col}_binary")
                    else:
                        predictor_cols.append(col)
                    predictor_names.append(f"{vep}{' (Rescue)' if 'Rescue' in col else ''}")
                    break

    # OncoKB for ground truth
    oncokb_col = None
    for col in ["oncogenic_binary", "oncogenic", "OncoKB_oncogenic"]:
        if col in scored_df.columns:
            # Convert to binary if needed
            if scored_df[col].dtype == object:
                scored_df["oncokb_binary"] = scored_df[col].apply(
                    lambda x: 1 if 'oncogenic' in str(x).lower() else 0
                )
                oncokb_col = "oncokb_binary"
            else:
                oncokb_col = col
            break

    print(f"\nEvaluating {len(predictor_cols)} predictors:")
    for name in predictor_names:
        print(f"  - {name}")

    if oncokb_col:
        print(f"\nUsing {oncokb_col} as ground truth for AUC calculation")

    # Run evaluation
    evaluator = SurvivalEvaluator()
    results = compare_predictors(
        scored_df,
        predictor_cols,
        predictor_names,
        oncokb_col=oncokb_col,
        evaluator=evaluator
    )

    # Print summary
    print_summary(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed results CSV
    results_df = results.to_dataframe()
    results_df.to_csv(output_dir / "predictor_comparison.csv", index=False)

    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_patients": results.n_patients,
        "n_events": results.n_events,
        "n_predictors": len(results.predictor_results),
        "best_by_hazard_ratio": results.best_by_hr,
        "best_by_auc": results.best_by_auc,
        "predictors": {
            name: {
                "hazard_ratio": res.hazard_ratio,
                "auc_roc": res.auc_roc,
                "n_mutations": res.n_mutations
            }
            for name, res in results.predictor_results.items()
        }
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Visualizations
    try:
        plot_comparison(results, "hazard_ratio", output_dir / "hazard_ratio_comparison.png")
        if results.best_by_auc:
            plot_comparison(results, "auc_roc", output_dir / "auc_roc_comparison.png")

        # KM curves for ES Score
        es_predictor = [col for col in predictor_cols if "ES_score" in col]
        if es_predictor:
            evaluator.plot_kaplan_meier(
                scored_df,
                es_predictor[0],
                "ES Score",
                output_dir / "kaplan_meier_es_score.png"
            )
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")

    return results


def run_full_pipeline(
    output_dir: Path,
    smooth_kernel: int = 10,
    use_3d: bool = False,
    skip_download: bool = False,
    quick: bool = False
):
    """Run the complete benchmark pipeline."""
    print("\n" + "="*60)
    print("Variant Annotation Benchmark Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    data_dir = output_dir / "data"
    results_dir = output_dir / "results"

    # Step 1: Download
    if not skip_download:
        success = run_download(data_dir)
        if not success:
            print("Download failed. Please check your internet connection.")
            return None

    # Step 2: Score
    scored_df = run_scoring(
        data_dir,
        results_dir,
        smooth_kernel=smooth_kernel,
        use_3d=use_3d
    )

    if scored_df is None or len(scored_df) == 0:
        print("Scoring failed. No mutations could be scored.")
        return None

    # Step 3: Evaluate
    results = run_evaluation(
        data_dir,
        results_dir,
        scored_df=scored_df,
        include_veps=not quick
    )

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print(f"Results saved to: {results_dir}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ES Score benchmark against variant annotation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_benchmark.py --full --output_dir ./variant_benchmark

    # Quick test (ES Score only, no other VEPs)
    python run_benchmark.py --full --quick

    # Just download data
    python run_benchmark.py --download --output_dir ./variant_benchmark

    # Score and evaluate existing data
    python run_benchmark.py --data_dir ./variant_benchmark/data --output_dir ./variant_benchmark/results

Reference:
    Based on clinical-data-mining/variant-annotation
    Nature Communications (2025) - Validating ML cancer driver mutation predictions
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
        help="Only download data"
    )
    mode_group.add_argument(
        "--score",
        action="store_true",
        help="Only score mutations (requires downloaded data)"
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
        default="./variant_annotation_benchmark",
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
        help="CSV file with scored mutations (for --evaluate)"
    )

    # Options
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only evaluate ES Score (skip other VEPs)"
    )
    parser.add_argument(
        "--no_veps",
        action="store_true",
        help="Don't include other VEP predictors in evaluation"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.full:
        run_full_pipeline(
            output_dir,
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d,
            skip_download=args.skip_download,
            quick=args.quick
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
            smooth_kernel=args.smooth_kernel,
            use_3d=args.use_3d
        )

    elif args.evaluate:
        data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
        results_dir = output_dir / "results"

        scored_df = None
        if args.scored_file:
            scored_df = pd.read_csv(args.scored_file)

        run_evaluation(
            data_dir,
            results_dir,
            scored_df=scored_df,
            include_veps=not args.no_veps
        )


if __name__ == "__main__":
    main()
