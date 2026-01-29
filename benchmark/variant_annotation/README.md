# Variant Annotation Benchmark

Evaluates ES Score against real-world clinical outcomes using the MSK-IMPACT NSCLC cohort.

## Overview

This benchmark compares ES Score with other Variant Effect Predictors (VEPs) using:
- **Survival outcomes** (Cox proportional hazards models)
- **OncoKB classifications** (ROC-AUC)
- **Real-world patient data** (7,965 NSCLC patients)

Based on the methodology from:
- **Repository**: https://github.com/clinical-data-mining/variant-annotation
- **Paper**: [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-63461-8) - Validating machine learning cancer driver mutation predictions against real-world clinical data

## Quick Start

```bash
# Run full benchmark pipeline
python run_benchmark.py --full

# Quick test (ES Score only)
python run_benchmark.py --full --quick

# With custom output directory
python run_benchmark.py --full --output_dir ./my_benchmark
```

## Data

The benchmark uses MSK-IMPACT NSCLC data:

| Dataset | Description | Size |
|---------|-------------|------|
| `clinical` | Clinical data with survival outcomes | ~8K patients |
| `mutations` | Missense mutations with VEP annotations | ~100K mutations |

### VEP Predictors Evaluated

- **ES Score** (this project)
- SIFT
- PolyPhen
- CADD
- REVEL
- MutationAssessor
- AlphaMissense
- VEST
- MetaLR/MetaSVM
- FATHMM
- PROVEAN
- LRT
- MutationTaster

## Usage

### Full Pipeline

```bash
python run_benchmark.py --full --output_dir ./variant_annotation_benchmark
```

This will:
1. Download MSK-IMPACT NSCLC data
2. Compute ES scores for all mutations
3. Evaluate all predictors using survival analysis
4. Generate comparison plots and reports

### Step-by-Step

```bash
# 1. Download data only
python run_benchmark.py --download --output_dir ./benchmark

# 2. Score mutations only
python run_benchmark.py --score --data_dir ./benchmark/data --output_dir ./benchmark

# 3. Evaluate only
python run_benchmark.py --evaluate --data_dir ./benchmark/data --output_dir ./benchmark/results
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--smooth_kernel` | 10 | pLDDT smoothing kernel size |
| `--use_3d` | False | Use 3D spatial averaging |
| `--skip_download` | False | Skip data download in full mode |
| `--quick` | False | Quick mode (ES Score only) |
| `--no_veps` | False | Exclude other VEP predictors |

## Output

```
variant_annotation_benchmark/
├── data/
│   ├── msk_impact_nsclc_clinical.csv
│   └── msk_impact_nsclc_missense_mutations.csv
└── results/
    ├── scored_mutations.csv           # ES scores for all mutations
    ├── predictor_comparison.csv       # Full comparison table
    ├── summary.json                   # Summary statistics
    ├── hazard_ratio_comparison.png    # Forest plot of HRs
    ├── auc_roc_comparison.png         # AUC comparison bar chart
    └── kaplan_meier_es_score.png      # Survival curves
```

## Evaluation Metrics

### Survival Analysis

- **Hazard Ratio (HR)**: Higher HR = better at predicting poor survival
- **Log-rank p-value**: Statistical significance of survival difference
- **Median survival**: Kaplan-Meier estimated median survival times

### Classification Metrics

- **AUC-ROC**: Area under ROC curve for OncoKB classification
- **AUC-PR**: Area under precision-recall curve
- **Sensitivity/Specificity**: Binary classification performance
- **F1 Score**: Harmonic mean of precision and recall

## Requirements

```bash
# Core requirements
pip install numpy pandas scipy scikit-learn matplotlib tqdm

# For survival analysis (recommended)
pip install lifelines

# Or install all via conda
conda env create -f environment.yml
```

## Methodology

### ES Score Computation

```python
ES_score = normalize(gradient(pLDDT) × ESM_score)
```

- **pLDDT**: AlphaFold confidence (smoothed with Gaussian kernel)
- **ESM**: Evolutionary scores from ESM1b language model
- **Gradient**: Squared gradient highlights structural transitions

### Survival Evaluation

1. Mutations classified as pathogenic/benign by each predictor
2. Cox proportional hazards models fitted
3. Hazard ratios compared between groups
4. Kaplan-Meier curves visualized

## Citation

If you use this benchmark, please cite:

```bibtex
@article{variant_annotation_2025,
  title={Validating machine learning cancer driver mutation predictions
         against real-world clinical data},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-025-63461-8}
}
```

## License

This benchmark code follows the ES Score project license. The MSK-IMPACT data is provided by Memorial Sloan Kettering Cancer Center.
