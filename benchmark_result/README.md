# ES Score vs AlphaMissense Benchmark on ProteinGym

This directory contains the benchmark comparing ES Score predictions with AlphaMissense on the ProteinGym DMS (Deep Mutational Scanning) dataset.

## Quick Start

```bash
# Quick test with 50 assays (recommended for initial testing)
cd benchmark_result
uv run python run_es_vs_alphamissense.py --max_assays 50

# Full benchmark (all 217 assays - takes longer)
uv run python run_es_vs_alphamissense.py --full

# Skip download if data already exists
uv run python run_es_vs_alphamissense.py --skip_download --max_assays 50
```

## Prerequisites

1. **AlphaMissense bulk data** must be downloaded to `/mnt/storage/alphamissense/`:
   ```bash
   python -m benchmark.alphamissense.fetcher --download --data_dir /mnt/storage/alphamissense
   ```

2. **ESM LLR data** should exist in the project's `esm1b_LLR/` and `esm_ALL_hotspot/` directories.
   - The ESM LLR data was generated for ~1225 cancer genes using the COSMIC Cancer Gene Census
   - Not all ProteinGym genes have ESM data, so ESM LLR evaluation may have fewer assays
   - Symlinks should point to `/mnt/storage/es/data/esm1b_LLR` and `/mnt/storage/es/data/esm_ALL_hotspot`

3. **pLDDT data** should exist in `plddt/9606.pLDDT.tdt`.

## Three-Way Comparison

The benchmark compares three scoring methods:
1. **ES Score**: Combined evolutionary (ESM) and structural (pLDDT gradient) signals
2. **AlphaMissense**: Google DeepMind's pathogenicity predictor
3. **ESM LLR**: Raw evolutionary signal from ESM language model (without pLDDT)

ESM LLR provides a baseline to understand the contribution of the evolutionary signal alone
vs the combined ES Score (evolutionary + structural).

## Output Structure

After running the benchmark, results are saved to:

```
benchmark_result/
├── data/                              # Downloaded ProteinGym data
│   ├── DMS_substitutions/             # Individual assay CSV files
│   └── DMS_ProteinGym_substitutions.csv
├── results/
│   ├── es_score/                      # ES Score evaluation
│   │   ├── detailed_results.csv       # Per-assay metrics
│   │   ├── summary.json               # Aggregate statistics
│   │   └── benchmark_results.png      # Visualization
│   ├── alphamissense/                 # AlphaMissense evaluation
│   │   ├── detailed_results.csv
│   │   ├── summary.json
│   │   └── benchmark_results.png
│   ├── method_comparison.csv          # Side-by-side comparison
│   ├── benchmark_summary.json         # Overall summary
│   └── es_vs_alphamissense_comparison.png  # Comparison visualization
└── README.md
```

## Metrics

The benchmark evaluates both methods using:

- **Spearman correlation (ρ)**: Rank correlation between predicted scores and experimental DMS fitness scores
- **AUC-ROC**: Area under ROC curve for binary classification (when DMS_score_bin available)
- **MCC**: Matthews Correlation Coefficient
- **NDCG**: Normalized Discounted Cumulative Gain
- **Top-K Recall**: Recall at top 10% predictions

## Expected Results

Based on ProteinGym leaderboard benchmarks, typical performance ranges:

| Method | Mean Spearman ρ |
|--------|-----------------|
| AlphaMissense | ~0.48 |
| EVE | ~0.45 |
| Tranception L | ~0.46 |
| ESM-1v | ~0.42 |

ES Score is expected to show competitive performance, particularly on proteins with:
- Well-defined structural features (high pLDDT regions)
- Known hotspot mutations
- Strong evolutionary conservation signals

## Interpreting Results

### Score Directionality
- **DMS scores**: Higher values = better fitness/function
- **AlphaMissense**: Higher values = more pathogenic/damaging
- **ES Score**: Higher values = higher structural gradient (potential functional importance)

### Correlation Interpretation
- **AlphaMissense negative correlation**: Expected! High pathogenicity should correlate with low fitness (negative DMS score)
- **ES Score correlation**: Positive correlation means high ES scores correlate with high fitness, negative means high ES scores correlate with low fitness

### Comparison Notes
1. **Raw Spearman ρ comparison**: Take absolute values when comparing predictive power, since AlphaMissense predicts pathogenicity (negative correlation with fitness) while ES Score may predict functional importance
2. **AUC-ROC**: More directly comparable as it measures classification performance regardless of score direction
3. **Per-assay scatter plot**: Shows how methods perform on individual assays
4. **Distribution comparison**: Box plots show variance in performance across assays
