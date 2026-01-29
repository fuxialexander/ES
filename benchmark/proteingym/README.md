# ProteinGym Benchmark for ES Score

Automated evaluation pipeline for benchmarking ES scores against the [ProteinGym](https://proteingym.org) Deep Mutational Scanning (DMS) benchmark.

## Overview

ProteinGym is a comprehensive benchmark for protein fitness prediction containing:
- **~2.7M** missense variants across **217 DMS assays**
- Ground truth experimental fitness measurements
- Standard evaluation metrics (Spearman correlation, AUC, NDCG, etc.)

This module provides:
1. **Automated data download** from ProteinGym
2. **ES score computation** for all benchmark proteins
3. **Evaluation pipeline** with standard metrics
4. **Comparison** with baseline methods

## Quick Start

```bash
# Run the complete pipeline (download, score, evaluate)
python run_benchmark.py --full --output_dir ./proteingym_results

# Quick test with 10 assays
python run_benchmark.py --full --max_assays 10 --output_dir ./test_results

# Just download data
python run_benchmark.py --download --output_dir ./proteingym_data
```

## Pipeline Components

### 1. Data Download (`download_data.py`)

Downloads ProteinGym datasets from the official source.

```bash
python download_data.py --output_dir ./data --dataset substitutions reference
```

Available datasets:
- `substitutions`: DMS substitution benchmark (~1GB)
- `indels`: DMS indels benchmark (~200MB)
- `reference`: Assay metadata and UniProt mappings
- `zero_shot_scores`: Pre-computed baseline scores (~4.4GB)
- `msa`: Multiple Sequence Alignments (~5.2GB)
- `structures`: AlphaFold2 structures (~84MB)

### 2. Data Loader (`proteingym_loader.py`)

Parses and manages ProteinGym benchmark data.

```python
from proteingym_loader import ProteinGymLoader, DMSAssay

# Load data
loader = ProteinGymLoader("./data")

# List available assays
assays = loader.list_assays()
print(f"Found {len(assays)} assays")

# Load a specific assay
assay = loader.load_assay("BRCA1_HUMAN_RING")
print(f"Variants: {len(assay.data)}")
print(f"Sequence length: {len(assay.target_seq)}")
```

### 3. ES Score Computation (`es_scorer.py`)

Computes ES scores for ProteinGym proteins.

```python
from es_scorer import create_scorer_from_project

# Create scorer using project defaults
scorer = create_scorer_from_project(
    smooth_kernel=10,
    use_3d=False
)

# Score all assays
scored_assays = scorer.score_all_assays(loader)
```

### 4. Evaluation (`evaluate.py`)

Evaluates predictions against ground truth.

```python
from evaluate import evaluate_benchmark, print_summary

# Run evaluation
results = evaluate_benchmark(scored_assays, method_name="ES Score")

# Print summary
print_summary(results)
# Output:
# Mean Spearman: 0.XXX ± 0.XXX
# Mean AUC: 0.XXX ± 0.XXX
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Spearman ρ** | Rank correlation with DMS fitness scores |
| **AUC-ROC** | Binary classification (fit vs not-fit) |
| **MCC** | Matthews Correlation Coefficient |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **Top-K Recall** | Recall of top fitness variants |

## Output Files

After running the benchmark, you'll find:

```
proteingym_results/
├── data/
│   └── DMS_ProteinGym_substitutions/
│       ├── BRCA1_HUMAN_RING.csv
│       ├── P53_HUMAN_Giacomelli.csv
│       └── ...
└── results/
    ├── scored_variants.csv      # ES scores for all variants
    ├── detailed_results.csv     # Per-assay metrics
    ├── summary.json             # Aggregated metrics
    ├── comparison.csv           # Comparison with baselines
    └── benchmark_results.png    # Visualization
```

## DMS Data Format

Each ProteinGym assay CSV contains:

| Column | Description |
|--------|-------------|
| `mutant` | Mutation string (e.g., "A1P" or "A1P:D2N") |
| `mutated_sequence` | Full mutant protein sequence |
| `DMS_score` | Experimental fitness measurement |
| `DMS_score_bin` | Binary fitness (1=fit, 0=not fit) |
| `target_seq` | Wild-type reference sequence |

## Baseline Comparison

The pipeline compares ES Score against known baselines:

| Method | Mean Spearman |
|--------|---------------|
| Tranception L | 0.46 |
| EVE | 0.45 |
| VESPA | 0.44 |
| MSA Transformer | 0.43 |
| ESM-1v | 0.42 |

## Advanced Usage

### Custom Scoring Parameters

```python
scorer = ESScorer(
    plddt_file="path/to/plddt.tdt",
    uniprot_mapping_file="path/to/mapping.txt",
    esm_dir="path/to/esm_scores/",
    smooth_kernel=15,
    smooth_method='gaussian',
    interaction_threshold=20,
    use_3d=True
)
```

### Filtering Assays

```python
# Only score assays for specific genes
human_assays = [a for a in loader.list_assays() if "HUMAN" in a]
scored = scorer.score_all_assays(loader, assay_ids=human_assays)
```

### Custom Evaluation

```python
from evaluate import compute_spearman, compute_auc

# Evaluate single assay
rho, pval = compute_spearman(predictions, targets)
auc = compute_auc(predictions, binary_labels)
```

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- tqdm

All dependencies are included in the project's `environment.yml`.

## References

- [ProteinGym Paper](https://papers.nips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)
- [ProteinGym GitHub](https://github.com/OATML-Markslab/ProteinGym)
- [ProteinGym Website](https://proteingym.org)

## Citation

If you use this benchmark pipeline, please cite:

```bibtex
@article{notin2023proteingym,
  title={ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design},
  author={Notin, Pascal and others},
  journal={NeurIPS},
  year={2023}
}
```
