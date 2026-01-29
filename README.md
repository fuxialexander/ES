# ES Score: Unsupervised Prediction of Cancer Hotspots

Combines evolutionary (ESM language model) and structural (AlphaFold pLDDT) information to predict cancer mutation hotspots.

## Quick Start

```bash
git clone git@github.com:fuxialexander/ES.git
conda env create -f environment.yml
conda activate es
cd website
python app.py  # Runs on port 4568
```

## Usage

### Compute ES Scores for New Genes

```bash
python plot.py --transition cosmic_aa_transition.csv --gap 5 --interaction 15 \
  --hotspot 0.1 --kernel 10 --smooth_method 'gaussian' \
  plddt/9606.pLDDT.tdt uniprot_to_genename.txt {data_folder} genes.txt
```

**Requirements:**
- `9606.pLDDT.tdt`: Download from [ProcessedAlphafold](https://github.com/normandavey/ProcessedAlphafold/blob/main/9606.pLDDT.tdt.zip)
- `{data_folder}/genes.txt`: List of genes to analyze
- `{data_folder}/mutations.txt`: Mutation data (recurrence, gene, position)

Example data: [rank_all_cosmic/](https://github.com/fuxialexander/ES/tree/main/rank_all_cosmic)

### Generate Visualizations

```bash
python plot.py --plot --transition cosmic_aa_transition.csv --gap 5 --interaction 15 \
  --hotspot 0.1 --kernel 10 --smooth_method 'gaussian' \
  plddt/9606.pLDDT.tdt uniprot_to_genename.txt {data_folder} genes.txt
```

## Benchmarks

ES Score has been validated against multiple benchmarks:

### 1. COSMIC/OncoKB Benchmark
ROC curve analysis comparing ES Score against EVE, CTAT 3D, and other methods using OncoKB oncogenic annotations.

```bash
cd benchmark
python plot_roc.py
```

### 2. ProteinGym Benchmark
Evaluation against 217 Deep Mutational Scanning (DMS) assays from ProteinGym.

```bash
cd benchmark/proteingym
python run_benchmark.py --full --max_assays 50
```

### 3. Variant Annotation Benchmark (MSK-IMPACT NSCLC)
Real-world clinical validation using survival outcomes from 7,965 NSCLC patients.

```bash
cd benchmark/variant_annotation
python run_benchmark.py --full
```

Based on [clinical-data-mining/variant-annotation](https://github.com/clinical-data-mining/variant-annotation) and [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-63461-8).

**Metrics:**
- Hazard ratios (Cox proportional hazards)
- Kaplan-Meier survival curves
- AUC-ROC against OncoKB classifications
- Comparison with 13 VEP predictors (SIFT, PolyPhen, CADD, REVEL, AlphaMissense, etc.)

## Output Format

Results are saved to `{data_folder}/genes.txt.scores.txt`:

```
gene    position    ES_score    mutation_category
ABL1    1           0.234       not_mutated
ABL1    315         0.892       hotspot
```

## References

- **AlphaFold pLDDT**: https://github.com/normandavey/ProcessedAlphafold
- **COSMIC database**: https://cancer.sanger.ac.uk/cosmic
- **ESM models**: https://github.com/facebookresearch/esm
- **ProteinGym**: https://proteingym.org/
