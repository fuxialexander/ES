# CLAUDE.md - AI Assistant Guide for ES Score Project

## Project Overview

**ES Score** is a bioinformatics research project for unsupervised prediction of cancer mutation hotspots by combining evolutionary and structural information. The algorithm integrates:

- **Evolutionary signals**: ESM language model scores
- **Structural signals**: AlphaFold-predicted pLDDT confidence scores
- **Mutation data**: COSMIC database mutation frequencies
- **Protein interactions**: 3D pairwise distance calculations

## Quick Reference

```bash
# Setup environment (using UV - recommended)
uv sync                          # Install dependencies
uv sync --extra survival         # Include survival analysis (lifelines)
uv sync --extra all              # Install all optional dependencies

# Alternative: Setup environment (using Conda - legacy)
conda env create -f environment.yml
conda activate es

# Run main analysis
uv run python plot.py --transition cosmic_aa_transition.csv --gap 5 --interaction 15 \
  --hotspot 0.1 --kernel 10 --smooth_method 'gaussian' \
  plddt/9606.pLDDT.tdt uniprot_to_genename.txt {data_folder} genes.txt

# Run web visualization
cd website && uv run python app.py  # Runs on port 4568
```

## Directory Structure

```
ES/
├── plot.py                    # Main computation engine - Gene class & ES scoring
├── plot_gs_rank.py            # Statistical analysis & PCA-based ranking
├── plot_all_freq.py           # Mutation frequency visualization
├── cosmic_aa_transition.py    # COSMIC mutation data parser
├── bcr_abl1_lddt.py           # BCR-ABL1 pLDDT score extraction
├── plot_slim_interface.py     # SLiM/interface mutation analysis
├── pyproject.toml             # UV/pip dependencies and project config
├── .python-version            # Python version for UV
├── environment.yml            # Conda environment (legacy, 200+ dependencies)
├── uniprot_to_genename.txt    # UniProt ID ↔ Gene name mapping
├── cosmic_aa_transition.csv   # Amino acid transition probability matrix
├── data.feather               # Serialized data (Git LFS tracked)
├── cosmic.feather             # COSMIC mutation data (Git LFS tracked)
│
├── tests/                     # Unit tests
│   └── test_proteingym.py     # ProteinGym benchmark tests (40 tests)
│
├── website/                   # Dash-based interactive web application
│   ├── app.py                 # Main web app entry point
│   └── data.csv               # Precomputed ES scores for display
│
├── rank_all_cosmic/           # ES scores for ~302 cancer-related genes
├── bcr_abl1/                  # BCR-ABL1 fusion gene analysis
├── nt5c2_dimer/               # NT5C2 dimer structure analysis
└── benchmark/                 # Benchmarking pipelines
    ├── plot_roc.py            # COSMIC/OncoKB ROC curve analysis
    ├── plot_eve.py            # EVE model comparison
    ├── alphamissense/         # AlphaMissense prediction fetcher
    │   ├── README.md          # Usage documentation
    │   ├── fetcher.py         # Main unified interface
    │   ├── data_loader.py     # Bulk data loading from Zenodo
    │   ├── ensembl_client.py  # Ensembl VEP REST API client
    │   └── hegelab_client.py  # HegeLab web resource client
    ├── proteingym/            # ProteinGym DMS benchmark (217 assays)
    │   ├── run_benchmark.py   # Main pipeline
    │   ├── proteingym_loader.py
    │   ├── es_scorer.py
    │   └── evaluate.py
    ├── variant_annotation/    # MSK-IMPACT NSCLC clinical benchmark
    │   ├── run_benchmark.py   # Main pipeline
    │   ├── download_data.py   # Data downloader
    │   ├── data_loader.py     # Clinical/mutation loader
    │   ├── es_scorer.py       # ES score computation
    │   └── evaluate.py        # Survival analysis (Cox PH)
    └── clinvar/               # ClinVar pathogenic variant benchmark
        ├── run_benchmark.py   # Main pipeline
        ├── download_data.py   # ClinVar data downloader
        ├── clinvar_loader.py  # Pathogenic/benign variant loader
        ├── es_scorer.py       # ES score computation
        └── evaluate.py        # Classification metrics (AUC-ROC, MCC, etc.)
```

## Core Architecture

### Main Classes & Functions

**`plot.py` - Central computation module:**

```python
class Gene:
    """Encapsulates per-protein ES score computation"""
    __init__(data, plddt, esm, ...)     # Initialize with mutation data
    get_pairwise_distance()              # 3D CA-atom distance calculations
    get_final_score_gated_grad()         # 2D ES score calculation
    get_final_score_gated_grad_3d()      # 3D-aware scoring with interactions

# Utility functions
normalize(x)                             # Min-max normalization
smooth(arr, kernel, method)              # Gaussian/convolution smoothing
square_grad(f)                           # Squared gradient computation
get_3d_avg/max/prod(x, matrix)           # 3D aggregation functions
```

### ES Score Algorithm

```python
ES_score = normalize(
    get_3d_avg(
        grad * esm,  # Structural gradient × Evolutionary signal
        pairwise_distance < interaction_threshold
    )
)
```

Where:
- `grad` = Normalized squared gradient of smoothed pLDDT
- `esm` = Negative mean ESM1b log-likelihood ratios
- `pairwise_distance` = 3D CA-atom distances from AlphaFold structure
- `3D averaging` = Mean signal of residues within distance threshold

## Data Flow

```
Input: COSMIC Database → mutations.txt, genes.txt
           ↓
┌─────────────────────────────────────────────┐
│  plot.py - Main Analysis Pipeline           │
│  1. Load AlphaFold pLDDT scores             │
│  2. Parse mutation frequencies               │
│  3. For each gene:                          │
│     - Extract structural data               │
│     - Compute pairwise distances            │
│     - Load ESM evolutionary scores          │
│     - Calculate ES score                    │
│     - Categorize mutations (hotspot/non)    │
│     - Generate visualization               │
└─────────────────────────────────────────────┘
           ↓
Output: genes.txt.scores.txt → Web interface, benchmarking
```

## Code Conventions

### Naming Conventions
- **Variables**: `snake_case` (e.g., `smooth_kernel`, `mut_prob`, `lddt`)
- **Functions**: `snake_case` (e.g., `query_gene_coding_sequence`)
- **Classes**: `PascalCase` (e.g., `Gene`)
- **Gene names**: `UPPERCASE` (e.g., `ABL1`, `NT5C2`, `BRAF`)

### Data Patterns
- **Vectorized NumPy operations** for performance-critical computation
- **Pandas DataFrames** for tabular data manipulation
- **BioPython PDBParser** for structure file parsing
- **Feather format** (Apache Arrow) for large dataset serialization

### Mutation Categorization
- **Hotspots**: Recurrence > threshold (default 0.1 = 10%)
- **Non-hotspots**: Recurrence ≤ threshold
- **Not mutated**: Positions without observed mutations

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gap` | 5 | Secondary structure gap tolerance |
| `--interaction` | 15 | Angstrom threshold for 3D interactions |
| `--hotspot` | 0.1 | Recurrence threshold for hotspot definition |
| `--kernel` | 10 | Smoothing kernel size |
| `--smooth_method` | 'gaussian' | Smoothing method: 'gaussian' or 'convolution' |
| `--dimer` | False | Use dimer structure (for NT5C2, BCR-ABL1) |
| `--plot` | False | Generate visualization plots |

## Input File Formats

### pLDDT File (`9606.pLDDT.tdt`)
```
UniProtID    comma,separated,pLDDT,scores
P00519       85.2,87.1,89.3,91.0,...
```

### Genes File (`genes.txt`)
```
ABL1
BRAF
KRAS
```

### Mutations File (`mutations.txt`)
```
recurrence    gene_name    position
15            ABL1         315
8             BRAF         600
```

### Output File (`genes.txt.scores.txt`)
```
gene    position    ES_score    mutation_category
ABL1    1           0.234       not_mutated
ABL1    315         0.892       hotspot
```

## Development Workflow

### Adding a New Gene Analysis
1. Create input directory with `genes.txt` and `mutations.txt`
2. Run: `python plot.py ... input_dir genes.txt`
3. Output: `input_dir/genes.txt.scores.txt`

### Running Benchmarks

**1. COSMIC/OncoKB Benchmark** (ROC curves vs EVE, CTAT 3D):
```bash
cd benchmark
uv run python plot_roc.py
```

**2. ProteinGym Benchmark** (217 DMS assays, Spearman correlation):
```bash
cd benchmark/proteingym
uv run python run_benchmark.py --full --max_assays 50

# With AlphaMissense comparison (requires bulk data in /mnt/storage/alphamissense)
uv run python run_benchmark.py --full --max_assays 50 --include_alphamissense

# Download AlphaMissense bulk data first if needed (~1.2 GB)
python -m benchmark.alphamissense.fetcher --download --data_dir /mnt/storage/alphamissense
```

**3. Variant Annotation Benchmark** (MSK-IMPACT NSCLC, survival analysis):
```bash
# Requires survival extra for lifelines package
uv sync --extra survival
cd benchmark/variant_annotation
uv run python run_benchmark.py --full
```
Based on [clinical-data-mining/variant-annotation](https://github.com/clinical-data-mining/variant-annotation).
Reference: [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-63461-8)

**4. ClinVar Pathogenic Variant Benchmark** (pathogenic vs benign classification):
```bash
cd benchmark/clinvar
uv run python run_benchmark.py --full

# Cancer genes only (COSMIC Cancer Gene Census)
uv run python run_benchmark.py --full --cancer-genes-only

# Use ProteinGym pre-processed clinical data
uv run python run_benchmark.py --full --source proteingym

# Filter by minimum review status (2+ stars for higher confidence)
uv run python run_benchmark.py --full --min-stars 2
```
Uses ClinVar pathogenic/benign variant classifications as ground truth.
Key differences from vanilla ClinVar:
- Only pathogenic and benign variants (excludes VUS)
- Optional filtering by review status (1-4 stars)
- Focuses on missense variants for protein-level analysis
- Can use ProteinGym's curated clinical_substitutions (~500K variants)

### Web Development
```bash
cd website
uv run python app.py  # Dash server on port 4568
```

### AlphaMissense Predictions
```bash
# Download bulk data (once, ~1.2 GB to /mnt/storage)
python -m benchmark.alphamissense.fetcher --download --data_dir /mnt/storage/alphamissense

# Query a variant
python -m benchmark.alphamissense.fetcher --uniprot P00533 --variant L858R

# Or use Ensembl VEP API for small queries
python -m benchmark.alphamissense.ensembl_client --hgvs "ENST00000275493:c.2573T>G"
```

Python usage:
```python
from benchmark.alphamissense import AlphaMissenseFetcher, EnsemblVEPClient

# Bulk data (for large-scale analysis)
fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")
result = fetcher.get_score("P00533", "L858R")  # Returns (score, classification)

# Or Ensembl VEP API (for small queries)
client = EnsemblVEPClient()
result = client.get_alphamissense_hgvs("ENST00000275493:c.2573T>G")
```

## Dependencies

**Package Manager**: UV (pyproject.toml) - recommended
- `uv sync` - Install core dependencies
- `uv sync --extra survival` - Include lifelines for survival analysis
- `uv sync --extra dev` - Include development tools
- `uv sync --extra all` - Install all optional dependencies

**Legacy**: Conda (environment.yml) - still supported but not recommended

**Key Libraries**:
- **Bioinformatics**: BioPython (>=1.79), PyEnsembl (>=2.0)
- **Data**: pandas (>=1.4), numpy (>=1.22), scipy (>=1.8)
- **ML**: scikit-learn (>=1.0)
- **Visualization**: matplotlib (>=3.5), plotly (>=5.9), seaborn (>=0.11)
- **Web**: Dash (>=2.7)
- **Statistics**: statannot (>=0.2.3), statannotations (>=0.4)
- **Serialization**: PyArrow (>=8.0) for Feather format
- **Survival Analysis** (optional): lifelines (>=0.27)

**Python Version**: 3.10+

## Special Cases

### BCR-ABL1 Fusion Gene
- Not in standard AlphaFold database
- Handled via `bcr_abl1/` directory with custom pLDDT extraction
- Use `--dimer` flag for proper analysis

### NT5C2 Dimer
- Requires dimer structure consideration
- Analysis in `nt5c2_dimer/` directory
- Use `--dimer` flag

## Testing

Unit tests are available in the `tests/` directory. Run with:
```bash
uv run pytest tests/ -v
```

Test coverage includes:
- **ProteinGym benchmark**: `tests/test_proteingym.py` - 40 tests covering mutation parsing, DMS data loading, evaluation metrics (Spearman, AUC, MCC, NDCG, Top-K recall), ES scorer, and full pipeline integration
  - Uses dummy predictors to test pipeline without requiring large data files

Validation benchmarks:
- **COSMIC/OncoKB**: `benchmark/plot_roc.py` - ROC curves vs EVE, CTAT 3D
- **ProteinGym**: `benchmark/proteingym/` - Spearman correlation on 217 DMS assays
- **Clinical Validation**: `benchmark/variant_annotation/` - Survival analysis on MSK-IMPACT NSCLC
- **ClinVar Classification**: `benchmark/clinvar/` - AUC-ROC, AUC-PR, MCC for pathogenic/benign classification
- **Statistical tests**: Mann-Whitney U tests in `plot_gs_rank.py`

## Git Workflow

- **Large files**: `.feather` files tracked via Git LFS (see `.gitattributes`)
- **Branch naming**: Feature branches follow `claude/{feature}-{id}` pattern

## Common Tasks for AI Assistants

### When modifying ES score calculation:
1. Primary file: `plot.py` - the `Gene` class
2. Key methods: `get_final_score_gated_grad()`, `get_final_score_gated_grad_3d()`
3. Test changes against benchmark data in `rank_all_cosmic/`

### When updating visualization:
1. Static plots: `plot.py` - `plot_curve()` method
2. Web interface: `website/app.py`
3. Statistical figures: `plot_gs_rank.py`

### When adding new genes/mutations:
1. Update input files in target directory
2. Ensure UniProt mapping exists in `uniprot_to_genename.txt`
3. Run `plot.py` with appropriate parameters

### When debugging data issues:
1. Check feather file loading (requires Git LFS)
2. Verify pLDDT file format (tab-separated, comma-separated values)
3. Confirm gene name mappings

### When working with benchmarks:
1. **COSMIC/OncoKB**: `benchmark/plot_roc.py` - modify scoring methods or add new baselines
2. **ProteinGym**: `benchmark/proteingym/` - modular pipeline with separate loader, scorer, evaluator
3. **Variant Annotation**: `benchmark/variant_annotation/` - clinical validation with survival analysis
   - Requires `lifelines` package for full Cox PH analysis
   - Downloads data from [clinical-data-mining/variant-annotation](https://github.com/clinical-data-mining/variant-annotation)
   - Compares ES Score with 13 VEP predictors (SIFT, PolyPhen, CADD, REVEL, AlphaMissense, etc.)
4. **ClinVar**: `benchmark/clinvar/` - pathogenic/benign variant classification
   - Downloads from NCBI ClinVar or uses ProteinGym's curated clinical_substitutions
   - Evaluates AUC-ROC, AUC-PR, MCC, sensitivity, specificity
   - Optional filtering by review status (1-4 stars) and cancer genes (COSMIC CGC)
   - Comparison with ProteinGym clinical benchmark methodology

## External Resources

- **AlphaFold pLDDT data**: https://github.com/normandavey/ProcessedAlphafold
- **COSMIC database**: https://cancer.sanger.ac.uk/cosmic
- **ESM models**: https://github.com/facebookresearch/esm
- **ProteinGym**: https://proteingym.org/
- **Variant Annotation**: https://github.com/clinical-data-mining/variant-annotation
- **ClinVar database**: https://www.ncbi.nlm.nih.gov/clinvar/
