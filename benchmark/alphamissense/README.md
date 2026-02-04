# AlphaMissense Prediction Fetcher

This module provides utilities to retrieve AlphaMissense pathogenicity predictions for variants using multiple data sources.

## Overview

AlphaMissense (Cheng et al., Science 2023) is a deep learning model that predicts the pathogenicity of all possible amino acid substitutions in the human proteome. This module provides:

1. **Bulk data loading** from Zenodo (recommended for large-scale analysis)
2. **Ensembl VEP REST API** integration (for small-scale queries)
3. **HegeLab web resource** client (for structure-based queries)

## Quick Start

```python
from benchmark.alphamissense import AlphaMissenseFetcher

# Initialize with bulk data directory
fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")

# Download bulk data (only needed once, ~1.2 GB)
fetcher.download_bulk_data()

# Query a variant
result = fetcher.get_score("P00533", "L858R")  # EGFR L858R
if result:
    score, classification = result
    print(f"Score: {score}, Class: {classification}")
```

## Data Sources

### 1. Zenodo Bulk Data (Recommended)

For large-scale analysis, download the pre-computed predictions from Zenodo:

```python
fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")
fetcher.download_bulk_data(files=["aa_substitutions", "gene_hg38"])
```

Available files:
- `aa_substitutions` (1.2 GB): 216M amino acid substitutions by UniProt ID
- `hg38` (643 MB): 71M SNV variants with hg38 genomic coordinates
- `hg19` (622 MB): 71M SNV variants with hg19 genomic coordinates
- `gene_hg38` (254 KB): Gene-level average scores
- `gene_hg19` (244 KB): Gene-level average scores

### 2. Ensembl VEP REST API

For small numbers of variants (<100), use the Ensembl VEP API:

```python
from benchmark.alphamissense import EnsemblVEPClient

client = EnsemblVEPClient()

# Query by HGVS notation
result = client.get_alphamissense_hgvs("ENST00000275493:c.2573T>G")
print(f"EGFR L858R: {result.score} ({result.classification})")

# Query by genomic coordinates
result = client.get_alphamissense_region("7", 140453136, "A", "T")

# Batch query
results = client.batch_query_hgvs([
    "ENST00000275493:c.2573T>G",
    "ENST00000256078:c.35G>A",
])
```

### 3. HegeLab Web Resource

For structure-based queries and PDB files with integrated scores:

```python
from benchmark.alphamissense import HegeLab_AMClient

client = HegeLab_AMClient()

# Download PDB with AlphaMissense scores in B-factor column
pdb_content = client.get_structure_pdb("P00533", output_path="EGFR_AM.pdb")

# Query hotspot API for a residue
result = client.get_hotspot("P00533", 858)
```

## Classification Thresholds

AlphaMissense provides three classifications based on pathogenicity scores:

| Classification | Score Range | Interpretation |
|---------------|-------------|----------------|
| `likely_benign` | < 0.34 | Likely benign variant |
| `ambiguous` | 0.34 - 0.564 | Uncertain significance |
| `likely_pathogenic` | > 0.564 | Likely pathogenic variant |

## Integration with Benchmark Pipeline

### Scoring Mutations

```python
import pandas as pd
from benchmark.alphamissense import AlphaMissenseFetcher

# Load your mutations
mutations = pd.DataFrame({
    "uniprot_id": ["P00533", "P00533", "P01116"],
    "protein_variant": ["L858R", "T790M", "G12V"],
})

# Score mutations
fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")
scored = fetcher.score_mutations_batch(
    mutations,
    uniprot_col="uniprot_id",
    variant_col="protein_variant",
)

print(scored[["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"]])
```

### Gene Name to UniProt Mapping

If using gene names instead of UniProt IDs:

```python
fetcher = AlphaMissenseFetcher(
    data_dir="/mnt/storage/alphamissense",
    uniprot_mapping_file="uniprot_to_genename.txt",
)

# Query by gene name
result = fetcher.get_score("EGFR", "L858R")
```

## Command-Line Interface

```bash
# Check available data
python -m benchmark.alphamissense.fetcher --check --data_dir /mnt/storage/alphamissense

# Download bulk data
python -m benchmark.alphamissense.fetcher --download --data_dir /mnt/storage/alphamissense

# Query a variant
python -m benchmark.alphamissense.fetcher --uniprot P00533 --variant L858R

# Get all variants for a protein
python -m benchmark.alphamissense.fetcher --uniprot P00533 --all

# Query via Ensembl VEP
python -m benchmark.alphamissense.ensembl_client --hgvs "ENST00000275493:c.2573T>G"
```

## File Formats

### Amino Acid Substitutions File

| Column | Description |
|--------|-------------|
| `uniprot_id` | UniProt accession (e.g., P00533) |
| `protein_variant` | Variant string (e.g., L858R) |
| `am_pathogenicity` | Pathogenicity score (0-1) |
| `am_class` | Classification (likely_benign/ambiguous/likely_pathogenic) |

### Genomic Coordinates File (hg38/hg19)

| Column | Description |
|--------|-------------|
| `CHROM` | Chromosome (e.g., chr7) |
| `POS` | Genomic position (1-based) |
| `REF` | Reference nucleotide |
| `ALT` | Alternate nucleotide |
| `uniprot_id` | UniProt accession |
| `transcript_id` | Ensembl transcript ID |
| `protein_variant` | Amino acid change |
| `am_pathogenicity` | Pathogenicity score |
| `am_class` | Classification |

## Citation

If you use AlphaMissense predictions, please cite:

> Cheng, J., Novati, G., Pan, J. et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense. Science 381, eadg7492 (2023). https://doi.org/10.1126/science.adg7492

## License

AlphaMissense predictions are licensed under Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Data Sources

- Zenodo: https://zenodo.org/records/10813168
- Ensembl VEP: https://rest.ensembl.org/documentation/info/vep_hgvs_get
- HegeLab: https://alphamissense.hegelab.org
