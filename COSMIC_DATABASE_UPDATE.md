# COSMIC Database Update: v2015 to v103 (November 2024)

## Summary

The COSMIC mutation database has been updated from the February 2015 version to
COSMIC v103 (released October 2024). This is a critical update as the database
has grown substantially and many artifacts have been removed since 2015.

## Data Location

All COSMIC data files are stored in `/mnt/storage/ES/raw/` (not in git):

### Processed Mutation Files
- `mutations.txt.2015` - Original 2015 mutations (backup)
- `mutations.txt.v103` - New v103 mutations (to use, copy to `rank_all_cosmic/mutations.txt`)
- `CosmicMutantExport.missense.aachange.tsv` - Combined missense mutations

### Raw COSMIC v103 Files
- `Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz` - Genome-wide WGS/WES data
- `Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv.gz` - Gene panel data
- `Cosmic_MutantCensus_v103_GRCh38.tsv.gz` - Cancer Gene Census only

## Quick Start

To use the new v103 data:

```bash
# Copy v103 mutations to the working directory
cp /mnt/storage/ES/raw/mutations.txt.v103 rank_all_cosmic/mutations.txt

# Then run the ES Score pipeline
python plot.py ...
```

## Data Growth

| Metric | Old (2015) | New (v103) | Change |
|--------|------------|------------|--------|
| Total mutations | 4.8M | 14.9M | +210% |
| Unique positions | 2.8M | 4.3M | +51% |
| Unique genes | 19,431 | 19,458 | +0.1% |

### Key Oncogene Mutation Counts
| Gene | Old (2015) | New (v103) | Change |
|------|------------|------------|--------|
| TP53 | 34,573 | 621,197 | +1697% |
| KRAS | 48,297 | 193,934 | +302% |
| BRAF | 57,210 | 136,531 | +139% |
| ABL1 | 1,011 | 2,316 | +129% |
| EGFR | 16,860 | 23,137 | +37% |
| PIK3CA | 19,169 | 24,958 | +30% |
| NRAS | 8,166 | 8,397 | +3% |

## Processing Scripts

### `process_cosmic_v103.py`
Process a single COSMIC v103 file to the expected format.

```bash
python process_cosmic_v103.py \
    --input /mnt/storage/ES/raw/Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz \
    --output /mnt/storage/ES/raw/CosmicMutantExport.missense.aachange.tsv
```

### `process_cosmic_v103_combined.py`
Combine GenomeScreens + TargetedScreens (positive) to recreate full Cosmic_Mutant.

```bash
python process_cosmic_v103_combined.py \
    --genome /mnt/storage/ES/raw/Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz \
    --targeted /mnt/storage/ES/raw/Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv.gz \
    --output /mnt/storage/ES/raw/CosmicMutantExport.missense.aachange.tsv
```

### `generate_mutations_from_cosmic.py`
Generate mutations.txt for the ES Score pipeline.

```bash
python generate_mutations_from_cosmic.py \
    --input /mnt/storage/ES/raw/CosmicMutantExport.missense.aachange.tsv \
    --output /mnt/storage/ES/raw/mutations.txt.v103
```

### `update_cosmic_database.py`
Full update script (download + process).

```bash
# Set credentials
export COSMIC_EMAIL="your@email.com"
export COSMIC_PASSWORD="yourpassword"

# Run full update
python update_cosmic_database.py

# Or skip download if files exist
python update_cosmic_database.py --skip-download
```

## COSMIC v103 Data Structure

COSMIC v103 splits mutation data into multiple files:
- **GenomeScreensMutant**: WGS/WES from genome-wide screens
- **CompleteTargetedScreensMutant**: Gene panel sequencing (includes POSITIVE_SCREEN column)
- **MutantCensus**: Only Cancer Gene Census genes

To recreate the equivalent of the old `CosmicMutantExport`:
1. Take all data from GenomeScreensMutant
2. Add positive screens (POSITIVE_SCREEN='y') from CompleteTargetedScreensMutant

## Download Instructions

COSMIC requires registration and authentication:

1. Generate auth string:
```bash
echo 'email@example.com:password' | base64
```

2. Get download URL:
```bash
curl -H "Authorization: Basic <auth_string>" \
    "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted?path=grch38/cosmic/v103/Cosmic_GenomeScreensMutant_Tsv_v103_GRCh38.tar&bucket=downloads"
```

3. Download using returned URL (valid for 1 hour):
```bash
curl "<download_url>" --output Cosmic_GenomeScreensMutant_Tsv_v103_GRCh38.tar
```

## Next Steps After Update

1. Copy v103 mutations: `cp /mnt/storage/ES/raw/mutations.txt.v103 rank_all_cosmic/mutations.txt`
2. Regenerate ES scores: `python plot.py ...`
3. Regenerate cosmic.feather: `python plot_gs_rank.py`
4. Validate with benchmarks: `python benchmark/plot_roc.py`

## Notes

- The transition matrix (`cosmic_aa_transition.csv`) was not updated as it
  requires CCDS sequence files. The nucleotide-to-amino-acid mutation
  probabilities are relatively stable across database versions.
- Gene coverage remains the same (19,431 → 19,458 genes)
- Some genes removed in data curation (H3-3A, H1-4, etc.) remain missing
- Data files are excluded from git to avoid large file issues
