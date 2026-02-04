#!/usr/bin/env python
"""
Process COSMIC v103 MutantCensus data to the format expected by the ES Score pipeline.

Input: Cosmic_MutantCensus_v103_GRCh38.tsv.gz
Output: CosmicMutantExport.missense.aachange.tsv (compatible with existing pipeline)

The old format had columns (no header):
    0: gene.cosmic (gene name)
    1: enst_id (Ensembl transcript ID)
    2: mut.rec (mutation record - originally just a placeholder, used for counting)
    3: tumor (tumor/sample name)
    4: Mutation (amino acid change, e.g., "A264V")

The new COSMIC v103 format has proper headers with columns:
    GENE_SYMBOL, TRANSCRIPT_ACCESSION, MUTATION_AA, MUTATION_DESCRIPTION, etc.

This script extracts missense mutations and converts to the old format.
"""

import gzip
import pandas as pd
import re
import argparse
from pathlib import Path


def parse_mutation_aa(mutation_aa: str) -> str | None:
    """
    Parse MUTATION_AA field to extract simple amino acid change.

    Input formats:
        - "p.S296T" -> "S296T"
        - "p.Ser296Thr" -> "S296T" (need to convert 3-letter to 1-letter)
        - "p.?" -> None (invalid)

    Returns None if the mutation cannot be parsed.
    """
    if pd.isna(mutation_aa) or mutation_aa == "p.?" or not mutation_aa.startswith("p."):
        return None

    # Remove "p." prefix
    change = mutation_aa[2:]

    # If already in single-letter format (e.g., "S296T")
    if re.match(r'^[A-Z]\d+[A-Z]$', change):
        return change

    # Convert 3-letter amino acid codes to 1-letter
    aa_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    # Match 3-letter format (e.g., "Ser296Thr")
    match = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', change)
    if match:
        ref_3letter, pos, alt_3letter = match.groups()
        ref = aa_map.get(ref_3letter)
        alt = aa_map.get(alt_3letter)
        if ref and alt:
            return f"{ref}{pos}{alt}"

    return None


def process_cosmic_v103(input_file: Path, output_file: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Process COSMIC v103 MutantCensus file to the old format.

    Args:
        input_file: Path to Cosmic_MutantCensus_v103_GRCh38.tsv.gz
        output_file: Path to output CosmicMutantExport.missense.aachange.tsv
        verbose: Print progress information

    Returns:
        DataFrame with the processed data
    """
    if verbose:
        print(f"Reading COSMIC v103 data from {input_file}...")

    # Read the gzipped TSV file
    # Only read columns we need to save memory
    usecols = ['GENE_SYMBOL', 'TRANSCRIPT_ACCESSION', 'SAMPLE_NAME',
               'MUTATION_AA', 'MUTATION_DESCRIPTION']

    df = pd.read_csv(input_file, sep='\t', compression='gzip', usecols=usecols,
                     dtype={'GENE_SYMBOL': str, 'TRANSCRIPT_ACCESSION': str,
                            'SAMPLE_NAME': str, 'MUTATION_AA': str,
                            'MUTATION_DESCRIPTION': str})

    if verbose:
        print(f"  Total records: {len(df):,}")

    # Filter for missense variants only
    df = df[df['MUTATION_DESCRIPTION'] == 'missense_variant'].copy()
    if verbose:
        print(f"  Missense variants: {len(df):,}")

    # Parse MUTATION_AA to get simple amino acid change format
    df['Mutation'] = df['MUTATION_AA'].apply(parse_mutation_aa)

    # Remove rows where we couldn't parse the mutation
    df = df[df['Mutation'].notna()].copy()
    if verbose:
        print(f"  After parsing AA changes: {len(df):,}")

    # Extract ENST ID without version suffix
    df['enst_id'] = df['TRANSCRIPT_ACCESSION'].str.split('.').str[0]

    # Create output dataframe in old format
    # Columns: gene.cosmic, enst_id, mut.rec (placeholder=1), tumor, Mutation
    output_df = pd.DataFrame({
        'gene.cosmic': df['GENE_SYMBOL'],
        'enst_id': df['enst_id'],
        'mut.rec': 1,  # Placeholder, will be counted during analysis
        'tumor': df['SAMPLE_NAME'],
        'Mutation': df['Mutation']
    })

    # Remove duplicates (same mutation in same sample for same transcript)
    output_df = output_df.drop_duplicates()
    if verbose:
        print(f"  After removing duplicates: {len(output_df):,}")

    # Sort by gene and position for easier debugging
    output_df = output_df.sort_values(['gene.cosmic', 'enst_id', 'Mutation'])

    # Save without header (to match old format)
    output_df.to_csv(output_file, sep='\t', header=False, index=False)
    if verbose:
        print(f"  Saved to {output_file}")

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Process COSMIC v103 MutantCensus data for ES Score pipeline')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('/mnt/storage/ES/raw/Cosmic_MutantCensus_v103_GRCh38.tsv.gz'),
                        help='Input COSMIC v103 file')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('CosmicMutantExport.missense.aachange.tsv'),
                        help='Output file in old COSMIC format')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    process_cosmic_v103(args.input, args.output, verbose=not args.quiet)
    print("Processing complete!")


if __name__ == '__main__':
    main()
