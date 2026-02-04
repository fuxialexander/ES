#!/usr/bin/env python
"""
Process COSMIC v103 combined data (GenomeScreens + TargetedScreens) to match
the old CosmicMutantExport format.

According to COSMIC documentation:
"The Cosmic_Mutant file can be re-created by linking GenomeScreensMutant
with the positive data (POSITIVE_SCREEN='y') from CompleteTargetedScreensMutant"

Input:
    - Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz
    - Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv.gz

Output:
    - CosmicMutantExport.missense.aachange.tsv (combined mutations)
"""

import gzip
import pandas as pd
import re
import argparse
from pathlib import Path


def parse_mutation_aa(mutation_aa: str) -> str | None:
    """Parse MUTATION_AA field to extract simple amino acid change."""
    if pd.isna(mutation_aa) or mutation_aa == "p.?" or not mutation_aa.startswith("p."):
        return None

    change = mutation_aa[2:]

    if re.match(r'^[A-Z]\d+[A-Z]$', change):
        return change

    aa_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    match = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', change)
    if match:
        ref_3letter, pos, alt_3letter = match.groups()
        ref = aa_map.get(ref_3letter)
        alt = aa_map.get(alt_3letter)
        if ref and alt:
            return f"{ref}{pos}{alt}"

    return None


def process_cosmic_combined(
    genome_file: Path,
    targeted_file: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process combined COSMIC v103 files.

    Args:
        genome_file: Path to Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz
        targeted_file: Path to Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv.gz
        output_file: Path to output CosmicMutantExport.missense.aachange.tsv
        verbose: Print progress information

    Returns:
        DataFrame with the combined processed data
    """
    usecols = ['GENE_SYMBOL', 'TRANSCRIPT_ACCESSION', 'SAMPLE_NAME',
               'MUTATION_AA', 'MUTATION_DESCRIPTION']

    # Process GenomeScreensMutant
    if verbose:
        print(f"Reading Genome Screens from {genome_file}...")

    df_genome = pd.read_csv(
        genome_file, sep='\t', compression='gzip', usecols=usecols,
        dtype={'GENE_SYMBOL': str, 'TRANSCRIPT_ACCESSION': str,
               'SAMPLE_NAME': str, 'MUTATION_AA': str,
               'MUTATION_DESCRIPTION': str}
    )
    df_genome = df_genome[df_genome['MUTATION_DESCRIPTION'] == 'missense_variant'].copy()
    if verbose:
        print(f"  Genome Screens missense: {len(df_genome):,}")

    # Process CompleteTargetedScreensMutant (only positive screens)
    if verbose:
        print(f"Reading Targeted Screens from {targeted_file}...")

    usecols_targeted = usecols + ['POSITIVE_SCREEN']
    df_targeted = pd.read_csv(
        targeted_file, sep='\t', compression='gzip', usecols=usecols_targeted,
        dtype={'GENE_SYMBOL': str, 'TRANSCRIPT_ACCESSION': str,
               'SAMPLE_NAME': str, 'MUTATION_AA': str,
               'MUTATION_DESCRIPTION': str, 'POSITIVE_SCREEN': str}
    )
    # Filter for positive screens only
    df_targeted = df_targeted[df_targeted['POSITIVE_SCREEN'] == 'y'].copy()
    df_targeted = df_targeted[df_targeted['MUTATION_DESCRIPTION'] == 'missense_variant'].copy()
    df_targeted = df_targeted.drop(columns=['POSITIVE_SCREEN'])
    if verbose:
        print(f"  Targeted Screens missense (positive): {len(df_targeted):,}")

    # Combine both datasets
    if verbose:
        print("Combining datasets...")
    df = pd.concat([df_genome, df_targeted], ignore_index=True)
    if verbose:
        print(f"  Combined total: {len(df):,}")

    # Parse mutations
    df['Mutation'] = df['MUTATION_AA'].apply(parse_mutation_aa)
    df = df[df['Mutation'].notna()].copy()
    if verbose:
        print(f"  After parsing AA changes: {len(df):,}")

    # Extract ENST ID without version
    df['enst_id'] = df['TRANSCRIPT_ACCESSION'].str.split('.').str[0]

    # Create output dataframe
    output_df = pd.DataFrame({
        'gene.cosmic': df['GENE_SYMBOL'],
        'enst_id': df['enst_id'],
        'mut.rec': 1,
        'tumor': df['SAMPLE_NAME'],
        'Mutation': df['Mutation']
    })

    # Remove duplicates
    output_df = output_df.drop_duplicates()
    if verbose:
        print(f"  After removing duplicates: {len(output_df):,}")

    # Sort
    output_df = output_df.sort_values(['gene.cosmic', 'enst_id', 'Mutation'])

    # Save
    output_df.to_csv(output_file, sep='\t', header=False, index=False)
    if verbose:
        print(f"  Saved to {output_file}")

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Process COSMIC v103 combined data')
    parser.add_argument('--genome', '-g', type=Path,
                        default=Path('/mnt/storage/ES/raw/Cosmic_GenomeScreensMutant_v103_GRCh38.tsv.gz'),
                        help='Genome Screens file')
    parser.add_argument('--targeted', '-t', type=Path,
                        default=Path('/mnt/storage/ES/raw/Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv.gz'),
                        help='Complete Targeted Screens file')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('CosmicMutantExport.missense.aachange.tsv'),
                        help='Output file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    if not args.genome.exists():
        raise FileNotFoundError(f"Genome Screens file not found: {args.genome}")
    if not args.targeted.exists():
        raise FileNotFoundError(f"Targeted Screens file not found: {args.targeted}")

    process_cosmic_combined(args.genome, args.targeted, args.output, verbose=not args.quiet)
    print("Processing complete!")


if __name__ == '__main__':
    main()
