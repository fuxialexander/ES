#!/usr/bin/env python
"""
Generate mutations.txt from COSMIC data for the ES Score pipeline.

Input: CosmicMutantExport.missense.aachange.tsv (processed COSMIC data)
Output: mutations.txt (recurrence, gene, position format)

This script:
1. Reads the processed COSMIC missense mutation data
2. Counts recurrence per gene/position
3. Optionally filters by gene list
4. Outputs in the format expected by plot.py
"""

import pandas as pd
import argparse
import re
from pathlib import Path


def extract_position(mutation: str) -> int | None:
    """Extract position number from mutation string like 'A264V' -> 264"""
    match = re.match(r'^[A-Z](\d+)[A-Z]$', mutation)
    if match:
        return int(match.group(1))
    return None


def generate_mutations(
    cosmic_file: Path,
    output_file: Path,
    gene_list: Path | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate mutations.txt from COSMIC data.

    Args:
        cosmic_file: Path to CosmicMutantExport.missense.aachange.tsv
        output_file: Path to output mutations.txt
        gene_list: Optional path to genes.txt to filter by
        verbose: Print progress information

    Returns:
        DataFrame with the mutation data
    """
    if verbose:
        print(f"Reading COSMIC data from {cosmic_file}...")

    # Read COSMIC data (no header, columns: gene, enst_id, mut.rec, tumor, Mutation)
    df = pd.read_csv(
        cosmic_file, sep='\t', header=None,
        names=['gene', 'enst_id', 'mut_rec', 'tumor', 'Mutation']
    )

    if verbose:
        print(f"  Total records: {len(df):,}")
        print(f"  Unique genes: {df['gene'].nunique():,}")

    # Extract position from mutation
    df['position'] = df['Mutation'].apply(extract_position)
    df = df[df['position'].notna()].copy()
    df['position'] = df['position'].astype(int)

    if verbose:
        print(f"  After extracting positions: {len(df):,}")

    # If gene list provided, filter to only those genes
    if gene_list is not None and gene_list.exists():
        genes = pd.read_csv(gene_list, header=None, names=['gene'])['gene'].tolist()
        df = df[df['gene'].isin(genes)]
        if verbose:
            print(f"  After filtering to gene list: {len(df):,}")

    # Count recurrence per gene/position (across all samples)
    mutations = df.groupby(['gene', 'position']).size().reset_index(name='recurrence')

    if verbose:
        print(f"  Unique gene/position combinations: {len(mutations):,}")
        print(f"  Total mutation count: {mutations['recurrence'].sum():,}")

    # Sort by gene and position
    mutations = mutations.sort_values(['gene', 'position'])

    # Output format: recurrence, gene, position (tab-separated)
    mutations[['recurrence', 'gene', 'position']].to_csv(
        output_file, sep='\t', header=False, index=False
    )

    if verbose:
        print(f"  Saved to {output_file}")

    return mutations


def main():
    parser = argparse.ArgumentParser(
        description='Generate mutations.txt from COSMIC data')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('CosmicMutantExport.missense.aachange.v103.tsv'),
                        help='Input COSMIC processed file')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('rank_all_cosmic/mutations.v103.txt'),
                        help='Output mutations file')
    parser.add_argument('--genes', '-g', type=Path, default=None,
                        help='Optional gene list to filter by')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    generate_mutations(args.input, args.output, args.genes, verbose=not args.quiet)
    print("Generation complete!")


if __name__ == '__main__':
    main()
