#!/usr/bin/env python3
"""
Generate ESM Log-Likelihood Ratio (LLR) scores for proteins.

This script computes ESM1b log-likelihood ratios for all possible single amino acid
substitutions at each position in a protein sequence. The LLR scores indicate the
predicted effect of each mutation based on evolutionary context.

Output formats:
1. UniProt format: {UNIPROT_ID}_LLR.csv - rows are amino acids, columns are positions
2. Gene format: {GENE_NAME}.csv - rows are individual mutations

Usage:
    python generate_esm_llr.py --gene ABL1 --output_dir /mnt/storage/es/data
    python generate_esm_llr.py --gene_list genes.txt --output_dir /mnt/storage/es/data
    python generate_esm_llr.py --uniprot P00519 --sequence MSEQ... --output_dir /mnt/storage/es/data
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)


def load_esm_model(model_name: str = "esm1b_t33_650M_UR50S", device: str = "cuda"):
    """
    Load ESM model and alphabet.

    Args:
        model_name: Name of the ESM model to load
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: Batch converter for sequences
    """
    import esm

    print(f"Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Model loaded on CPU")

    batch_converter = alphabet.get_batch_converter()

    return model, alphabet, batch_converter, device


def compute_llr_for_sequence(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str = "cuda",
    batch_size: int = 50,
    max_seq_len: int = 1022  # ESM1b has 1024 limit including BOS/EOS tokens
) -> np.ndarray:
    """
    Compute log-likelihood ratios for all possible mutations in a sequence.

    The LLR is computed as: log P(mutant) - log P(wildtype)
    where P is the ESM model's probability for the amino acid at that position.

    For sequences longer than max_seq_len, uses a sliding window approach
    with overlapping windows to ensure all positions are covered with good context.

    Args:
        sequence: Protein sequence (one-letter amino acid codes)
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: Device ('cuda' or 'cpu')
        batch_size: Number of positions to process at once
        max_seq_len: Maximum sequence length for the model

    Returns:
        LLR matrix of shape (20, seq_len) where rows are amino acids and columns are positions
    """
    seq_len = len(sequence)
    llr_matrix = np.zeros((20, seq_len))

    # Get the token indices for standard amino acids
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AMINO_ACIDS}

    if seq_len <= max_seq_len:
        # Sequence fits in one pass
        llr_matrix = _compute_llr_single_pass(
            sequence, model, alphabet, batch_converter, device, aa_to_idx
        )
    else:
        # Use sliding window for long sequences
        llr_matrix = _compute_llr_sliding_window(
            sequence, model, alphabet, batch_converter, device, aa_to_idx, max_seq_len
        )

    return llr_matrix


def _compute_llr_single_pass(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    aa_to_idx: Dict[str, int]
) -> np.ndarray:
    """Compute LLR for a sequence that fits within model limits."""
    seq_len = len(sequence)
    llr_matrix = np.zeros((20, seq_len))

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if device == "cuda":
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = results["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

        for pos in range(seq_len):
            token_pos = pos + 1  # Account for BOS token
            wt_aa = sequence[pos]
            if wt_aa not in AA_SET:
                continue

            wt_idx = aa_to_idx[wt_aa]
            wt_log_prob = log_probs[0, token_pos, wt_idx].item()

            for i, aa in enumerate(AMINO_ACIDS):
                mut_idx = aa_to_idx[aa]
                mut_log_prob = log_probs[0, token_pos, mut_idx].item()
                llr_matrix[i, pos] = mut_log_prob - wt_log_prob

    return llr_matrix


def _compute_llr_sliding_window(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    aa_to_idx: Dict[str, int],
    max_seq_len: int
) -> np.ndarray:
    """
    Compute LLR using sliding window for long sequences.

    Uses overlapping windows and averages predictions for positions
    that appear in multiple windows.
    """
    seq_len = len(sequence)
    llr_matrix = np.zeros((20, seq_len))
    count_matrix = np.zeros(seq_len)

    # Window parameters
    window_size = max_seq_len
    # Use larger overlap for better predictions at window edges
    overlap = window_size // 4  # 25% overlap
    step = window_size - overlap

    # Calculate number of windows needed
    n_windows = max(1, (seq_len - overlap) // step + 1)

    print(f"  Using sliding window: {n_windows} windows of size {window_size}")

    for win_idx in range(n_windows):
        start = win_idx * step
        end = min(start + window_size, seq_len)

        # Adjust start for the last window to ensure we cover the end
        if end == seq_len and end - start < window_size:
            start = max(0, end - window_size)

        window_seq = sequence[start:end]
        window_len = len(window_seq)

        data = [("protein", window_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if device == "cuda":
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)

            for win_pos in range(window_len):
                global_pos = start + win_pos
                token_pos = win_pos + 1

                wt_aa = sequence[global_pos]
                if wt_aa not in AA_SET:
                    continue

                wt_idx = aa_to_idx[wt_aa]
                wt_log_prob = log_probs[0, token_pos, wt_idx].item()

                for i, aa in enumerate(AMINO_ACIDS):
                    mut_idx = aa_to_idx[aa]
                    mut_log_prob = log_probs[0, token_pos, mut_idx].item()
                    llr_matrix[i, global_pos] += mut_log_prob - wt_log_prob

                count_matrix[global_pos] += 1

    # Average predictions for positions covered by multiple windows
    for pos in range(seq_len):
        if count_matrix[pos] > 0:
            llr_matrix[:, pos] /= count_matrix[pos]

    return llr_matrix


def compute_llr_masked(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute LLR using masked language model approach.

    For each position, mask that position and get the model's prediction.
    This is slower but may be more accurate for some applications.

    Args:
        sequence: Protein sequence
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: Device

    Returns:
        LLR matrix of shape (20, seq_len)
    """
    seq_len = len(sequence)
    llr_matrix = np.zeros((20, seq_len))

    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AMINO_ACIDS}
    mask_idx = alphabet.mask_idx

    for pos in tqdm(range(seq_len), desc="Computing masked LLR", leave=False):
        wt_aa = sequence[pos]
        if wt_aa not in AA_SET:
            continue

        # Create masked sequence
        masked_seq = sequence[:pos] + "<mask>" + sequence[pos + 1:]
        data = [("protein", masked_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if device == "cuda":
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)

            token_pos = pos + 1  # Account for BOS token
            wt_idx = aa_to_idx[wt_aa]
            wt_log_prob = log_probs[0, token_pos, wt_idx].item()

            for i, aa in enumerate(AMINO_ACIDS):
                mut_idx = aa_to_idx[aa]
                mut_log_prob = log_probs[0, token_pos, mut_idx].item()
                llr_matrix[i, pos] = mut_log_prob - wt_log_prob

    return llr_matrix


def get_sequence_from_uniprot(uniprot_id: str) -> Optional[str]:
    """
    Fetch protein sequence from UniProt.

    Args:
        uniprot_id: UniProt accession ID

    Returns:
        Protein sequence or None if not found
    """
    import requests

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            sequence = ''.join(lines[1:])
            return sequence
        else:
            print(f"Warning: Could not fetch sequence for {uniprot_id} (status {response.status_code})")
            return None
    except Exception as e:
        print(f"Warning: Error fetching sequence for {uniprot_id}: {e}")
        return None


def get_gene_to_uniprot_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load gene name to UniProt ID mapping."""
    mapping = {}
    if mapping_file.exists():
        df = pd.read_csv(mapping_file, sep='\t')
        if 'From' in df.columns and 'To' in df.columns:
            # UniProt -> Gene mapping, need to reverse
            mapping = df.set_index('To')['From'].to_dict()
        elif len(df.columns) >= 2:
            # Assume first column is gene, second is UniProt
            mapping = df.set_index(df.columns[0])[df.columns[1]].to_dict()
    return mapping


def fetch_uniprot_id_for_gene(gene_name: str) -> Optional[str]:
    """
    Fetch UniProt ID for a gene name using UniProt API.

    Args:
        gene_name: Gene symbol (e.g., "ABL1", "BRAF")

    Returns:
        UniProt ID or None if not found
    """
    import requests
    import time

    # Use UniProt REST API to search for the gene
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"(gene:{gene_name}) AND (organism_id:9606) AND (reviewed:true)",
        "format": "json",
        "size": 1,
        "fields": "accession,gene_names,length"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]["primaryAccession"]
        else:
            print(f"Warning: UniProt API returned status {response.status_code} for {gene_name}")
    except Exception as e:
        print(f"Warning: Error querying UniProt for {gene_name}: {e}")

    # Rate limiting
    time.sleep(0.5)
    return None


def build_gene_to_uniprot_mapping(
    genes: List[str],
    existing_mapping: Dict[str, str],
    output_file: Optional[Path] = None
) -> Dict[str, str]:
    """
    Build/extend gene to UniProt mapping by querying UniProt API.

    Args:
        genes: List of gene names
        existing_mapping: Existing mapping to extend
        output_file: Optional file to save the mapping

    Returns:
        Updated mapping dictionary
    """
    mapping = existing_mapping.copy()
    missing = [g for g in genes if g not in mapping]

    if missing:
        print(f"Fetching UniProt IDs for {len(missing)} genes...")
        for gene in tqdm(missing, desc="Fetching UniProt IDs"):
            uniprot_id = fetch_uniprot_id_for_gene(gene)
            if uniprot_id:
                mapping[gene] = uniprot_id

        if output_file:
            # Save extended mapping
            df = pd.DataFrame([
                {"From": uid, "To": gene}
                for gene, uid in mapping.items()
            ])
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Saved extended mapping to {output_file}")

    return mapping


def save_llr_uniprot_format(
    llr_matrix: np.ndarray,
    sequence: str,
    uniprot_id: str,
    output_dir: Path
):
    """
    Save LLR matrix in UniProt format.

    Format: rows are amino acids, columns are "REF position" (e.g., "M 1")
    """
    seq_len = len(sequence)
    columns = [f"{sequence[i]} {i+1}" for i in range(seq_len)]

    df = pd.DataFrame(
        llr_matrix,
        index=AMINO_ACIDS,
        columns=columns
    )

    output_path = output_dir / f"{uniprot_id}_LLR.csv"
    df.to_csv(output_path)
    print(f"Saved UniProt format: {output_path}")


def save_llr_gene_format(
    llr_matrix: np.ndarray,
    sequence: str,
    gene_name: str,
    output_dir: Path
):
    """
    Save LLR matrix in gene format compatible with ES Score pipeline.

    Format expected by plot.py and benchmark/es_scorer.py:
    - Column 0: Index
    - Column 1: Mutation (e.g., "M1A")
    - Column 2: Gene name
    - Columns 3+: ESM scores (the code does .mean(axis=1) on columns 3+)

    The code uses: esm['esm'] = esm.iloc[:,3:].astype(float).mean(axis=1)
    So we put the LLR score in column 3 (and could add more ensemble scores later).
    """
    rows = []
    seq_len = len(sequence)

    for pos in range(seq_len):
        wt_aa = sequence[pos]
        if wt_aa not in AA_SET:
            continue

        for i, mut_aa in enumerate(AMINO_ACIDS):
            if mut_aa == wt_aa:
                continue  # Skip wildtype

            mutation = f"{wt_aa}{pos+1}{mut_aa}"
            llr_score = llr_matrix[i, pos]

            rows.append({
                "Index": len(rows),
                "Mutation": mutation,
                "Gene": gene_name,
                "ESM1b_LLR": llr_score  # Score goes in column 3+
            })

    df = pd.DataFrame(rows)
    output_path = output_dir / f"{gene_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved gene format: {output_path}")


def process_gene(
    gene_name: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    output_dir: Path,
    gene_to_uniprot: Dict[str, str],
    use_masked: bool = False
) -> bool:
    """
    Process a single gene: get sequence, compute LLR, save results.

    Returns:
        True if successful, False otherwise
    """
    # Get UniProt ID
    uniprot_id = gene_to_uniprot.get(gene_name)
    if not uniprot_id:
        print(f"Warning: No UniProt ID found for {gene_name}")
        return False

    # Get sequence
    sequence = get_sequence_from_uniprot(uniprot_id)
    if not sequence:
        print(f"Warning: Could not get sequence for {gene_name} ({uniprot_id})")
        return False

    # Skip very long sequences (memory constraints)
    if len(sequence) > 2700:
        print(f"Warning: Sequence too long for {gene_name} ({len(sequence)} aa)")
        return False

    print(f"Processing {gene_name} ({uniprot_id}): {len(sequence)} aa")

    # Compute LLR
    if use_masked:
        llr_matrix = compute_llr_masked(sequence, model, alphabet, batch_converter, device)
    else:
        llr_matrix = compute_llr_for_sequence(sequence, model, alphabet, batch_converter, device)

    # Save in both formats
    uniprot_dir = output_dir / "esm1b_LLR"
    gene_dir = output_dir / "esm_ALL_hotspot"
    uniprot_dir.mkdir(parents=True, exist_ok=True)
    gene_dir.mkdir(parents=True, exist_ok=True)

    save_llr_uniprot_format(llr_matrix, sequence, uniprot_id, uniprot_dir)
    save_llr_gene_format(llr_matrix, sequence, gene_name, gene_dir)

    return True


def process_uniprot(
    uniprot_id: str,
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    output_dir: Path,
    use_masked: bool = False
) -> bool:
    """
    Process a protein by UniProt ID with provided sequence.
    """
    if len(sequence) > 2700:
        print(f"Warning: Sequence too long for {uniprot_id} ({len(sequence)} aa)")
        return False

    print(f"Processing {uniprot_id}: {len(sequence)} aa")

    if use_masked:
        llr_matrix = compute_llr_masked(sequence, model, alphabet, batch_converter, device)
    else:
        llr_matrix = compute_llr_for_sequence(sequence, model, alphabet, batch_converter, device)

    uniprot_dir = output_dir / "esm1b_LLR"
    uniprot_dir.mkdir(parents=True, exist_ok=True)

    save_llr_uniprot_format(llr_matrix, sequence, uniprot_id, uniprot_dir)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM LLR scores for proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--gene", type=str, help="Single gene name to process"
    )
    parser.add_argument(
        "--gene_list", type=str, help="File with list of gene names (one per line)"
    )
    parser.add_argument(
        "--uniprot", type=str, help="UniProt ID to process"
    )
    parser.add_argument(
        "--sequence", type=str, help="Protein sequence (required with --uniprot)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/storage/es/data",
        help="Output directory for LLR files"
    )
    parser.add_argument(
        "--mapping_file", type=str,
        help="Gene to UniProt mapping file (tab-separated)"
    )
    parser.add_argument(
        "--model", type=str, default="esm1b_t33_650M_UR50S",
        help="ESM model to use"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--masked", action="store_true",
        help="Use masked LLR computation (slower but more accurate)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip genes that already have LLR files"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.gene, args.gene_list, args.uniprot]):
        parser.error("Must specify --gene, --gene_list, or --uniprot")

    if args.uniprot and not args.sequence:
        # Try to fetch sequence
        print(f"Fetching sequence for {args.uniprot}...")
        args.sequence = get_sequence_from_uniprot(args.uniprot)
        if not args.sequence:
            parser.error(f"Could not fetch sequence for {args.uniprot}. Please provide --sequence")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, alphabet, batch_converter, device = load_esm_model(args.model, args.device)

    # Process based on input type
    if args.uniprot:
        success = process_uniprot(
            args.uniprot, args.sequence, model, alphabet, batch_converter,
            device, output_dir, args.masked
        )
        print(f"Processing {'succeeded' if success else 'failed'}")

    else:
        # Load gene to UniProt mapping
        if args.mapping_file:
            mapping_file = Path(args.mapping_file)
        else:
            # Try default location
            script_dir = Path(__file__).parent.parent
            mapping_file = script_dir / "uniprot_to_genename.txt"

        gene_to_uniprot = get_gene_to_uniprot_mapping(mapping_file)

        # Get list of genes
        if args.gene:
            genes = [args.gene]
        else:
            with open(args.gene_list, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]

        # Build/extend mapping for genes not in the file
        gene_to_uniprot = build_gene_to_uniprot_mapping(
            genes, gene_to_uniprot,
            output_file=output_dir / "gene_to_uniprot_mapping.tsv"
        )

        print(f"Processing {len(genes)} genes...")

        successful = 0
        failed = []

        for gene in tqdm(genes, desc="Processing genes"):
            # Check if already exists
            if args.skip_existing:
                gene_file = output_dir / "esm_ALL_hotspot" / f"{gene}.csv"
                if gene_file.exists():
                    print(f"Skipping {gene} (already exists)")
                    successful += 1
                    continue

            try:
                if process_gene(
                    gene, model, alphabet, batch_converter, device,
                    output_dir, gene_to_uniprot, args.masked
                ):
                    successful += 1
                else:
                    failed.append(gene)
            except Exception as e:
                print(f"Error processing {gene}: {e}")
                failed.append(gene)

        print(f"\nCompleted: {successful}/{len(genes)} genes processed successfully")
        if failed:
            print(f"Failed genes: {', '.join(failed[:20])}")
            if len(failed) > 20:
                print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
