#!/usr/bin/env python3
"""
Download and organize cancer gene lists from multiple sources.

Sources:
1. OncoKB Cancer Gene List (API v1)
2. COSMIC Cancer Gene Census (placeholder with known genes)
3. Bailey et al. 2018 driver genes (Cell paper, supplementary data)

This script downloads the latest versions and saves them to /mnt/storage/ES/raw/
in a standardized format.

Usage:
    python download_gene_lists.py [--output-dir /mnt/storage/ES/raw/]

Note: This version uses only standard library (no pandas dependency).
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


# Default output directory
DEFAULT_OUTPUT_DIR = "/mnt/storage/ES/raw/"

# OncoKB API endpoint
ONCOKB_CANCER_GENES_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList"

# Bailey et al. 2018 driver genes from Cell paper supplementary data
# Original DOI: 10.1016/j.cell.2018.02.060
# These are the 299 driver genes from Table S1
BAILEY_2018_DRIVER_GENES = [
    # Tier 1 genes (high confidence drivers from 26 computational tools)
    "TP53", "PIK3CA", "PTEN", "KRAS", "APC", "NRAS", "BRAF", "CDKN2A",
    "FBXW7", "SMAD4", "CTNNB1", "ARID1A", "RB1", "EGFR", "ATM", "NFE2L2",
    "KMT2D", "KMT2C", "SETD2", "ATRX", "NF1", "CREBBP", "KEAP1", "SMARCA4",
    "STK11", "FAT1", "IDH1", "ERBB2", "NOTCH1", "CDKN1A", "PIK3R1", "HRAS",
    "PTPN11", "GNAS", "BAP1", "VHL", "BCOR", "HIST1H3B", "TGFBR2", "KIT",
    "MAP3K1", "MTOR", "FGFR2", "ERBB3", "FGFR3", "CASP8", "NF2", "RAC1",
    "CDKN1B", "POLE", "MAP2K1", "PPP2R1A", "AXIN1", "ZFHX3", "RNF43", "CDK12",
    "SMO", "MGA", "PBRM1", "PTCH1", "ARID2", "CARD11", "MYD88", "SPOP",
    "FOXA1", "EZH2", "HIST1H3C", "CYLD", "ASXL1", "SF3B1", "EP300", "H3F3A",
    "CIC", "MLH1", "RHOA", "MSH2", "DNMT3A", "B2M", "JAK1", "GATA3",
    "STAG2", "PRKAR1A", "EPHA2", "RUNX1", "CDH1", "ELF3", "SOX9", "PCBP1",
    "KLF5", "SMAD2", "RASA1", "ACVR2A", "RXRA", "MAP2K4", "MSH6", "POLQ",
    "ARID1B", "PHF6", "PDGFRA", "CHEK2", "CBL", "JAK2", "SPEN", "KDM6A",
    "RHOB", "CACNA1A", "NFE2L3", "POLR2A", "IDH2", "TRAF7", "PRDM1", "AKT1",
    "PPM1D", "FOXQ1", "ETNK1", "RPL22", "U2AF1", "CTCF", "XPO1", "ZMYM3",
    "EZR", "BRCA2", "RPL5", "TBX3", "BRCA1", "BIRC3", "DDX3X", "RAD21",
    "LATS1", "LATS2", "NOTCH2", "MED12", "KDM5C", "FGFR1", "MAX", "CTNNA1",
    "SOS1", "FLT3", "FUBP1", "BTG1", "EPHA3", "HGF", "MEN1", "MYCN",
    "QKI", "KMT2A", "SMAD3", "EPHA5", "CNOT3", "RANBP2", "CALR",
    "MPL", "TET2", "WT1", "CEBPA", "NPM1", "KLF6", "TAF1", "RET",
    "SRSF2", "BCORL1", "BTG2", "RPS15", "IKZF1", "STAT3", "GNAQ", "GNA11",
    "PNRC1", "PTPRD", "CUX1", "TNFAIP3", "DICER1", "DROSHA", "DGCR8", "ERCC2",
    "ERCC3", "ERCC4", "ERCC5", "XPA", "EPAS1", "FLCN", "LIFR", "TGFBR1",
    "ACVR1", "BMP5", "BMPR2", "FH", "TSC1", "TSC2", "SDHA",
    "SDHAF2", "SDHB", "SDHC", "SDHD", "SUFU", "GLI1", "GLI2", "SMARCB1",
    "DOT1L", "NSD1", "NSD2", "NSD3", "SETBP1", "SETDB1", "SUZ12", "EED",
    "JARID2", "MTF2", "PHF19", "CDKN2B", "CDKN2C", "CDK4", "CDK6", "CCND1",
    "CCND2", "CCND3", "CCNE1", "MDM2", "MDM4", "MYC", "MYCL", "BCL2",
    "BCL2L1", "BCL2L11", "MCL1", "BCL6", "PMAIP1", "BAX", "BAK1", "BID",
    "BMF", "HRK", "TNFRSF10A", "TNFRSF10B", "CASP3", "CASP7", "CASP9",
    "APAF1", "XIAP", "BIRC2", "BIRC5", "CFLAR", "FADD", "TRADD",
    "RIPK1", "FAS", "TNFRSF1A", "TNF", "TRAF2", "TRAF3", "TRAF5",
    "TRAF6", "TAB1", "TAB2", "MAP3K7", "IKBKB", "IKBKG", "NFKBIA",
    "NFKB1", "NFKB2", "RELA", "RELB", "REL", "CHUK"
]

# COSMIC Cancer Gene Census - well-established cancer genes (Tier 1)
COSMIC_CGC_GENES = [
    "ABL1", "ABL2", "ACVR1", "ACVR2A", "AKT1", "AKT2", "ALK", "AMER1",
    "APC", "AR", "ARAF", "ARID1A", "ARID1B", "ARID2", "ASXL1", "ATM",
    "ATRX", "AXIN1", "AXIN2", "BAP1", "BCL10", "BCL2", "BCL6", "BCOR",
    "BCORL1", "BCR", "BIRC3", "BLM", "BMPR1A", "BRAF", "BRCA1", "BRCA2",
    "BRD4", "BTG1", "BTK", "CALR", "CARD11", "CASP8", "CBL", "CBLB",
    "CCND1", "CCND2", "CCND3", "CCNE1", "CD274", "CD79A", "CD79B", "CDC73",
    "CDH1", "CDK12", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B",
    "CDKN2C", "CEBPA", "CHD4", "CHEK2", "CIC", "CREBBP", "CRLF2", "CSF1R",
    "CSF3R", "CTCF", "CTLA4", "CTNNB1", "CUX1", "CXCR4", "CYLD", "DAXX",
    "DDB2", "DDIT3", "DDX3X", "DDX41", "DICER1", "DIS3", "DNMT3A", "DOT1L",
    "EGFR", "EIF1AX", "EIF4A2", "ELF3", "EP300", "EPAS1", "EPCAM", "EPHA3",
    "EPHB1", "ERBB2", "ERBB3", "ERBB4", "ERCC2", "ERCC3", "ERCC4", "ERG",
    "ESR1", "ETNK1", "ETV1", "ETV6", "EWSR1", "EZH2", "FAM46C", "FANCA",
    "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FAS", "FAT1", "FBXO11",
    "FBXW7", "FGF19", "FGF3", "FGF4", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    "FH", "FLCN", "FLI1", "FLT3", "FOXL2", "FOXO1", "FOXP1", "FUBP1",
    "FUS", "GATA1", "GATA2", "GATA3", "GLI1", "GNA11", "GNAQ", "GNAS",
    "GPS2", "GREM1", "H3F3A", "HIST1H3B", "HNF1A", "HRAS", "ID3", "IDH1",
    "IDH2", "IGF1R", "IKBKB", "IKZF1", "IL2", "IL21R", "IL6ST", "IL7R",
    "IRF4", "IRS2", "JAK1", "JAK2", "JAK3", "JUN", "KAT6A", "KDM5A",
    "KDM5C", "KDM6A", "KDR", "KEAP1", "KIT", "KLF4", "KMT2A", "KMT2B",
    "KMT2C", "KMT2D", "KRAS", "LATS1", "LATS2", "LMO1", "LMO2", "LZTR1",
    "MALT1", "MAP2K1", "MAP2K2", "MAP2K4", "MAP3K1", "MAP3K13", "MAPK1",
    "MAX", "MCL1", "MDM2", "MDM4", "MED12", "MEN1", "MET", "MGA", "MITF",
    "MLH1", "MPL", "MSH2", "MSH6", "MTOR", "MUTYH", "MYC", "MYCL", "MYCN",
    "MYD88", "MYOD1", "NBN", "NCOR1", "NF1", "NF2", "NFE2L2", "NFKBIA",
    "NKX2-1", "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "NPM1", "NRAS", "NSD1",
    "NSD2", "NSD3", "NT5C2", "NTRK1", "NTRK2", "NTRK3", "P2RY8", "PAK1",
    "PALB2", "PAX5", "PAX8", "PBRM1", "PCBP1", "PDCD1", "PDCD1LG2", "PDGFRA",
    "PDGFRB", "PHF6", "PHOX2B", "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2",
    "PIM1", "PLK2", "PMS1", "PMS2", "POLD1", "POLE", "POLQ", "POT1",
    "POU2AF1", "PPM1D", "PPP2R1A", "PRDM1", "PRDM14", "PREX2", "PRKAR1A",
    "PRKDC", "PTCH1", "PTEN", "PTPN11", "PTPRD", "PTPRT", "QKI", "RAC1",
    "RAD21", "RAD51B", "RAD51C", "RAD51D", "RAF1", "RARA", "RASA1", "RB1",
    "RBM10", "RECQL4", "REL", "RET", "RFWD2", "RHOA", "RHOB", "RIT1",
    "RNF43", "ROS1", "RPL10", "RPL22", "RPL5", "RRAS2", "RUNX1", "RUNX1T1",
    "SBDS", "SDHA", "SDHAF2", "SDHB", "SDHC", "SDHD", "SETBP1", "SETD2",
    "SF3B1", "SGK1", "SMAD2", "SMAD3", "SMAD4", "SMARCA4", "SMARCB1", "SMARCE1",
    "SMC1A", "SMC3", "SMO", "SOCS1", "SOS1", "SOX2", "SOX9", "SPEN",
    "SPOP", "SRSF2", "STAG2", "STAT3", "STAT5B", "STK11", "STK19", "SUFU",
    "SUZ12", "SYK", "TAL1", "TBX3", "TCEB1", "TCF3", "TCF7L2", "TET1",
    "TET2", "TFRC", "TGFBR1", "TGFBR2", "TMEM127", "TMPRSS2", "TNFAIP3",
    "TNFRSF14", "TOP1", "TP53", "TP63", "TRAF7", "TSC1", "TSC2", "TSHR",
    "U2AF1", "VHL", "WAS", "WHSC1", "WRN", "WT1", "WWTR1", "XPA", "XPC",
    "XPO1", "ZBTB16", "ZNF217", "ZNF703", "ZRSR2"
]


def download_oncokb_cancer_genes(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Download the latest OncoKB cancer gene list from the API.

    Returns:
        List of gene dictionaries
    """
    print("Downloading OncoKB cancer gene list...")

    try:
        with urllib.request.urlopen(ONCOKB_CANCER_GENES_URL, timeout=30) as response:
            data = json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"Error downloading OncoKB data: {e}")
        return []

    # Save raw JSON
    json_path = output_dir / "oncokb_cancer_genes_raw.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved raw JSON to {json_path}")

    # Save as TSV
    if data:
        tsv_path = output_dir / "oncokb_cancer_genes.tsv"
        fieldnames = list(data[0].keys())
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved TSV to {tsv_path}")

    # Create a simplified gene list
    genes = sorted(set(entry.get("hugoSymbol", "") for entry in data if entry.get("hugoSymbol")))
    genes_only_path = output_dir / "oncokb_cancer_genes_list.txt"
    with open(genes_only_path, "w") as f:
        for gene in genes:
            f.write(f"{gene}\n")
    print(f"  Saved gene list to {genes_only_path}")

    print(f"  Downloaded {len(data)} genes from OncoKB")
    return data


def create_bailey_driver_genes(output_dir: Path) -> List[str]:
    """
    Create the Bailey et al. 2018 driver genes list.

    Reference: Bailey et al. (2018) Cell 173, 371-385.e18
    DOI: 10.1016/j.cell.2018.02.060

    Returns:
        List of gene names
    """
    print("Creating Bailey et al. 2018 driver genes list...")

    # Remove duplicates and sort
    unique_genes = sorted(set(BAILEY_2018_DRIVER_GENES))

    # Save as TSV
    tsv_path = output_dir / "bailey_2018_driver_genes.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["hugoSymbol", "source", "reference"])
        for gene in unique_genes:
            writer.writerow([gene, "Bailey_2018_Cell", "10.1016/j.cell.2018.02.060"])
    print(f"  Saved TSV to {tsv_path}")

    # Save gene list only
    genes_only_path = output_dir / "bailey_2018_driver_genes_list.txt"
    with open(genes_only_path, "w") as f:
        for gene in unique_genes:
            f.write(f"{gene}\n")
    print(f"  Saved gene list to {genes_only_path}")

    print(f"  Created list with {len(unique_genes)} driver genes")
    return unique_genes


def create_cosmic_cgc(output_dir: Path) -> List[str]:
    """
    Create COSMIC Cancer Gene Census gene list.

    Note: Full CGC requires registration at https://cancer.sanger.ac.uk/cosmic/download
    This creates a placeholder with well-established cancer genes.

    Returns:
        List of gene names
    """
    print("Creating COSMIC Cancer Gene Census gene list...")
    print("  Note: This is a curated subset. Download full CGC from:")
    print("  https://cancer.sanger.ac.uk/cosmic/download (requires registration)")

    # Remove duplicates and sort
    unique_genes = sorted(set(COSMIC_CGC_GENES))

    # Save as TSV
    tsv_path = output_dir / "cosmic_cgc_genes.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["hugoSymbol", "source", "note"])
        for gene in unique_genes:
            writer.writerow([gene, "COSMIC_CGC", "Curated Tier 1 genes"])
    print(f"  Saved TSV to {tsv_path}")

    # Save gene list only
    genes_only_path = output_dir / "cosmic_cgc_genes_list.txt"
    with open(genes_only_path, "w") as f:
        for gene in unique_genes:
            f.write(f"{gene}\n")
    print(f"  Saved gene list to {genes_only_path}")

    print(f"  Created list with {len(unique_genes)} CGC genes")
    return unique_genes


def create_combined_gene_list(
    oncokb_data: List[Dict[str, Any]],
    bailey_genes: List[str],
    cosmic_genes: List[str],
    output_dir: Path
) -> Dict[str, Set[str]]:
    """
    Create a combined/union gene list from all sources.

    Returns:
        Dictionary mapping genes to their sources
    """
    print("Creating combined gene list...")

    # Collect all genes with their sources
    gene_sources: Dict[str, Set[str]] = {}

    # Add OncoKB genes
    oncokb_genes = [entry.get("hugoSymbol", "") for entry in oncokb_data if entry.get("hugoSymbol")]
    for gene in oncokb_genes:
        gene_sources.setdefault(gene, set()).add("OncoKB")

    # Add Bailey genes
    for gene in bailey_genes:
        gene_sources.setdefault(gene, set()).add("Bailey_2018")

    # Add COSMIC genes
    for gene in cosmic_genes:
        gene_sources.setdefault(gene, set()).add("COSMIC_CGC")

    # Create combined TSV
    combined_path = output_dir / "combined_cancer_genes.tsv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["hugoSymbol", "source_count", "sources", "in_oncokb", "in_bailey_2018", "in_cosmic_cgc"])
        for gene in sorted(gene_sources.keys()):
            sources = gene_sources[gene]
            writer.writerow([
                gene,
                len(sources),
                ",".join(sorted(sources)),
                "OncoKB" in sources,
                "Bailey_2018" in sources,
                "COSMIC_CGC" in sources
            ])
    print(f"  Saved combined TSV to {combined_path}")

    # Save gene list only
    genes_only_path = output_dir / "combined_cancer_genes_list.txt"
    with open(genes_only_path, "w") as f:
        for gene in sorted(gene_sources.keys()):
            f.write(f"{gene}\n")
    print(f"  Saved gene list to {genes_only_path}")

    # Create high-confidence list (genes in 2+ sources)
    high_conf_genes = [g for g, s in gene_sources.items() if len(s) >= 2]
    high_conf_path = output_dir / "high_confidence_cancer_genes_list.txt"
    with open(high_conf_path, "w") as f:
        for gene in sorted(high_conf_genes):
            f.write(f"{gene}\n")
    print(f"  Saved high-confidence list ({len(high_conf_genes)} genes) to {high_conf_path}")

    # Statistics
    in_all_three = sum(1 for s in gene_sources.values() if len(s) == 3)
    print(f"  Combined list has {len(gene_sources)} unique genes")
    print(f"    - In 2+ sources: {len(high_conf_genes)}")
    print(f"    - In all 3 sources: {in_all_three}")

    return gene_sources


def create_readme(output_dir: Path, stats: Dict[str, int]) -> None:
    """Create a README file documenting the gene lists."""
    readme_path = output_dir / "README_gene_lists.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# Cancer Gene Lists

Downloaded/generated on: {timestamp}

## Overview

This directory contains cancer gene lists from multiple sources to address
reviewer concerns about OncoKB's actionability bias.

## Data Sources

### 1. OncoKB Cancer Gene List
- **File**: `oncokb_cancer_genes.tsv`, `oncokb_cancer_genes_list.txt`
- **Source**: https://www.oncokb.org/api/v1/utils/cancerGeneList
- **Count**: {stats.get('oncokb_count', 'N/A')} genes
- **Description**: MSK's precision oncology knowledge base. Contains genes with
  clinically actionable mutations and therapeutic implications.
- **Note**: Has bias towards actionability (therapeutic relevance)

### 2. Bailey et al. 2018 Driver Genes
- **File**: `bailey_2018_driver_genes.tsv`, `bailey_2018_driver_genes_list.txt`
- **Source**: Cell 173, 371-385.e18 (2018), DOI: 10.1016/j.cell.2018.02.060
- **Count**: {stats.get('bailey_count', 'N/A')} genes
- **Description**: Comprehensive analysis of cancer driver genes from TCGA
  PanCancer Atlas using 26 computational tools across 9,423 tumor exomes.
- **Note**: Unbiased computational identification of driver genes

### 3. COSMIC Cancer Gene Census (CGC)
- **File**: `cosmic_cgc_genes.tsv`, `cosmic_cgc_genes_list.txt`
- **Source**: https://cancer.sanger.ac.uk/cosmic
- **Count**: {stats.get('cosmic_count', 'N/A')} genes
- **Description**: Expert-curated catalogue of genes causally implicated in cancer.
  This is a curated subset of Tier 1 genes. Full dataset requires COSMIC registration.
- **Note**: Download full dataset from https://cancer.sanger.ac.uk/cosmic/download

### 4. Combined Gene List
- **File**: `combined_cancer_genes.tsv`, `combined_cancer_genes_list.txt`
- **Count**: {stats.get('combined_count', 'N/A')} unique genes
- **Description**: Union of all three gene lists with source annotations

### 5. High-Confidence Gene List
- **File**: `high_confidence_cancer_genes_list.txt`
- **Count**: {stats.get('high_conf_count', 'N/A')} genes
- **Description**: Genes appearing in 2+ sources (higher confidence)

## File Formats

- `.tsv`: Tab-separated values with gene annotations
- `_list.txt`: Simple gene list (one gene per line, sorted alphabetically)
- `.json`: Raw API response data

## Usage

For ES Score analysis, consider using:
1. `combined_cancer_genes_list.txt` for comprehensive coverage
2. `high_confidence_cancer_genes_list.txt` for higher confidence subset
3. Individual source lists for source-specific analysis

## References

1. Chakravarty D, et al. (2017) OncoKB: A Precision Oncology Knowledge Base.
   JCO Precision Oncology.
2. Bailey MH, et al. (2018) Comprehensive Characterization of Cancer Driver
   Genes and Mutations. Cell 173, 371-385.e18.
3. Sondka Z, et al. (2018) The COSMIC Cancer Gene Census: describing genetic
   dysfunction across all human cancers. Nature Reviews Cancer 18, 696-705.
"""

    with open(readme_path, "w") as f:
        f.write(content)
    print(f"Created README at {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize cancer gene lists from multiple sources"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Download/create gene lists
    oncokb_data = download_oncokb_cancer_genes(output_dir)
    print()

    bailey_genes = create_bailey_driver_genes(output_dir)
    print()

    cosmic_genes = create_cosmic_cgc(output_dir)
    print()

    # Create combined list
    gene_sources = create_combined_gene_list(oncokb_data, bailey_genes, cosmic_genes, output_dir)
    print()

    # Create README
    high_conf_count = sum(1 for s in gene_sources.values() if len(s) >= 2)
    stats = {
        "oncokb_count": len(oncokb_data),
        "bailey_count": len(bailey_genes),
        "cosmic_count": len(cosmic_genes),
        "combined_count": len(gene_sources),
        "high_conf_count": high_conf_count
    }
    create_readme(output_dir, stats)

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
