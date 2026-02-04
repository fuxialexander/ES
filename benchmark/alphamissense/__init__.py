"""
AlphaMissense prediction fetcher module.

This module provides utilities to retrieve AlphaMissense pathogenicity predictions
for variants using multiple data sources:

1. Bulk data from Zenodo (recommended for large-scale analysis)
2. Ensembl VEP REST API (for small-scale queries)
3. HegeLab web resource (for individual lookups and structure-based queries)

Example usage:
    from benchmark.alphamissense import AlphaMissenseFetcher

    # Initialize with bulk data (recommended)
    fetcher = AlphaMissenseFetcher(data_dir="/mnt/storage/alphamissense")
    fetcher.download_bulk_data()  # Only needed once

    # Query variants
    score = fetcher.get_score("P00533", "L858R")  # Gene/UniProt + mutation
    scores = fetcher.get_scores_for_gene("EGFR")  # All possible substitutions

    # Or use Ensembl VEP API directly (for small queries)
    from benchmark.alphamissense import EnsemblVEPClient
    client = EnsemblVEPClient()
    result = client.get_alphamissense("ENST00000275493", "c.2573T>G")
"""

from .fetcher import AlphaMissenseFetcher
from .ensembl_client import EnsemblVEPClient
from .hegelab_client import HegeLab_AMClient
from .data_loader import AlphaMissenseLoader

__all__ = [
    "AlphaMissenseFetcher",
    "EnsemblVEPClient",
    "HegeLab_AMClient",
    "AlphaMissenseLoader",
]
