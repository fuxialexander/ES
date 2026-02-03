#!/usr/bin/env python3
"""
HegeLab AlphaMissense web resource client.

Provides access to AlphaMissense data via the HegeLab web interface:
https://alphamissense.hegelab.org

Features:
- Individual variant lookups
- Structure files with AlphaMissense scores in B-factor column
- Hotspot API for residue-level queries

Best for:
- Quick individual lookups
- Accessing structure data with integrated scores
- When you need hotspot analysis in structural context
"""

import json
import urllib.request
import urllib.error
import urllib.parse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


# HegeLab API base URL
HEGELAB_BASE = "https://alphamissense.hegelab.org"


@dataclass
class HegeLab_AMResult:
    """Container for HegeLab AlphaMissense result."""
    uniprot_id: str
    residue: Optional[int]
    ref_aa: Optional[str]
    alt_aa: Optional[str]
    score: Optional[float]
    classification: Optional[str]
    raw_response: Optional[dict] = None


@dataclass
class HotspotResult:
    """Container for hotspot API result."""
    uniprot_id: str
    residue: int
    mean_score: Optional[float]
    scores: Optional[Dict[str, float]]  # alt_aa -> score
    raw_response: Optional[dict] = None


class HegeLab_AMClient:
    """
    Client for HegeLab AlphaMissense web resource.

    Provides access to:
    - Individual variant predictions
    - Structure files (PDB) with scores in B-factor column
    - Hotspot API for per-residue queries

    Example usage:
        client = HegeLab_AMClient()

        # Get PDB structure with scores
        pdb_content = client.get_structure_pdb("P00533")

        # Query hotspot API for a residue
        result = client.get_hotspot("P00533", 858)
    """

    def __init__(
        self,
        base_url: str = HEGELAB_BASE,
        rate_limit: float = 5.0,  # conservative rate limit
        timeout: int = 30,
    ):
        """
        Initialize the HegeLab client.

        Args:
            base_url: HegeLab base URL
            rate_limit: Maximum requests per second
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        parse_json: bool = True,
    ) -> Union[dict, str]:
        """
        Make a request to HegeLab.

        Args:
            url: Full URL to request
            parse_json: Whether to parse response as JSON

        Returns:
            Parsed JSON dict or raw text
        """
        self._wait_for_rate_limit()

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (ES Score Benchmark)"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                if parse_json:
                    return json.loads(content)
                return content
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {} if parse_json else ""
            raise

    def get_structure_pdb(
        self,
        uniprot_id: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Get AlphaFold structure with AlphaMissense scores in B-factor column.

        The returned PDB file has mean AlphaMissense scores per residue
        stored in the B-factor (temperature factor) column.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')
            output_path: Optional path to save the PDB file

        Returns:
            PDB file content as string
        """
        url = f"{self.base_url}/pdb/AF-{uniprot_id}-F1-AM_v4.pdb"

        try:
            content = self._make_request(url, parse_json=False)
        except urllib.error.HTTPError:
            return ""

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)

        return content

    def get_hotspot(
        self,
        uniprot_id: str,
        residue: int,
    ) -> HotspotResult:
        """
        Query the hotspot API for a specific residue.

        Returns AlphaMissense scores for all possible substitutions
        at the specified position.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')
            residue: 1-based residue number

        Returns:
            HotspotResult with scores for all substitutions
        """
        url = f"{self.base_url}/hotspotapi?uid={uniprot_id}&resi={residue}"

        try:
            response = self._make_request(url, parse_json=True)
        except (urllib.error.HTTPError, json.JSONDecodeError):
            return HotspotResult(
                uniprot_id=uniprot_id,
                residue=residue,
                mean_score=None,
                scores=None,
            )

        if not response:
            return HotspotResult(
                uniprot_id=uniprot_id,
                residue=residue,
                mean_score=None,
                scores=None,
            )

        # Parse response - format depends on API
        # Typically returns scores for each possible substitution
        scores = {}
        mean_score = None

        if isinstance(response, dict):
            # Handle dict response with variant scores
            for key, value in response.items():
                if key == "mean" or key == "mean_score":
                    mean_score = float(value)
                elif len(key) == 1 and key.isalpha():  # Single AA code
                    scores[key] = float(value)

        return HotspotResult(
            uniprot_id=uniprot_id,
            residue=residue,
            mean_score=mean_score,
            scores=scores if scores else None,
            raw_response=response,
        )

    def get_structure_page_url(self, uniprot_id: str) -> str:
        """
        Get URL to the interactive structure viewer page.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')

        Returns:
            URL to the structure viewer
        """
        return f"{self.base_url}/structure/{uniprot_id}"

    def download_structures_batch(
        self,
        uniprot_ids: List[str],
        output_dir: Union[str, Path],
        progress: bool = True,
    ) -> Dict[str, bool]:
        """
        Download multiple structure files.

        Args:
            uniprot_ids: List of UniProt accessions
            output_dir: Directory to save PDB files
            progress: Show progress

        Returns:
            Dict mapping UniProt ID to download success status
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        total = len(uniprot_ids)

        for i, uniprot_id in enumerate(uniprot_ids, 1):
            if progress:
                print(f"Downloading {i}/{total}: {uniprot_id}", end="\r")

            output_path = output_dir / f"AF-{uniprot_id}-F1-AM_v4.pdb"
            content = self.get_structure_pdb(uniprot_id, output_path)
            results[uniprot_id] = bool(content)

        if progress:
            n_success = sum(results.values())
            print(f"\nDownloaded {n_success}/{total} structures")

        return results

    def parse_pdb_bfactors(
        self,
        pdb_content: str,
    ) -> List[float]:
        """
        Extract B-factor values (AlphaMissense scores) from PDB content.

        Args:
            pdb_content: PDB file content as string

        Returns:
            List of B-factor values per residue (CA atoms)
        """
        bfactors = []

        for line in pdb_content.split("\n"):
            if line.startswith("ATOM") and " CA " in line:
                # B-factor is in columns 61-66 (1-indexed)
                try:
                    bfactor = float(line[60:66].strip())
                    bfactors.append(bfactor)
                except (ValueError, IndexError):
                    continue

        return bfactors


def download_zenodo_structures(
    output_dir: Union[str, Path],
    timeout: int = 300,
) -> bool:
    """
    Download bulk structure files from Zenodo.

    The Zenodo archive contains PDB files with AlphaMissense scores
    for all human proteins.

    Args:
        output_dir: Directory to save the zip file
        timeout: Download timeout in seconds

    Returns:
        True if download succeeded
    """
    zenodo_url = "https://zenodo.org/records/10023059/files/AM_structures.zip"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "AM_structures.zip"

    print(f"Downloading AlphaMissense structure archive from Zenodo...")
    print(f"  This is a large file and may take a while...")

    try:
        request = urllib.request.Request(
            zenodo_url,
            headers={"User-Agent": "Mozilla/5.0 (ES Score Benchmark)"},
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                print(f"  Size: {total_size / 1024 / 1024 / 1024:.1f} GB")

            downloaded = 0
            chunk_size = 8192
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        print(f"\n  Saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query AlphaMissense via HegeLab")
    parser.add_argument("--uniprot", type=str, help="UniProt ID")
    parser.add_argument("--residue", type=int, help="Residue number for hotspot query")
    parser.add_argument("--structure", action="store_true", help="Download structure PDB")
    parser.add_argument("--output", type=str, help="Output path for structure")

    args = parser.parse_args()

    client = HegeLab_AMClient()

    if args.uniprot:
        if args.residue:
            result = client.get_hotspot(args.uniprot, args.residue)
            print(f"Hotspot result for {args.uniprot} residue {args.residue}:")
            print(f"  Mean score: {result.mean_score}")
            print(f"  Scores: {result.scores}")
        elif args.structure:
            content = client.get_structure_pdb(args.uniprot, args.output)
            if content:
                print(f"Structure retrieved for {args.uniprot}")
                if args.output:
                    print(f"  Saved to: {args.output}")
                else:
                    bfactors = client.parse_pdb_bfactors(content)
                    print(f"  Residues: {len(bfactors)}")
                    if bfactors:
                        print(f"  Score range: {min(bfactors):.3f} - {max(bfactors):.3f}")
            else:
                print(f"No structure found for {args.uniprot}")
        else:
            print(f"Structure page URL: {client.get_structure_page_url(args.uniprot)}")
    else:
        print("Usage: python hegelab_client.py --uniprot <ID> [--residue <N>] [--structure]")
