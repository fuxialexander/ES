#!/usr/bin/env python3
"""
Ensembl VEP REST API client for AlphaMissense predictions.

Uses the Ensembl Variant Effect Predictor (VEP) REST API to retrieve
AlphaMissense pathogenicity scores for individual variants.

Best for:
- Small numbers of variants (< 100)
- When you need additional VEP annotations alongside AlphaMissense
- When bulk data download is not feasible

Rate limits: Ensembl REST API allows ~15 requests/second for unregistered users.
For higher throughput, consider using bulk data or registering for an API key.

API documentation: https://rest.ensembl.org/documentation/info/vep_hgvs_get
"""

import json
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


# Ensembl REST API base URL
ENSEMBL_REST_BASE = "https://rest.ensembl.org"

# Default parameters for VEP queries
DEFAULT_VEP_PARAMS = {
    "content-type": "application/json",
    "AlphaMissense": "1",  # Enable AlphaMissense annotations
}


@dataclass
class AlphaMissenseResult:
    """Container for AlphaMissense result from Ensembl VEP."""
    score: Optional[float]
    classification: Optional[str]
    transcript_id: Optional[str]
    protein_position: Optional[int]
    amino_acids: Optional[str]  # e.g., "V/E" for V600E
    raw_response: Optional[dict] = None


class EnsemblVEPClient:
    """
    Client for Ensembl VEP REST API with AlphaMissense support.

    Example usage:
        client = EnsemblVEPClient()

        # Query by HGVS notation
        result = client.get_alphamissense_hgvs("ENST00000275493:c.2573T>G")

        # Query by genomic coordinates (hg38)
        result = client.get_alphamissense_region("7", 140453136, "A", "T")

        # Batch query multiple variants
        results = client.batch_query([
            {"hgvs": "ENST00000275493:c.2573T>G"},
            {"hgvs": "ENST00000288602:c.35G>T"},
        ])
    """

    def __init__(
        self,
        base_url: str = ENSEMBL_REST_BASE,
        rate_limit: float = 15.0,  # requests per second
        timeout: int = 30,
    ):
        """
        Initialize the Ensembl VEP client.

        Args:
            base_url: Ensembl REST API base URL
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
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        data: Optional[str] = None,
    ) -> dict:
        """
        Make a request to the Ensembl REST API.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            method: HTTP method
            data: POST data (for batch queries)

        Returns:
            JSON response as dictionary

        Raises:
            urllib.error.HTTPError: On HTTP errors
            ValueError: On JSON parsing errors
        """
        self._wait_for_rate_limit()

        # Build URL
        url = f"{self.base_url}{endpoint}"
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"

        # Create request
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            url,
            headers=headers,
            method=method,
            data=data.encode("utf-8") if data else None,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                # Back off and retry
                time.sleep(1.0)
                return self._make_request(endpoint, params, method, data)
            raise

    def get_alphamissense_hgvs(
        self,
        hgvs_notation: str,
        species: str = "human",
    ) -> AlphaMissenseResult:
        """
        Get AlphaMissense prediction for a variant using HGVS notation.

        Args:
            hgvs_notation: HGVS variant notation (e.g., "ENST00000275493:c.2573T>G")
            species: Species name (default: human)

        Returns:
            AlphaMissenseResult with score and classification
        """
        endpoint = f"/vep/{species}/hgvs/{urllib.parse.quote(hgvs_notation)}"
        params = DEFAULT_VEP_PARAMS.copy()

        try:
            response = self._make_request(endpoint, params)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # Invalid HGVS notation
                return AlphaMissenseResult(
                    score=None,
                    classification=None,
                    transcript_id=None,
                    protein_position=None,
                    amino_acids=None,
                )
            raise

        return self._parse_vep_response(response)

    def get_alphamissense_region(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        species: str = "human",
        assembly: str = "GRCh38",
    ) -> AlphaMissenseResult:
        """
        Get AlphaMissense prediction for a variant using genomic coordinates.

        Args:
            chromosome: Chromosome (e.g., "7" or "chr7")
            position: 1-based genomic position
            ref: Reference allele
            alt: Alternate allele
            species: Species name (default: human)
            assembly: Genome assembly (default: GRCh38)

        Returns:
            AlphaMissenseResult with score and classification
        """
        # Normalize chromosome format
        chrom = chromosome.replace("chr", "")

        # Build region string: chr:start-end:strand/allele
        region = f"{chrom}:{position}-{position}:1/{alt}"
        endpoint = f"/vep/{species}/region/{urllib.parse.quote(region)}"

        params = DEFAULT_VEP_PARAMS.copy()

        try:
            response = self._make_request(endpoint, params)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return AlphaMissenseResult(
                    score=None,
                    classification=None,
                    transcript_id=None,
                    protein_position=None,
                    amino_acids=None,
                )
            raise

        return self._parse_vep_response(response)

    def batch_query_hgvs(
        self,
        hgvs_notations: List[str],
        species: str = "human",
        chunk_size: int = 200,
    ) -> List[AlphaMissenseResult]:
        """
        Batch query multiple variants by HGVS notation.

        Uses POST endpoint for efficient batch queries.

        Args:
            hgvs_notations: List of HGVS variant notations
            species: Species name (default: human)
            chunk_size: Number of variants per request (max 200)

        Returns:
            List of AlphaMissenseResult objects
        """
        results = []

        # Process in chunks
        for i in range(0, len(hgvs_notations), chunk_size):
            chunk = hgvs_notations[i : i + chunk_size]

            endpoint = f"/vep/{species}/hgvs"
            params = DEFAULT_VEP_PARAMS.copy()

            data = json.dumps({"hgvs_notations": chunk})

            try:
                response = self._make_request(endpoint, params, method="POST", data=data)
                for item in response:
                    results.append(self._parse_vep_response([item]))
            except urllib.error.HTTPError as e:
                # Add None results for failed batch
                results.extend([
                    AlphaMissenseResult(
                        score=None,
                        classification=None,
                        transcript_id=None,
                        protein_position=None,
                        amino_acids=None,
                    )
                    for _ in chunk
                ])

        return results

    def batch_query_regions(
        self,
        variants: List[Dict[str, Union[str, int]]],
        species: str = "human",
        chunk_size: int = 200,
    ) -> List[AlphaMissenseResult]:
        """
        Batch query multiple variants by genomic coordinates.

        Args:
            variants: List of dicts with keys: chrom, pos, ref, alt
            species: Species name (default: human)
            chunk_size: Number of variants per request (max 200)

        Returns:
            List of AlphaMissenseResult objects
        """
        results = []

        # Process in chunks
        for i in range(0, len(variants), chunk_size):
            chunk = variants[i : i + chunk_size]

            endpoint = f"/vep/{species}/region"
            params = DEFAULT_VEP_PARAMS.copy()

            # Format variants for POST request
            formatted = []
            for v in chunk:
                chrom = str(v["chrom"]).replace("chr", "")
                pos = v["pos"]
                ref = v["ref"]
                alt = v["alt"]
                formatted.append(f"{chrom} {pos} . {ref} {alt} . . .")

            data = json.dumps({"variants": formatted})

            try:
                response = self._make_request(endpoint, params, method="POST", data=data)
                for item in response:
                    results.append(self._parse_vep_response([item]))
            except urllib.error.HTTPError:
                # Add None results for failed batch
                results.extend([
                    AlphaMissenseResult(
                        score=None,
                        classification=None,
                        transcript_id=None,
                        protein_position=None,
                        amino_acids=None,
                    )
                    for _ in chunk
                ])

        return results

    def _parse_vep_response(self, response: Union[list, dict]) -> AlphaMissenseResult:
        """
        Parse VEP response to extract AlphaMissense data.

        Args:
            response: VEP API response

        Returns:
            AlphaMissenseResult with extracted data
        """
        if isinstance(response, list):
            if len(response) == 0:
                return AlphaMissenseResult(
                    score=None,
                    classification=None,
                    transcript_id=None,
                    protein_position=None,
                    amino_acids=None,
                )
            response = response[0]

        # Look for AlphaMissense data in transcript consequences
        transcript_consequences = response.get("transcript_consequences", [])

        for tc in transcript_consequences:
            # AlphaMissense data can be in two formats:
            # 1. Nested: {"alphamissense": {"am_pathogenicity": 0.99, "am_class": "likely_pathogenic"}}
            # 2. Flat: {"alphamissense_score": 0.99, "alphamissense_class": "likely_pathogenic"}

            am_data = tc.get("alphamissense", {})
            am_score = am_data.get("am_pathogenicity") if am_data else tc.get("alphamissense_score")
            am_class = am_data.get("am_class") if am_data else tc.get("alphamissense_class")

            if am_score is not None:
                return AlphaMissenseResult(
                    score=float(am_score),
                    classification=am_class,
                    transcript_id=tc.get("transcript_id"),
                    protein_position=tc.get("protein_start"),
                    amino_acids=tc.get("amino_acids"),
                    raw_response=response,
                )

        # No AlphaMissense data found
        return AlphaMissenseResult(
            score=None,
            classification=None,
            transcript_id=transcript_consequences[0].get("transcript_id") if transcript_consequences else None,
            protein_position=transcript_consequences[0].get("protein_start") if transcript_consequences else None,
            amino_acids=transcript_consequences[0].get("amino_acids") if transcript_consequences else None,
            raw_response=response,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query AlphaMissense via Ensembl VEP API")
    parser.add_argument("--hgvs", type=str, help="HGVS notation (e.g., ENST00000275493:c.2573T>G)")
    parser.add_argument("--chrom", type=str, help="Chromosome")
    parser.add_argument("--pos", type=int, help="Position")
    parser.add_argument("--ref", type=str, help="Reference allele")
    parser.add_argument("--alt", type=str, help="Alternate allele")

    args = parser.parse_args()

    client = EnsemblVEPClient()

    if args.hgvs:
        result = client.get_alphamissense_hgvs(args.hgvs)
        print(f"AlphaMissense result for {args.hgvs}:")
        print(f"  Score: {result.score}")
        print(f"  Classification: {result.classification}")
        print(f"  Transcript: {result.transcript_id}")
        print(f"  Position: {result.protein_position}")
        print(f"  Amino acids: {result.amino_acids}")
    elif args.chrom and args.pos and args.ref and args.alt:
        result = client.get_alphamissense_region(args.chrom, args.pos, args.ref, args.alt)
        print(f"AlphaMissense result for {args.chrom}:{args.pos} {args.ref}>{args.alt}:")
        print(f"  Score: {result.score}")
        print(f"  Classification: {result.classification}")
        print(f"  Transcript: {result.transcript_id}")
        print(f"  Position: {result.protein_position}")
        print(f"  Amino acids: {result.amino_acids}")
    else:
        print("Usage: python ensembl_client.py --hgvs <HGVS>")
        print("   or: python ensembl_client.py --chrom <chr> --pos <pos> --ref <ref> --alt <alt>")
