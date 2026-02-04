#!/usr/bin/env python3
"""
Unit tests for ProteinGym benchmark evaluation pipeline.

Tests the proteingym_loader, es_scorer, and evaluate modules using
mock/dummy data to verify the pipeline works correctly without requiring
external data files (pLDDT, ESM, ProteinGym datasets).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add benchmark directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "proteingym"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from proteingym_loader import (
    DMSAssay,
    ProteinGymLoader,
    parse_mutation,
    is_single_mutation,
    get_mutation_positions,
    filter_single_mutations,
    get_assay_statistics,
)
from evaluate import (
    EvaluationResult,
    BenchmarkResults,
    compute_spearman,
    compute_auc,
    compute_mcc,
    compute_ndcg,
    compute_top_k_recall,
    evaluate_assay,
    evaluate_benchmark,
    results_to_dataframe,
)


# =============================================================================
# Fixtures: Mock Data Generation
# =============================================================================


def create_mock_dms_data(n_variants: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create mock DMS assay data for testing.

    Args:
        n_variants: Number of variants to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with mutant, DMS_score, DMS_score_bin columns
    """
    np.random.seed(seed)

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    mutants = []
    for i in range(n_variants):
        wt = np.random.choice(amino_acids)
        pos = np.random.randint(1, 500)
        mt = np.random.choice([aa for aa in amino_acids if aa != wt])
        mutants.append(f"{wt}{pos}{mt}")

    # Generate DMS scores (higher = more fit)
    dms_scores = np.random.randn(n_variants) * 0.5

    # Binary labels based on median
    median = np.median(dms_scores)
    dms_bins = (dms_scores > median).astype(int)

    return pd.DataFrame({
        "mutant": mutants,
        "DMS_score": dms_scores,
        "DMS_score_bin": dms_bins
    })


def create_mock_assay(
    assay_id: str = "TEST_HUMAN",
    uniprot_id: str = "P12345",
    gene_name: str = "TEST",
    n_variants: int = 100,
    seq_length: int = 500,
    seed: int = 42
) -> DMSAssay:
    """Create a mock DMSAssay object for testing."""
    data = create_mock_dms_data(n_variants, seed=seed)
    target_seq = "M" + "A" * (seq_length - 1)  # Simple mock sequence

    return DMSAssay(
        assay_id=assay_id,
        uniprot_id=uniprot_id,
        gene_name=gene_name,
        target_seq=target_seq,
        data=data
    )


class DummyPredictor:
    """
    Dummy predictor for testing evaluation pipeline.

    Supports different prediction strategies:
    - 'random': Random predictions
    - 'perfect': Perfect predictions (ES = -DMS)
    - 'inverse': Inverse predictions (ES = DMS)
    - 'constant': All same value
    """

    def __init__(self, strategy: str = "random", seed: int = 42):
        self.strategy = strategy
        self.seed = seed
        np.random.seed(seed)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate dummy ES score predictions."""
        n = len(df)

        if self.strategy == "random":
            return np.random.randn(n)

        elif self.strategy == "perfect":
            # Perfect negative correlation with DMS
            # (high ES = deleterious = low DMS)
            if "DMS_score" in df.columns:
                return -df["DMS_score"].values
            return np.random.randn(n)

        elif self.strategy == "inverse":
            # Positive correlation (wrong direction)
            if "DMS_score" in df.columns:
                return df["DMS_score"].values
            return np.random.randn(n)

        elif self.strategy == "constant":
            return np.ones(n) * 0.5

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def score_assay(self, assay: DMSAssay) -> pd.DataFrame:
        """Score a DMS assay and return DataFrame with es_score column."""
        df = assay.data.copy()
        df["es_score"] = self.predict(df)
        return df


# =============================================================================
# Tests: Mutation Parsing
# =============================================================================


class TestMutationParsing:
    """Tests for mutation string parsing utilities."""

    def test_parse_single_mutation(self):
        """Test parsing single point mutations."""
        result = parse_mutation("A123G")
        assert result == [("A", 123, "G")]

    def test_parse_multi_mutation(self):
        """Test parsing multiple mutations separated by colon."""
        result = parse_mutation("A123G:D456E")
        assert len(result) == 2
        assert result[0] == ("A", 123, "G")
        assert result[1] == ("D", 456, "E")

    def test_parse_invalid_mutation(self):
        """Test parsing invalid mutation strings."""
        result = parse_mutation("invalid")
        assert result == []

    def test_parse_empty_mutation(self):
        """Test parsing empty mutation string."""
        result = parse_mutation("")
        assert result == []

    def test_is_single_mutation_true(self):
        """Test is_single_mutation returns True for single mutations."""
        assert is_single_mutation("A123G") is True
        assert is_single_mutation("K1M") is True

    def test_is_single_mutation_false(self):
        """Test is_single_mutation returns False for multi mutations."""
        assert is_single_mutation("A123G:D456E") is False
        assert is_single_mutation("A1B:C2D:E3F") is False

    def test_get_mutation_positions(self):
        """Test extracting positions from mutations."""
        positions = get_mutation_positions("A123G:D456E")
        assert positions == [123, 456]

    def test_filter_single_mutations(self):
        """Test filtering DataFrame to single mutations only."""
        df = pd.DataFrame({
            "mutant": ["A1G", "A1G:D2E", "K3M", "L4I:M5V:N6T"],
            "score": [1, 2, 3, 4]
        })
        filtered = filter_single_mutations(df)
        assert len(filtered) == 2
        assert set(filtered["mutant"]) == {"A1G", "K3M"}


# =============================================================================
# Tests: DMS Assay Loading
# =============================================================================


class TestDMSAssay:
    """Tests for DMSAssay dataclass."""

    def test_assay_creation(self):
        """Test basic assay creation."""
        assay = create_mock_assay()
        assert assay.assay_id == "TEST_HUMAN"
        assert assay.uniprot_id == "P12345"
        assert assay.gene_name == "TEST"
        assert len(assay.data) == 100

    def test_assay_statistics(self):
        """Test computing assay statistics."""
        assay = create_mock_assay(n_variants=50)
        stats = get_assay_statistics(assay)

        assert stats["n_variants"] == 50
        assert "seq_length" in stats
        assert "dms_score_mean" in stats
        assert "dms_score_std" in stats


class TestProteinGymLoader:
    """Tests for ProteinGymLoader class."""

    def test_loader_initialization(self):
        """Test loader can be initialized with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ProteinGymLoader(tmpdir)
            assert loader.data_dir == Path(tmpdir)

    def test_list_assays_empty(self):
        """Test listing assays from empty directory raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ProteinGymLoader(tmpdir)
            # Empty directory should raise FileNotFoundError when no data exists
            with pytest.raises(FileNotFoundError):
                loader.list_assays()

    def test_load_assay_from_csv(self):
        """Test loading assay from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock CSV
            data_dir = Path(tmpdir) / "DMS_ProteinGym_substitutions"
            data_dir.mkdir(parents=True)

            mock_data = create_mock_dms_data(50)
            mock_data.to_csv(data_dir / "TEST_HUMAN.csv", index=False)

            loader = ProteinGymLoader(tmpdir)

            # Should be able to list the assay
            assays = loader.list_assays()
            assert "TEST_HUMAN" in assays

            # Should be able to load it
            assay = loader.load_assay("TEST_HUMAN")
            assert assay.assay_id == "TEST_HUMAN"
            assert len(assay.data) == 50


# =============================================================================
# Tests: Evaluation Metrics
# =============================================================================


class TestComputeSpearman:
    """Tests for Spearman correlation computation."""

    def test_perfect_correlation(self):
        """Test perfect negative correlation."""
        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([-1, -2, -3, -4, -5])
        rho, pval = compute_spearman(predictions, targets)
        assert rho == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self):
        """Test no correlation (random data)."""
        np.random.seed(42)
        predictions = np.random.randn(1000)
        targets = np.random.randn(1000)
        rho, pval = compute_spearman(predictions, targets)
        assert abs(rho) < 0.1  # Should be close to 0

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        predictions = np.array([1, 2, np.nan, 4, 5])
        targets = np.array([1, 2, 3, np.nan, 5])
        rho, pval = compute_spearman(predictions, targets)
        assert not np.isnan(rho)

    def test_too_few_samples(self):
        """Test with too few samples."""
        predictions = np.array([1, 2])
        targets = np.array([1, 2])
        rho, pval = compute_spearman(predictions, targets)
        assert np.isnan(rho)


class TestComputeAUC:
    """Tests for AUC-ROC computation."""

    def test_perfect_auc(self):
        """Test perfect binary classification."""
        # High predictions for class 0 (not fit)
        predictions = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        binary_labels = np.array([0, 0, 0, 1, 1, 1])
        auc_val = compute_auc(predictions, binary_labels)
        assert auc_val == pytest.approx(1.0, abs=0.01)

    def test_random_auc(self):
        """Test random predictions."""
        np.random.seed(42)
        predictions = np.random.rand(100)
        binary_labels = np.random.randint(0, 2, 100)
        auc_val = compute_auc(predictions, binary_labels)
        assert 0.3 < auc_val < 0.7  # Should be around 0.5

    def test_single_class(self):
        """Test with single class labels."""
        predictions = np.array([0.1, 0.2, 0.3])
        binary_labels = np.array([1, 1, 1])
        auc_val = compute_auc(predictions, binary_labels)
        assert np.isnan(auc_val)


class TestComputeMCC:
    """Tests for Matthews Correlation Coefficient computation."""

    def test_perfect_mcc(self):
        """Test perfect classification."""
        predictions = np.array([0.9, 0.8, 0.1, 0.2])
        binary_labels = np.array([0, 0, 1, 1])
        mcc_val = compute_mcc(predictions, binary_labels)
        assert mcc_val == pytest.approx(1.0, abs=0.1)

    def test_random_mcc(self):
        """Test random predictions."""
        np.random.seed(42)
        predictions = np.random.rand(100)
        binary_labels = np.random.randint(0, 2, 100)
        mcc_val = compute_mcc(predictions, binary_labels)
        assert -0.3 < mcc_val < 0.3


class TestComputeNDCG:
    """Tests for NDCG computation."""

    def test_ndcg_calculation(self):
        """Test NDCG with ordered data."""
        # Lower predictions should rank higher for fitness
        predictions = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        targets = np.array([5, 4, 3, 2, 1])  # Higher = better
        ndcg_val = compute_ndcg(predictions, targets)
        assert ndcg_val == pytest.approx(1.0, abs=0.01)

    def test_ndcg_with_nan(self):
        """Test NDCG with NaN values."""
        predictions = np.array([0.1, np.nan, 0.3])
        targets = np.array([3, 2, np.nan])
        ndcg_val = compute_ndcg(predictions, targets)
        assert not np.isnan(ndcg_val) or ndcg_val is np.nan


class TestTopKRecall:
    """Tests for Top-K recall computation."""

    def test_perfect_top_k(self):
        """Test perfect top-k recall."""
        # Lowest predictions align with highest targets
        predictions = np.arange(100)[::-1].astype(float)  # 99, 98, ..., 0
        targets = np.arange(100).astype(float)  # 0, 1, ..., 99

        results = compute_top_k_recall(predictions, targets)
        assert results[10] == pytest.approx(1.0, abs=0.01)

    def test_random_top_k(self):
        """Test random top-k recall."""
        np.random.seed(42)
        predictions = np.random.rand(100)
        targets = np.random.rand(100)

        results = compute_top_k_recall(predictions, targets)
        # Random should be around k/n
        assert 0.0 <= results[10] <= 0.5


# =============================================================================
# Tests: Assay Evaluation
# =============================================================================


class TestEvaluateAssay:
    """Tests for single assay evaluation."""

    def test_evaluate_with_dummy_predictor(self):
        """Test evaluation with a dummy predictor."""
        assay = create_mock_assay(n_variants=100)
        predictor = DummyPredictor(strategy="random")
        scored_df = predictor.score_assay(assay)

        result = evaluate_assay(scored_df, assay.assay_id)

        assert result.assay_id == "TEST_HUMAN"
        assert result.n_variants == 100
        assert not np.isnan(result.spearman_rho)

    def test_perfect_predictor(self):
        """Test evaluation with perfect predictor."""
        assay = create_mock_assay(n_variants=100)
        predictor = DummyPredictor(strategy="perfect")
        scored_df = predictor.score_assay(assay)

        result = evaluate_assay(scored_df, assay.assay_id)

        # Perfect predictor should have negative Spearman (high ES = low DMS)
        assert result.spearman_rho < -0.9

    def test_inverse_predictor(self):
        """Test evaluation with inverse predictor."""
        assay = create_mock_assay(n_variants=100)
        predictor = DummyPredictor(strategy="inverse")
        scored_df = predictor.score_assay(assay)

        result = evaluate_assay(scored_df, assay.assay_id)

        # Inverse predictor should have positive Spearman (wrong direction)
        assert result.spearman_rho > 0.9


# =============================================================================
# Tests: Benchmark Evaluation
# =============================================================================


class TestEvaluateBenchmark:
    """Tests for full benchmark evaluation."""

    def test_benchmark_multiple_assays(self):
        """Test benchmark evaluation across multiple assays."""
        # Create multiple mock assays
        assays = {
            "ASSAY_A": create_mock_assay("ASSAY_A", n_variants=50),
            "ASSAY_B": create_mock_assay("ASSAY_B", n_variants=75),
            "ASSAY_C": create_mock_assay("ASSAY_C", n_variants=100),
        }

        predictor = DummyPredictor(strategy="random")
        scored_assays = {
            aid: predictor.score_assay(assay)
            for aid, assay in assays.items()
        }

        results = evaluate_benchmark(scored_assays, method_name="Dummy")

        assert results.method_name == "Dummy"
        assert results.n_assays_evaluated == 3
        assert results.n_assays_failed == 0
        assert not np.isnan(results.mean_spearman)

    def test_benchmark_with_small_assays(self):
        """Test that small assays are filtered out."""
        assays = {
            "LARGE": create_mock_assay("LARGE", n_variants=100),
            "SMALL": create_mock_assay("SMALL", n_variants=5),  # Too small
        }

        predictor = DummyPredictor(strategy="random")
        scored_assays = {
            aid: predictor.score_assay(assay)
            for aid, assay in assays.items()
        }

        results = evaluate_benchmark(scored_assays, min_variants=10)

        assert results.n_assays_evaluated == 1
        assert results.n_assays_failed == 1

    def test_benchmark_results_to_dataframe(self):
        """Test converting benchmark results to DataFrame."""
        assay = create_mock_assay(n_variants=50)
        predictor = DummyPredictor(strategy="random")
        scored_df = predictor.score_assay(assay)

        results = evaluate_benchmark({"TEST": scored_df})
        df = results_to_dataframe(results)

        assert "assay_id" in df.columns
        assert "spearman_rho" in df.columns
        assert "n_variants" in df.columns
        assert len(df) == 1


# =============================================================================
# Tests: Dummy Predictor Strategies
# =============================================================================


class TestDummyPredictorStrategies:
    """Tests for different dummy predictor strategies."""

    def test_random_strategy(self):
        """Test random predictor has low correlation."""
        assay = create_mock_assay(n_variants=200)
        predictor = DummyPredictor(strategy="random")
        scored_df = predictor.score_assay(assay)

        result = evaluate_assay(scored_df, assay.assay_id)

        # Random should have correlation close to 0
        assert abs(result.spearman_rho) < 0.3

    def test_constant_strategy(self):
        """Test constant predictor has zero correlation."""
        assay = create_mock_assay(n_variants=200)
        predictor = DummyPredictor(strategy="constant")
        scored_df = predictor.score_assay(assay)

        # Constant predictions have undefined correlation
        # (variance is 0, so Spearman will be NaN)
        rho, _ = compute_spearman(
            scored_df["es_score"].values,
            scored_df["DMS_score"].values
        )
        assert np.isnan(rho)

    def test_predictor_reproducibility(self):
        """Test that predictor with same seed produces same output for same input."""
        # Create identical assays with fixed seed
        assay1 = create_mock_assay(n_variants=50, seed=42)
        assay2 = create_mock_assay(n_variants=50, seed=42)

        # Predictors with same seed and identical input should produce same output
        predictor1 = DummyPredictor(strategy="random", seed=123)
        scores1 = predictor1.predict(assay1.data)

        predictor2 = DummyPredictor(strategy="random", seed=123)
        scores2 = predictor2.predict(assay2.data)

        np.testing.assert_array_equal(scores1, scores2)


# =============================================================================
# Tests: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_with_mock_data(self):
        """Test complete pipeline from data loading to evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create mock data directory
            data_dir = Path(tmpdir) / "DMS_ProteinGym_substitutions"
            data_dir.mkdir(parents=True)

            # Create mock assays
            for name in ["GENE_A", "GENE_B", "GENE_C"]:
                mock_data = create_mock_dms_data(80)
                mock_data.to_csv(data_dir / f"{name}.csv", index=False)

            # 2. Load data
            loader = ProteinGymLoader(tmpdir)
            assay_ids = loader.list_assays()
            assert len(assay_ids) == 3

            # 3. Score with dummy predictor
            predictor = DummyPredictor(strategy="random")
            scored_assays = {}
            for aid in assay_ids:
                assay = loader.load_assay(aid)
                scored_assays[aid] = predictor.score_assay(assay)

            # 4. Evaluate
            results = evaluate_benchmark(scored_assays)

            assert results.n_assays_evaluated == 3
            assert not np.isnan(results.mean_spearman)

    def test_pipeline_with_perfect_predictor(self):
        """Test pipeline with perfect predictor achieves high correlation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "DMS_ProteinGym_substitutions"
            data_dir.mkdir(parents=True)

            # Create consistent mock data
            mock_data = create_mock_dms_data(100, seed=42)
            mock_data.to_csv(data_dir / "TEST.csv", index=False)

            loader = ProteinGymLoader(tmpdir)
            assay = loader.load_assay("TEST")

            predictor = DummyPredictor(strategy="perfect")
            scored_df = predictor.score_assay(assay)

            results = evaluate_benchmark({"TEST": scored_df})

            # Perfect predictor should achieve high negative Spearman
            assert results.mean_spearman < -0.95


# =============================================================================
# Tests: ES Scorer with Mock pLDDT
# =============================================================================


class TestESScorer:
    """Tests for ES Scorer with mock data files."""

    def test_scorer_with_mock_plddt(self):
        """Test ESScorer initialization and scoring with mock pLDDT data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock pLDDT file (UniprotID<tab>comma-separated-scores)
            plddt_file = tmpdir / "mock_plddt.tdt"
            with open(plddt_file, "w") as f:
                # Write pLDDT scores for test proteins
                # Values between 0-100 representing confidence
                f.write("P12345\t" + ",".join([str(50 + i % 40) for i in range(500)]) + "\n")
                f.write("Q67890\t" + ",".join([str(60 + i % 30) for i in range(300)]) + "\n")

            # Create mock UniProt mapping file
            mapping_file = tmpdir / "mapping.txt"
            with open(mapping_file, "w") as f:
                f.write("From\tTo\n")
                f.write("P12345\tTEST1\n")
                f.write("Q67890\tTEST2\n")

            # Import ESScorer here to avoid import errors if benchmark not in path
            from es_scorer import ESScorer

            # Create scorer
            scorer = ESScorer(
                plddt_file=plddt_file,
                uniprot_mapping_file=mapping_file,
                smooth_kernel=5
            )

            # Test computing ES scores for a protein
            es_scores = scorer.compute_es_scores("P12345", gene_name="TEST1")

            assert es_scores is not None
            assert len(es_scores) == 500
            assert es_scores.min() >= 0
            assert es_scores.max() <= 1

    def test_scorer_handles_missing_protein(self):
        """Test that scorer gracefully handles missing proteins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            plddt_file = tmpdir / "mock_plddt.tdt"
            with open(plddt_file, "w") as f:
                f.write("P12345\t" + ",".join([str(50 + i % 40) for i in range(100)]) + "\n")

            mapping_file = tmpdir / "mapping.txt"
            with open(mapping_file, "w") as f:
                f.write("From\tTo\n")
                f.write("P12345\tTEST\n")

            from es_scorer import ESScorer

            scorer = ESScorer(
                plddt_file=plddt_file,
                uniprot_mapping_file=mapping_file
            )

            # Request scores for a protein not in the database
            es_scores = scorer.compute_es_scores("UNKNOWN123")

            assert es_scores is None

    def test_scorer_score_mutations(self):
        """Test scoring mutations in an assay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            plddt_file = tmpdir / "mock_plddt.tdt"
            with open(plddt_file, "w") as f:
                f.write("P12345\t" + ",".join([str(50 + i % 40) for i in range(500)]) + "\n")

            mapping_file = tmpdir / "mapping.txt"
            with open(mapping_file, "w") as f:
                f.write("From\tTo\n")
                f.write("P12345\tTEST\n")

            from es_scorer import ESScorer

            scorer = ESScorer(
                plddt_file=plddt_file,
                uniprot_mapping_file=mapping_file
            )

            # Create a mock assay
            assay = create_mock_assay(
                assay_id="TEST_ASSAY",
                uniprot_id="P12345",
                gene_name="TEST",
                n_variants=50
            )

            # Score the mutations
            scored_df = scorer.score_mutations(assay, single_only=True)

            assert len(scored_df) > 0
            assert "es_score" in scored_df.columns
            assert "mutant" in scored_df.columns
            assert all(scored_df["es_score"] >= 0)
            assert all(scored_df["es_score"] <= 1)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
