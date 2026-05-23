"""Integration tests: scan real HuggingFace models over the network.

These tests download tokenizer files and load real tokenizers via
`transformers.AutoTokenizer`.  They validate that the structural
algorithm detection is correct and that the behavior probe never
overrides structural signals.

Run locally with:
    pytest tests/test_integration.py -v

Requires network access and HF Hub token for high-rate-limit models.
"""

import pytest

from tokenbreak_scanner.inspector import inspect_model
from tokenbreak_scanner.models import RiskLevel, TokenizerAlgorithm


@pytest.mark.integration
class TestRealModelIntegration:
    """Download and scan real HuggingFace models."""

    def test_xlm_roberta_base_unigram_structural(self) -> None:
        """FacebookAI/xlm-roberta-base → structural Unigram → LOW risk."""
        report = inspect_model("FacebookAI/xlm-roberta-base", download=True)

        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNIGRAM
        assert report.vulnerable_to_tokenbreak is False
        assert report.risk_level == RiskLevel.LOW
        assert report.confidence_score >= 0.85
        assert "xlm-roberta" in report.model_type.lower()

    def test_xlm_roberta_base_behavioral_does_not_override(self) -> None:
        """Regression: behavior probe must NOT flip Unigram→BPE."""
        report = inspect_model("FacebookAI/xlm-roberta-base", download=True)

        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNIGRAM
        assert report.behavioral_diagnostic is not None
        assert report.behavioral_diagnostic.consistent_with_algorithm is True
        # The report must never contradict itself
        assert "XLM-RoBERTa" not in report.recommendation \
            or "No action needed" in report.recommendation

    def test_qwen3_bpe_structural(self) -> None:
        """Qwen/Qwen3.6-27B → structural BPE → HIGH risk."""
        report = inspect_model("Qwen/Qwen3.6-27B", download=True)

        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.vulnerable_to_tokenbreak is True
        assert report.risk_level == RiskLevel.HIGH
        assert report.confidence_score >= 0.85
        assert "qwen" in report.model_type.lower()

    def test_qwen3_behavioral_present(self) -> None:
        """Qwen BPE model should have a diagnostic probe but not override BPE."""
        report = inspect_model("Qwen/Qwen3.6-27B", download=True)

        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.behavioral_diagnostic is not None
        # BPE can legitimately show high fragility; that's consistent
        assert report.behavioral_diagnostic.consistent_with_algorithm is True
