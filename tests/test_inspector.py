"""Tests for the model inspector."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tokenbreak_scanner.inspector import inspect_model
from tokenbreak_scanner.models import RiskLevel, TokenizerAlgorithm


@pytest.fixture(autouse=True)
def _mock_heavy_deps():
    """Mock transformers & network calls so tests are fast and CI-safe."""
    with (
        patch(
            "tokenbreak_scanner.inspector.AutoTokenizer.from_pretrained",
            side_effect=OSError("mocked: no real tokenizer in test"),
        ),
        patch(
            "tokenbreak_scanner.inspector.detect_from_remote_source",
            return_value=(None, ""),
        ),
    ):
        yield



class TestInspectModelLocal:
    def test_roberta_style_vulnerable(self, tmp_path: Path) -> None:
        """Simulate a RoBERTa model directory with BPE tokenizer."""
        model_dir = tmp_path / "roberta-vuln"
        model_dir.mkdir()

        config = {"model_type": "roberta", "architectures": ["RobertaForSequenceClassification"]}
        tokenizer_config = {"tokenizer_class": "RobertaTokenizerFast"}
        tokenizer_json = {"model": {"type": "BPE", "vocab": {}}}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        assert report.model_type == "roberta"
        assert report.model_family == "RoBERTa"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.vulnerable_to_tokenbreak is True
        assert report.risk_level == RiskLevel.HIGH
        assert "vulnerable" in report.recommendation.lower()

    def test_deberta_safe(self, tmp_path: Path) -> None:
        """Simulate a DeBERTa-v2 model directory with Unigram tokenizer."""
        model_dir = tmp_path / "deberta-safe"
        model_dir.mkdir()

        config = {"model_type": "deberta-v2", "architectures": ["DebertaV2ForSequenceClassification"]}
        tokenizer_config = {"tokenizer_class": "DebertaV2Tokenizer"}
        tokenizer_json = {"model": {"type": "Unigram", "vocab": {}}}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        assert report.model_type == "deberta-v2"
        assert report.model_family == "DeBERTa-v2"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNIGRAM
        assert report.vulnerable_to_tokenbreak is False
        assert report.risk_level == RiskLevel.LOW
        assert "no action needed" in report.recommendation.lower()

    def test_distilbert_vulnerable(self, tmp_path: Path) -> None:
        """Simulate a DistilBERT model directory with WordPiece tokenizer."""
        model_dir = tmp_path / "distilbert-vuln"
        model_dir.mkdir()

        config = {"model_type": "distilbert", "architectures": ["DistilBertForSequenceClassification"]}
        tokenizer_config = {"tokenizer_class": "DistilBertTokenizerFast"}
        tokenizer_json = {"model": {"type": "WordPiece", "vocab": {}}}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        assert report.model_type == "distilbert"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.WORDPIECE
        assert report.vulnerable_to_tokenbreak is True
        assert report.risk_level == RiskLevel.HIGH

    def test_no_tokenizer_json_fallback(self, tmp_path: Path) -> None:
        """When tokenizer.json is missing, fall back to tokenizer_config.json."""
        model_dir = tmp_path / "bert-no-tokenizer-json"
        model_dir.mkdir()

        config = {"model_type": "bert"}
        tokenizer_config = {"tokenizer_class": "BertTokenizerFast"}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        # No tokenizer.json

        report = inspect_model(str(model_dir))

        assert report.tokenizer_algorithm == TokenizerAlgorithm.WORDPIECE
        assert report.vulnerable_to_tokenbreak is True

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """An empty directory should be handled gracefully."""
        model_dir = tmp_path / "empty"
        model_dir.mkdir()

        report = inspect_model(str(model_dir))

        assert report.model_type == "unknown"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNKNOWN
        assert report.risk_level == RiskLevel.UNKNOWN

    def test_unknown_tokenizer_class(self, tmp_path: Path) -> None:
        """Custom tokenizer classes should result in UNKNOWN."""
        model_dir = tmp_path / "custom-model"
        model_dir.mkdir()

        config = {"model_type": "custom"}
        tokenizer_config = {"tokenizer_class": "MyCustomTokenizer"}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

        report = inspect_model(str(model_dir))

        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNKNOWN
        assert report.risk_level == RiskLevel.UNKNOWN


class TestFileNotFound:
    def test_nonexistent_path_no_download(self) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_model("/definitely/does/not/exist", download=False)
