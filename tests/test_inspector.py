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
        patch(
            "tokenbreak_scanner.inspector.detect_from_tokenization_behavior",
            return_value={
                "shifted": 0,
                "total": 80,
                "fragility": 0.0,
                "inconsistent_with_unigram": False,
                "detail": "mocked: no tokenizer available",
            },
        ),
        patch(
            "tokenbreak_scanner.inspector._is_behavior_consistent_with_algorithm",
            return_value=True,
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


    def test_nested_text_config_model_type(self, tmp_path: Path) -> None:
        """Multi-modal models with nested text_config should extract model_type."""
        model_dir = tmp_path / "lance-like"
        model_dir.mkdir()

        config = {
            "model_type": "lance",
            "text_config": {
                "model_type": "qwen2",
                "_name_or_path": "Qwen/Qwen2-7B",
            },
        }
        tokenizer_config = {"tokenizer_class": "Qwen2TokenizerFast"}
        tokenizer_json = {"model": {"type": "BPE", "vocab": {}}}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        # "lance" is still the primary model_type from config
        assert report.model_type == "lance"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.vulnerable_to_tokenbreak is True

    def test_config_only_model_graceful(self, tmp_path: Path) -> None:
        """Models shipping only config.json (no tokenizer files) should not crash."""
        model_dir = tmp_path / "config-only"
        model_dir.mkdir()

        config = {
            "model_type": "some_vision_model",
            "text_config": {
                "model_type": "llama",
                "_name_or_path": "meta-llama/Llama-2-7b",
            },
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        report = inspect_model(str(model_dir))

        # Should fall back to nested text_config model_type
        # and at minimum not crash
        assert report.model_type == "some_vision_model"

    def test_nested_text_config_model_type_used_for_family(self, tmp_path: Path) -> None:
        """When top-level model_type is unknown, nested model_type should inform detection."""
        model_dir = tmp_path / "nested-fallback"
        model_dir.mkdir()

        config = {
            "text_config": {
                "model_type": "bert",
            },
        }
        tokenizer_config = {"tokenizer_class": "BertTokenizerFast"}
        tokenizer_json = {"model": {"type": "WordPiece", "vocab": {}}}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        # When no top-level model_type, nested text_config.model_type is used
        assert report.model_type == "bert"
        assert report.tokenizer_algorithm == TokenizerAlgorithm.WORDPIECE


class TestFileNotFound:
    def test_nonexistent_path_no_download(self) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_model("/definitely/does/not/exist", download=False)


class TestDecisionTree:
    """Test the new decision-tree-based algorithm selection."""

    def test_tokenizer_json_wins_over_config(self, tmp_path: Path) -> None:
        """tokenizer.json should take priority over config.json model_type."""
        model_dir = tmp_path / "conflict-model"
        model_dir.mkdir()

        # config.json says BERT (WordPiece), but tokenizer.json says BPE
        config = {"model_type": "bert"}
        tokenizer_config = {"tokenizer_class": "BertTokenizerFast"}
        tokenizer_json = {"model": {"type": "BPE"}}  # Override!

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        # tokenizer.json wins → BPE
        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.vulnerable_to_tokenbreak is True
        # Confidence should be high — structural signals are trusted
        assert report.confidence_score >= 0.70

    def test_no_tokenizer_json_falls_back_to_runtime(self, tmp_path: Path) -> None:
        """Without tokenizer.json, config signals determine algorithm."""
        model_dir = tmp_path / "runtime-fallback"
        model_dir.mkdir()

        config = {"model_type": "roberta"}
        tokenizer_config = {"tokenizer_class": "RobertaTokenizerFast"}
        # No tokenizer.json

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

        report = inspect_model(str(model_dir))

        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.vulnerable_to_tokenbreak is True

    def test_confidence_drops_with_conflicting_signals(self, tmp_path: Path) -> None:
        """When metadata signals disagree, confidence should be lower."""
        model_dir = tmp_path / "messy-model"
        model_dir.mkdir()

        # Deliberately conflicting: model_type=BERT but tokenizer class=RobertaTokenizer
        config = {"model_type": "bert"}
        tokenizer_config = {"tokenizer_class": "RobertaTokenizerFast"}
        # No tokenizer.json

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

        report = inspect_model(str(model_dir))

        # With conflicting signals and no tokenizer.json, confidence should be moderate
        assert report.confidence_score < 0.90


class TestBehavioralDiagnostic:
    """Test that the behavioral diagnostic probe is reported independently."""

    def test_behavioral_diagnostic_none_when_no_tokenizer(self, tmp_path: Path) -> None:
        """When no tokenizer can be loaded, behavioral_diagnostic should be None."""
        model_dir = tmp_path / "no-tok"
        model_dir.mkdir()

        tokenizer_json = {"model": {"type": "Unigram"}}
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))
        # AutoTokenizer fails in mock → no loaded_tokenizer → no diagnostic
        assert report.behavioral_diagnostic is None

    def test_deberta_unigram_no_behavioral_override(self, tmp_path: Path) -> None:
        """XLM-RoBERTa/DeBERTa Unigram models should not be overridden by behavior probe."""
        model_dir = tmp_path / "unigram-structural"
        model_dir.mkdir()

        tokenizer_json = {"model": {"type": "Unigram"}}
        tokenizer_config = {"tokenizer_class": "XLMRobertaTokenizer"}
        config = {"model_type": "xlm-roberta"}

        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))
        # Structural signals say Unigram → Unigram, regardless of any mock behavior
        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNIGRAM
        assert report.vulnerable_to_tokenbreak is False
        assert report.risk_level == RiskLevel.LOW


class TestConfidenceModel:
    """Test the new confidence scoring."""

    def test_tokenizer_json_only_high_confidence(self, tmp_path: Path) -> None:
        """tokenizer.json alone should give high confidence."""
        model_dir = tmp_path / "json-only"
        model_dir.mkdir()

        tokenizer_json = {"model": {"type": "BPE", "vocab": {}}}
        (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        report = inspect_model(str(model_dir))

        assert report.tokenizer_algorithm == TokenizerAlgorithm.BPE
        assert report.confidence_score >= 0.85

    def test_empty_dir_low_confidence(self, tmp_path: Path) -> None:
        """Empty directory should have very low confidence."""
        model_dir = tmp_path / "empty-conf"
        model_dir.mkdir()

        report = inspect_model(str(model_dir))

        assert report.tokenizer_algorithm == TokenizerAlgorithm.UNKNOWN
        assert report.confidence_score <= 0.20
        assert report.risk_level == RiskLevel.UNKNOWN
