"""Tests for the CLI interface."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenbreak_scanner.cli import main


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
        # Mock the diagnostic tokenization probe since no real tokenizer
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



class TestCliOutput:
    def test_table_output(self, tmp_path: Path) -> None:
        """CLI prints a rich table by default."""
        model_dir = tmp_path / "roberta-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "roberta"}')
        (model_dir / "tokenizer_config.json").write_text(
            '{"tokenizer_class": "RobertaTokenizerFast"}'
        )
        (model_dir / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')

        runner = CliRunner()
        result = runner.invoke(main, [str(model_dir)])

        assert result.exit_code == 1  # vulnerable = exit 1
        assert "RoBERTa" in result.output
        assert "BPE" in result.output
        assert "YES" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        """CLI can emit JSON."""
        model_dir = tmp_path / "deberta-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "deberta-v2"}')
        (model_dir / "tokenizer_config.json").write_text(
            '{"tokenizer_class": "DebertaV2Tokenizer"}'
        )
        (model_dir / "tokenizer.json").write_text('{"model": {"type": "Unigram"}}')

        runner = CliRunner()
        result = runner.invoke(main, [str(model_dir), "--output", "json"])

        assert result.exit_code == 0  # safe = exit 0
        data = json.loads(result.output)
        assert data["model_type"] == "deberta-v2"
        assert data["tokenizer_algorithm"] == "Unigram"
        assert data["vulnerable_to_tokenbreak"] is False
        # Heavy metadata must be excluded from CI/CD-friendly JSON output
        assert "config_metadata" not in data
        assert "tokenizer_metadata" not in data

    def test_missing_directory(self) -> None:
        """Non-existent path without --download raises an error."""
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/path"])

        assert result.exit_code == 2
        assert "Error" in result.output

    def test_version_flag(self) -> None:
        from tokenbreak_scanner import __version__
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_batch_mode_no_dirs(self, tmp_path: Path) -> None:
        """Batch mode with no model directories exits gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [str(empty_dir), "--batch"])
        assert result.exit_code == 0
        assert "No model directories found" in result.output

    def test_batch_mode_with_models(self, tmp_path: Path) -> None:
        """Batch mode scans all model subdirectories."""
        for name, model_type in [("roberta-a", "roberta"), ("deberta-b", "deberta-v2")]:
            md = tmp_path / name
            md.mkdir()
            (md / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
            (md / "tokenizer_config.json").write_text(
                '{"tokenizer_class": "RobertaTokenizerFast"}'
                if model_type == "roberta" else '{"tokenizer_class": "DebertaV2Tokenizer"}'
            )
            (md / "tokenizer.json").write_text(
                '{"model": {"type": "BPE"}}'
                if model_type == "roberta" else '{"model": {"type": "Unigram"}}'
            )

        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path), "--batch"])

        assert result.exit_code == 1  # at least one vulnerable
        assert "VULN" in result.output
        assert "SAFE" in result.output
