"""Tests for tokenizer type detection logic."""

import pytest

from tokenbreak_scanner.models import TokenizerAlgorithm
from tokenbreak_scanner.tokenizers import (
    MODEL_FAMILY_MAP,
    MODEL_TYPE_MAP,
    TOKENIZER_CLASS_MAP,
    _is_behavior_consistent_with_algorithm,
    detect_from_tokenization_behavior,
    detect_tokenizer_from_config,
    detect_tokenizer_from_json,
    get_model_family,
    get_recommendation,
    is_vulnerable,
    resolve_sentencepiece_algorithm,
)


class TestDetectTokenizerFromJson:
    def test_bpe_from_tokenizer_json(self) -> None:
        data = {"model": {"type": "BPE"}}
        assert detect_tokenizer_from_json(data) == TokenizerAlgorithm.BPE

    def test_wordpiece_from_tokenizer_json(self) -> None:
        data = {"model": {"type": "WordPiece"}}
        assert detect_tokenizer_from_json(data) == TokenizerAlgorithm.WORDPIECE

    def test_unigram_from_tokenizer_json(self) -> None:
        data = {"model": {"type": "Unigram"}}
        assert detect_tokenizer_from_json(data) == TokenizerAlgorithm.UNIGRAM

    def test_sentencepiece_from_tokenizer_json(self) -> None:
        data = {"model": {"type": "SentencePiece"}}
        assert detect_tokenizer_from_json(data) == TokenizerAlgorithm.SENTENCEPIECE

    def test_top_level_type_fallback(self) -> None:
        data = {"type": "BPE"}
        assert detect_tokenizer_from_json(data) == TokenizerAlgorithm.BPE

    def test_missing_type_returns_none(self) -> None:
        data = {"model": {"vocab": []}}
        assert detect_tokenizer_from_json(data) is None


class TestDetectTokenizerFromConfig:
    def test_roberta_tokenizer_class(self) -> None:
        config = {"tokenizer_class": "RobertaTokenizerFast"}
        assert detect_tokenizer_from_config(config) == TokenizerAlgorithm.BPE

    def test_bert_tokenizer_class(self) -> None:
        config = {"tokenizer_class": "BertTokenizerFast"}
        assert detect_tokenizer_from_config(config) == TokenizerAlgorithm.WORDPIECE

    def test_deberta_tokenizer_class(self) -> None:
        config = {"tokenizer_class": "DebertaV2Tokenizer"}
        assert detect_tokenizer_from_config(config) == TokenizerAlgorithm.UNIGRAM

    def test_model_type_fallback(self) -> None:
        config = {"model_type": "roberta"}
        assert detect_tokenizer_from_config(config) == TokenizerAlgorithm.BPE

    def test_unknown_returns_none(self) -> None:
        config = {"tokenizer_class": "CustomTokenizer"}
        assert detect_tokenizer_from_config(config) is None


class TestModelFamilyMap:
    def test_known_families(self) -> None:
        assert get_model_family("roberta") == "RoBERTa"
        assert get_model_family("bert") == "BERT"
        assert get_model_family("distilbert") == "DistilBERT"
        assert get_model_family("deberta-v2") == "DeBERTa-v2"

    def test_unknown_family(self) -> None:
        assert get_model_family("custom") == "Custom"


class TestVulnerability:
    def test_bpe_is_vulnerable(self) -> None:
        assert is_vulnerable(TokenizerAlgorithm.BPE) is True

    def test_wordpiece_is_vulnerable(self) -> None:
        assert is_vulnerable(TokenizerAlgorithm.WORDPIECE) is True

    def test_unigram_is_not_vulnerable(self) -> None:
        assert is_vulnerable(TokenizerAlgorithm.UNIGRAM) is False

    def test_sentencepiece_is_not_vulnerable(self) -> None:
        assert is_vulnerable(TokenizerAlgorithm.SENTENCEPIECE) is False


class TestRecommendation:
    def test_unigram(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.UNIGRAM)
        assert "No action needed" in rec
        assert "resistant" in rec

    def test_bpe(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.BPE)
        assert "vulnerable" in rec
        assert "Unigram-based" in rec

    def test_wordpiece(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.WORDPIECE)
        assert "vulnerable" in rec

    def test_unknown_recommendation(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.UNKNOWN)
        assert "Could not determine" in rec


class TestMapCompleteness:
    def test_all_model_types_have_family(self) -> None:
        for model_type in MODEL_TYPE_MAP:
            assert model_type in MODEL_FAMILY_MAP or model_type.replace("-", "_") in [
                k.replace("-", "_") for k in MODEL_FAMILY_MAP
            ]

    def test_no_duplicate_keys_in_tokenizer_class_map(self) -> None:
        """Verify TOKENIZER_CLASS_MAP has no duplicate keys."""
        # Python dicts can't have duplicate keys, but we can verify
        # that the mapping is consistent (no BPE mapped tokenizer also
        # appearing in the Unigram section, etc.)
        bpe_keys = {k for k, v in TOKENIZER_CLASS_MAP.items() if v == TokenizerAlgorithm.BPE}
        wp_keys = {k for k, v in TOKENIZER_CLASS_MAP.items() if v == TokenizerAlgorithm.WORDPIECE}
        unigram_keys = {k for k, v in TOKENIZER_CLASS_MAP.items() if v == TokenizerAlgorithm.UNIGRAM}
        sp_keys = {k for k, v in TOKENIZER_CLASS_MAP.items() if v == TokenizerAlgorithm.SENTENCEPIECE}
        # No tokenizer should be in multiple categories
        assert not (bpe_keys & wp_keys), f"BPE+WordPiece overlap: {bpe_keys & wp_keys}"
        assert not (bpe_keys & unigram_keys), f"BPE+Unigram overlap: {bpe_keys & unigram_keys}"
        assert not (wp_keys & unigram_keys), f"WordPiece+Unigram overlap: {wp_keys & unigram_keys}"


class TestTokenizationBehavior:
    """Test the diagnostic tokenization sensitivity probe."""

    def test_none_tokenizer_returns_empty(self) -> None:
        """None tokenizer should return empty diagnostic."""
        result = detect_from_tokenization_behavior(None)
        assert result["shifted"] == 0
        assert result["fragility"] == 0.0
        assert "No tokenizer" in result["detail"]

    def test_mock_tokenizer_encoding(self) -> None:
        """Test with a mock tokenizer that simulates BPE fragility."""
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Simulate BPE: invisible prepend changes tokenization
                if text.startswith("\u200bpassword"):
                    return [100, 200, 300, 400, 500]  # Different tokens!
                if text == "password":
                    return [200, 300, 400, 500]
                # All other words: no shift
                return [42] * len(text)

        result = detect_from_tokenization_behavior(MockTokenizer())
        assert result["shifted"] > 0
        assert result["fragility"] > 0.0
        assert "password" in result["detail"]

    def test_mock_unigram_tokenizer(self) -> None:
        """Test with a mock tokenizer that simulates Unigram stability."""
        class MockUnigramTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Simulate Unigram: prepend changes length but word portion stable
                # Prepend adds one char → one token at start, rest same
                if text.startswith("\u200b"):
                    base = text[1:]  # strip the zero-width char
                    base_ids = [hash(c) % 1000 + 1000 for c in base]
                    return [999] + base_ids
                tokens = [hash(c) % 1000 + 1000 for c in text]
                return tokens

        result = detect_from_tokenization_behavior(MockUnigramTokenizer())
        assert result["shifted"] == 0
        assert result["fragility"] == 0.0
        assert "No shifts detected" in result["detail"]

    def test_short_perturbed_sequence_skipped(self) -> None:
        """When perturbed sequence is same length or shorter, count as tested but no shift."""
        class ShortTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Always return same-length encoding
                return [1] * len(text)

        result = detect_from_tokenization_behavior(ShortTokenizer())
        # Should not crash; tests were run but all matched
        assert result["shifted"] == 0
        assert result["fragility"] == 0.0

    def test_error_in_tokenization_is_handled(self) -> None:
        """When tokenizer.encode raises, return gracefully."""
        class BrokenTokenizer:
            def encode(self, text, add_special_tokens=False):
                raise RuntimeError("tokenizer crash")

        result = detect_from_tokenization_behavior(BrokenTokenizer())
        assert result["shifted"] == 0
        assert result["fragility"] == 0.0
        assert "error" in result["detail"].lower()


class TestSentencePieceResolution:
    """Test SentencePiece ambiguity resolution."""

    def test_no_model_file_returns_none(self, tmp_path) -> None:
        """When no .model file exists, return None."""
        import pathlib
        model_dir = pathlib.Path(tmp_path) / "no-sp-model"
        model_dir.mkdir()
        result = resolve_sentencepiece_algorithm(model_dir)
        assert result is None
