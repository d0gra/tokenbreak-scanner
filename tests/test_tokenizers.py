"""Tests for tokenizer type detection logic."""

import pytest

from tokenbreak_scanner.models import TokenizerAlgorithm
from tokenbreak_scanner.tokenizers import (
    MODEL_FAMILY_MAP,
    MODEL_TYPE_MAP,
    TOKENIZER_CLASS_MAP,
    detect_tokenizer_from_config,
    detect_tokenizer_from_json,
    get_model_family,
    get_recommendation,
    is_vulnerable,
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
        assert "structurally immune" in rec

    def test_bpe(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.BPE)
        assert "VULNERABILITY" in rec
        assert "Pre-Mapping Defense" in rec

    def test_wordpiece(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.WORDPIECE)
        assert "VULNERABILITY" in rec

    def test_unknown_recommendation(self) -> None:
        rec = get_recommendation(TokenizerAlgorithm.UNKNOWN)
        assert "manual architectural audit" in rec


class TestMapCompleteness:
    def test_all_model_types_have_family(self) -> None:
        for model_type in MODEL_TYPE_MAP:
            assert model_type in MODEL_FAMILY_MAP or model_type.replace("-", "_") in [
                k.replace("-", "_") for k in MODEL_FAMILY_MAP
            ]
