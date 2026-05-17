"""Pydantic data models for scanner reports."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TokenizerAlgorithm(str, Enum):
    """Known tokenizer algorithms relevant to TokenBreak."""

    BPE = "BPE"
    WORDPIECE = "WordPiece"
    UNIGRAM = "Unigram"
    SENTENCEPIECE = "SentencePiece"
    UNKNOWN = "Unknown"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    LOW = "Low"
    HIGH = "High"
    UNKNOWN = "Unknown"


class DetectionSource(BaseModel):
    """A single piece of evidence that contributed to the algorithm detection."""

    signal: str = Field(description="Name of the detection signal")
    value: Optional[str] = Field(default=None, description="Raw value returned by the signal")
    inferred: Optional[str] = Field(default=None, description="Algorithm inferred from this signal")
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence weight of this signal")
    reason: str = Field(default="", description="Human-readable explanation")


class ScannerReport(BaseModel):
    """Complete report for a scanned model."""

    model_name: str = Field(description="Name or identifier of the model")
    model_type: str = Field(description="Model architecture type (e.g., roberta, bert)")
    model_family: str = Field(description="High-level model family (e.g., RoBERTa, BERT)")
    tokenizer_class: str = Field(description="Tokenizer class name (e.g., RobertaTokenizerFast)")
    tokenizer_algorithm: TokenizerAlgorithm = Field(description="Detected tokenizer algorithm")
    vocab_size: Optional[int] = Field(default=None, description="Tokenizer vocabulary size")
    vulnerable_to_tokenbreak: bool = Field(description="Whether model is vulnerable to TokenBreak")
    risk_level: RiskLevel = Field(description="Risk level assessment")
    confidence_score: float = Field(
        default=0.0, ge=-0.01, le=1.01,
        description="Aggregated confidence score (0.0–1.0) for the detection",
    )
    detection_sources: List[DetectionSource] = Field(
        default_factory=list,
        description="Evidence tree showing why the algorithm was detected",
    )
    recommendation: str = Field(description="Remediation recommendation")
    source: str = Field(description="Source of scan: local path or HuggingFace/Custom Model ID")
    config_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata from config.json and tokenizer_config.json",
    )
    tokenizer_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metadata from tokenizer.json",
    )
