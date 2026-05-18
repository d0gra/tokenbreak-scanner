"""Model file introspection engine.

Scans downloaded model artifacts (config.json, tokenizer.json, tokenizer_config.json)
to determine tokenizer type, model family, and TokenBreak vulnerability.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer
from transformers.utils import cached_file

from .models import DetectionSource, RiskLevel, ScannerReport, TokenizerAlgorithm
from .tokenizers import (
    detect_from_remote_source,
    detect_from_runtime_tokenizer,
    detect_from_source_code,
    detect_tokenizer_from_config,
    detect_tokenizer_from_json,
    get_model_family,
    get_recommendation,
    is_vulnerable,
)

logger = logging.getLogger(__name__)

# Files we expect to find in a HuggingFace or custom model directory
CONFIG_FILENAME = "config.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
TOKENIZER_JSON_FILENAME = "tokenizer.json"

# Weights for each detection signal (must each be ≤ 1.0; total can exceed 1.0
# because we use a cap-and-normalise strategy).
SIGNAL_WEIGHTS: dict[str, float] = {
    "tokenizer.json model.type": 0.40,
    "runtime._tokenizer.model": 0.40,
    "source_code_fingerprint": 0.30,
    "remote_source_file": 0.30,
    "tokenizer_config.json class": 0.20,
    "config.json model_type": 0.15,
}


def _load_json(path: Path | str) -> dict[str, Any] | None:
    """Safely load a JSON file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _download_custom_tokenizer_files(source: str, resolved_dir: Path) -> None:
    """Download custom tokenizer Python files referenced in config.json auto_map.

    Many custom models (e.g. Nandi, StableLM) define their tokenizer in a
    separate ``tokenization_*.py`` file that is referenced via ``auto_map`` in
    ``config.json``.  Without this file ``AutoTokenizer.from_pretrained`` fails,
    drastically reducing detection confidence.
    """
    custom_files_to_try: set[str] = set()

    # Check config.json auto_map for tokenizer references
    config = _load_json(resolved_dir / CONFIG_FILENAME) or {}
    auto_map = config.get("auto_map", {})
    for key, value in auto_map.items():
        # auto_map values look like "tokenization_nandi.NandiTokenizer"
        if isinstance(value, str) and "." in value:
            module_name = value.split(".")[0]
            custom_files_to_try.add(f"{module_name}.py")

    # Check tokenizer_config.json for auto_map and tokenizer_class
    tok_config = _load_json(resolved_dir / TOKENIZER_CONFIG_FILENAME) or {}
    tok_auto_map = tok_config.get("auto_map", {})
    for key, value in tok_auto_map.items():
        if isinstance(value, str) and "." in value:
            module_name = value.split(".")[0]
            custom_files_to_try.add(f"{module_name}.py")

    # Also try the conventional name: tokenization_{model_type}.py
    model_type = config.get("model_type", "")
    if model_type:
        custom_files_to_try.add(f"tokenization_{model_type}.py")

    # Download each discovered file
    for filename in custom_files_to_try:
        if not (resolved_dir / filename).exists():
            try:
                result = cached_file(source, filename, _raise_exceptions_for_missing_entries=False)
                if result:
                    logger.info("Downloaded custom tokenizer file: %s", filename)
            except Exception:
                logger.debug("Could not download optional file '%s' for '%s'", filename, source)


def _resolve_model_path(source: str, *, download: bool = False) -> Path:
    """Resolve a model identifier to a local path.

    * If `source` is an existing local directory, return it directly.
    * If `source` looks like a HuggingFace / custom model ID and `download` is True,
      attempt to download/cache the tokenizer files via `transformers`.
    * Otherwise raise FileNotFoundError.
    """
    local = Path(source)
    if local.is_dir():
        return local.resolve()

    if download:
        logger.info("Downloading tokenizer files for '%s' from HuggingFace or custom source...", source)
        try:
            # Use cached_file to resolve and download individual files
            config_path = cached_file(source, CONFIG_FILENAME, _raise_exceptions_for_missing_entries=False)
            tokenizer_config_path = cached_file(
                source, TOKENIZER_CONFIG_FILENAME, _raise_exceptions_for_missing_entries=False
            )
            tokenizer_json_path = cached_file(
                source, TOKENIZER_JSON_FILENAME, _raise_exceptions_for_missing_entries=False
            )

            # Determine the snapshot directory
            resolved_dir = None
            if config_path:
                resolved_dir = Path(config_path).parent.resolve()
            elif tokenizer_config_path:
                resolved_dir = Path(tokenizer_config_path).parent.resolve()
            elif tokenizer_json_path:
                resolved_dir = Path(tokenizer_json_path).parent.resolve()

            # Also download custom tokenizer Python files referenced in
            # config.json auto_map or tokenizer_config.json (e.g.
            # tokenization_nandi.py for the Nandi model).
            if resolved_dir is not None:
                _download_custom_tokenizer_files(source, resolved_dir)
                return resolved_dir

        except Exception as exc:
            raise FileNotFoundError(
                f"Could not download or cache model '{source}' from HuggingFace or custom source."
            ) from exc

    raise FileNotFoundError(
        f"Model path not found: '{source}'. "
        "Provide a valid local directory or use --download to fetch from HuggingFace or custom source."
    )


def inspect_model(
    source: str,
    *,
    download: bool = False,
    trust_remote_code: bool = False,
) -> ScannerReport:
    """Inspect a model directory or HuggingFace/custom model ID and return an AI supply chain vulnerability report.

    Parameters
    ----------
    source
        Local model directory path or HuggingFace/custom model ID (e.g. ``distilbert-base-uncased``).
    download
        If True and ``source`` is a HuggingFace/custom model ID, download tokenizer files.
    trust_remote_code
        Passed through to ``transformers.AutoTokenizer`` when probing vocab size.
    """
    model_path = _resolve_model_path(source, download=download)

    # Load available metadata files
    config = _load_json(model_path / CONFIG_FILENAME) or {}
    tokenizer_config = _load_json(model_path / TOKENIZER_CONFIG_FILENAME) or {}
    tokenizer_json = _load_json(model_path / TOKENIZER_JSON_FILENAME)

    # Extract model type
    model_type = config.get("model_type", "")
    model_family = get_model_family(model_type)

    # ── Detection: collect signals, then aggregate ──
    sources: list[DetectionSource] = []

    # Signal 1: tokenizer.json "model.type" - most reliable
    if tokenizer_json is not None:
        algo = detect_tokenizer_from_json(tokenizer_json)
        if algo is not None:
            sources.append(
                DetectionSource(
                    signal="tokenizer.json model.type",
                    value=str(tokenizer_json.get("model", {}).get("type") or tokenizer_json.get("type")),
                    inferred=algo.value,
                    weight=SIGNAL_WEIGHTS["tokenizer.json model.type"],
                    reason="Direct algorithm type from tokenizers library metadata",
                )
            )
            logger.debug("Tokenizer algorithm detected from tokenizer.json: %s", algo)

    # Signal 2: Attempt to load AutoTokenizer and inspect Rust backend
    loaded_tokenizer: Optional[Any] = None
    vocab_size: Optional[int] = None
    tok_cls_name: str = "unknown"
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        vocab_size = len(loaded_tokenizer)
        tok_cls_name = loaded_tokenizer.__class__.__name__
    except Exception as exc:
        logger.warning("Could not load tokenizer via AutoTokenizer: %s", exc)
        # Fallback: load tokenizer.json directly via the tokenizers library.
        # This handles custom architectures (e.g. Nandi) that don't ship
        # a tokenization_*.py file but DO have a valid tokenizer.json.
        tokenizer_json_path = model_path / TOKENIZER_JSON_FILENAME
        if tokenizer_json_path.exists():
            try:
                from tokenizers import Tokenizer as HFTokenizer

                raw_tok = HFTokenizer.from_file(str(tokenizer_json_path))
                vocab_size = raw_tok.get_vocab_size()
                tok_cls_name = type(raw_tok.model).__name__ + "Tokenizer (direct)"
                # Also collect the runtime signal from the raw tokenizer
                runtime_type = type(raw_tok.model).__name__
                from .tokenizers import RUNTIME_MODEL_TYPE_MAP

                runtime_algo = RUNTIME_MODEL_TYPE_MAP.get(runtime_type)
                if runtime_algo is not None:
                    sources.append(
                        DetectionSource(
                            signal="runtime._tokenizer.model",
                            value=runtime_type,
                            inferred=runtime_algo.value,
                            weight=SIGNAL_WEIGHTS["runtime._tokenizer.model"],
                            reason="Direct tokenizers library model type (AutoTokenizer fallback)",
                        )
                    )
                logger.info("Recovered tokenizer info via direct tokenizer.json load: vocab=%s, type=%s", vocab_size, runtime_type)
            except Exception as fallback_exc:
                logger.warning("Direct tokenizer.json fallback also failed: %s", fallback_exc)

    if loaded_tokenizer is not None:
        algo, reason = detect_from_runtime_tokenizer(loaded_tokenizer)
        if algo is not None:
            sources.append(
                DetectionSource(
                    signal="runtime._tokenizer.model",
                    value=reason,
                    inferred=algo.value,
                    weight=SIGNAL_WEIGHTS["runtime._tokenizer.model"],
                    reason="Rust fast-tokenizer backend model type",
                )
            )

        # Signal 3: source-code fingerprint (if inspect.getsource succeeds)
        algo_src, reason_src = detect_from_source_code(loaded_tokenizer)
        if algo_src is not None:
            sources.append(
                DetectionSource(
                    signal="source_code_fingerprint",
                    value=reason_src,
                    inferred=algo_src.value,
                    weight=SIGNAL_WEIGHTS["source_code_fingerprint"],
                    reason="Keyword fingerprinting on tokenizer class source",
                )
            )

    # Signal 4: tokenizer_config.json → tokenizer_class / model_type
    algo_cfg = detect_tokenizer_from_config(tokenizer_config)
    if algo_cfg is not None:
        sources.append(
            DetectionSource(
                signal="tokenizer_config.json class",
                value=tokenizer_config.get("tokenizer_class", tokenizer_config.get("model_type", "")),
                inferred=algo_cfg.value,
                weight=SIGNAL_WEIGHTS["tokenizer_config.json class"],
                reason="Tokenizer class name or model_type from tokenizer_config.json",
            )
        )

    # Signal 5: config.json model_type fallback
    if model_type:
        from .tokenizers import MODEL_TYPE_MAP

        algo_meta = MODEL_TYPE_MAP.get(model_type)
        if algo_meta is not None:
            sources.append(
                DetectionSource(
                    signal="config.json model_type",
                    value=model_type,
                    inferred=algo_meta.value,
                    weight=SIGNAL_WEIGHTS["config.json model_type"],
                    reason="Architecture model_type from config.json",
                )
            )

    # Signal 6: remote source file for trust_remote_code models
    algo_remote, reason_remote = detect_from_remote_source(model_path, trust_remote_code=trust_remote_code)
    if algo_remote is not None:
        sources.append(
            DetectionSource(
                signal="remote_source_file",
                value=reason_remote,
                inferred=algo_remote.value,
                weight=SIGNAL_WEIGHTS["remote_source_file"],
                reason="Tokenization Python module downloaded from HF Hub",
            )
        )

    # ── Aggregate weighted votes ──
    algorithm = _aggregate_signals(sources)
    confidence_score = _confidence_from_sources(sources)

    # Risk assessment
    vulnerable = is_vulnerable(algorithm)
    risk_level = RiskLevel.HIGH if vulnerable else RiskLevel.LOW
    if algorithm == TokenizerAlgorithm.UNKNOWN:
        risk_level = RiskLevel.UNKNOWN

    recommendation = get_recommendation(algorithm)

    return ScannerReport(
        model_name=Path(source).name if Path(source).exists() else source,
        model_type=model_type or "unknown",
        model_family=model_family,
        tokenizer_class=tok_cls_name,
        tokenizer_algorithm=algorithm,
        vocab_size=vocab_size,
        vulnerable_to_tokenbreak=vulnerable,
        risk_level=risk_level,
        confidence_score=round(confidence_score, 3),
        detection_sources=sources,
        recommendation=recommendation,
        source=str(model_path),
        config_metadata={
            "config.json": config,
            "tokenizer_config.json": tokenizer_config,
            "detection_confidence": confidence_score,
        },
        tokenizer_metadata=tokenizer_json or {},
    )


def _aggregate_signals(sources: list[DetectionSource]) -> TokenizerAlgorithm:
    """Weighted-majority vote over detection signals.

    Each source contributes ``weight`` points to the algorithm it inferred.
    The algorithm with the highest total weight wins.  If no votes were cast,
    returns :attr:`TokenizerAlgorithm.UNKNOWN`.
    """
    from collections import defaultdict

    votes: dict[TokenizerAlgorithm, float] = defaultdict(float)
    for src in sources:
        if src.inferred:
            try:
                algo = TokenizerAlgorithm(src.inferred)
            except ValueError:
                continue
            votes[algo] += src.weight

    if not votes:
        return TokenizerAlgorithm.UNKNOWN

    best_algo = max(votes, key=lambda a: votes[a])
    return best_algo


def _confidence_from_sources(sources: list[DetectionSource]) -> float:
    """Cap-and-normalise confidence from evidence.

    Sum raw weights, then clamp to ``[0, 1]``.  This is deliberately simple so
    that adding more signals cannot push confidence past certainty.
    """
    total = sum(src.weight for src in sources)
    return min(total, 1.0)
