"""Tokenizer type detection and model-family mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .models import TokenizerAlgorithm

logger = logging.getLogger(__name__)

# Mapping from HuggingFace tokenizer class names → algorithm
TOKENIZER_CLASS_MAP: dict[str, TokenizerAlgorithm] = {
    # BPE family
    "RobertaTokenizer": TokenizerAlgorithm.BPE,
    "RobertaTokenizerFast": TokenizerAlgorithm.BPE,
    "GPT2Tokenizer": TokenizerAlgorithm.BPE,
    "GPT2TokenizerFast": TokenizerAlgorithm.BPE,
    "BartTokenizer": TokenizerAlgorithm.BPE,
    "BartTokenizerFast": TokenizerAlgorithm.BPE,
    "LlamaTokenizer": TokenizerAlgorithm.BPE,
    "LlamaTokenizerFast": TokenizerAlgorithm.BPE,
    "CodeLlamaTokenizer": TokenizerAlgorithm.BPE,
    "CodeLlamaTokenizerFast": TokenizerAlgorithm.BPE,
    "PreTrainedTokenizerFast": TokenizerAlgorithm.BPE,  # Often BPE (RoBERTa-style)
    "Qwen2Tokenizer": TokenizerAlgorithm.BPE,
    "Qwen2TokenizerFast": TokenizerAlgorithm.BPE,
    "GPTNeoXTokenizer": TokenizerAlgorithm.BPE,
    "GPTNeoXTokenizerFast": TokenizerAlgorithm.BPE,
    "GemmaTokenizer": TokenizerAlgorithm.BPE,
    "GemmaTokenizerFast": TokenizerAlgorithm.BPE,
    "Phi3Tokenizer": TokenizerAlgorithm.BPE,
    "Phi3TokenizerFast": TokenizerAlgorithm.BPE,
    "CohereTokenizer": TokenizerAlgorithm.BPE,
    "CohereTokenizerFast": TokenizerAlgorithm.BPE,
    "BloomTokenizer": TokenizerAlgorithm.BPE,
    "BloomTokenizerFast": TokenizerAlgorithm.BPE,
    "OLMoTokenizer": TokenizerAlgorithm.BPE,
    "OLMoTokenizerFast": TokenizerAlgorithm.BPE,
    "SmolLMTokenizer": TokenizerAlgorithm.BPE,
    "SmolLMTokenizerFast": TokenizerAlgorithm.BPE,
    "JaisTokenizer": TokenizerAlgorithm.BPE,
    "JaisTokenizerFast": TokenizerAlgorithm.BPE,
    "NemotronTokenizer": TokenizerAlgorithm.BPE,
    "NemotronTokenizerFast": TokenizerAlgorithm.BPE,
    "AyaTokenizer": TokenizerAlgorithm.BPE,
    "AyaTokenizerFast": TokenizerAlgorithm.BPE,
    # WordPiece family
    "BertTokenizer": TokenizerAlgorithm.WORDPIECE,
    "BertTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "DistilBertTokenizer": TokenizerAlgorithm.WORDPIECE,
    "DistilBertTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "ElectraTokenizer": TokenizerAlgorithm.WORDPIECE,
    "ElectraTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "MobileBertTokenizer": TokenizerAlgorithm.WORDPIECE,
    "MobileBertTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "BigBirdTokenizer": TokenizerAlgorithm.WORDPIECE,
    "BigBirdTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "ConvBertTokenizer": TokenizerAlgorithm.WORDPIECE,
    "ConvBertTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "FunnelTokenizer": TokenizerAlgorithm.WORDPIECE,
    "FunnelTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "LayoutLMTokenizer": TokenizerAlgorithm.WORDPIECE,
    "LayoutLMTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "LxmertTokenizer": TokenizerAlgorithm.WORDPIECE,
    "LxmertTokenizerFast": TokenizerAlgorithm.WORDPIECE,
    "AlbertTokenizer": TokenizerAlgorithm.UNIGRAM,
    "AlbertTokenizerFast": TokenizerAlgorithm.UNIGRAM,
    "DebertaV2Tokenizer": TokenizerAlgorithm.UNIGRAM,
    "DebertaV2TokenizerFast": TokenizerAlgorithm.UNIGRAM,
    "XLMRobertaTokenizer": TokenizerAlgorithm.UNIGRAM,
    "XLMRobertaTokenizerFast": TokenizerAlgorithm.UNIGRAM,
    "T5Tokenizer": TokenizerAlgorithm.SENTENCEPIECE,
    "T5TokenizerFast": TokenizerAlgorithm.SENTENCEPIECE,
    "MT5Tokenizer": TokenizerAlgorithm.SENTENCEPIECE,
    "MT5TokenizerFast": TokenizerAlgorithm.SENTENCEPIECE,
    "LlamaTokenizer": TokenizerAlgorithm.BPE,  # Modern LLaMA is BPE
    "LlamaTokenizerFast": TokenizerAlgorithm.BPE,
}

# Mapping from model_type (config.json) → expected algorithm
MODEL_TYPE_MAP: dict[str, TokenizerAlgorithm] = {
    "roberta": TokenizerAlgorithm.BPE,
    "gpt2": TokenizerAlgorithm.BPE,
    "gpt_neo": TokenizerAlgorithm.BPE,
    "gpt_neox": TokenizerAlgorithm.BPE,
    "gptj": TokenizerAlgorithm.BPE,
    "llama": TokenizerAlgorithm.BPE,
    "mistral": TokenizerAlgorithm.BPE,
    "mixtral": TokenizerAlgorithm.BPE,
    "falcon": TokenizerAlgorithm.BPE,
    "qwen2": TokenizerAlgorithm.BPE,
    "qwen2_5": TokenizerAlgorithm.BPE,
    "qwen3": TokenizerAlgorithm.BPE,
    "gemma": TokenizerAlgorithm.BPE,
    "gemma2": TokenizerAlgorithm.BPE,
    "phi3": TokenizerAlgorithm.BPE,
    "phi4": TokenizerAlgorithm.BPE,
    "cohere": TokenizerAlgorithm.BPE,
    "command-r": TokenizerAlgorithm.BPE,
    "bloom": TokenizerAlgorithm.BPE,
    "bigscience": TokenizerAlgorithm.BPE,
    "olmo": TokenizerAlgorithm.BPE,
    "olmoe": TokenizerAlgorithm.BPE,
    "smollm": TokenizerAlgorithm.BPE,
    "jais": TokenizerAlgorithm.BPE,
    "nemotron": TokenizerAlgorithm.BPE,
    "aya": TokenizerAlgorithm.BPE,
    "bert": TokenizerAlgorithm.WORDPIECE,
    "distilbert": TokenizerAlgorithm.WORDPIECE,
    "electra": TokenizerAlgorithm.WORDPIECE,
    "mobilebert": TokenizerAlgorithm.WORDPIECE,
    "big_bird": TokenizerAlgorithm.WORDPIECE,
    "convbert": TokenizerAlgorithm.WORDPIECE,
    "funnel": TokenizerAlgorithm.WORDPIECE,
    "layoutlm": TokenizerAlgorithm.WORDPIECE,
    "lxmert": TokenizerAlgorithm.WORDPIECE,
    "deberta-v2": TokenizerAlgorithm.UNIGRAM,
    "xlm-roberta": TokenizerAlgorithm.UNIGRAM,
    "albert": TokenizerAlgorithm.UNIGRAM,
    "t5": TokenizerAlgorithm.SENTENCEPIECE,
    "mt5": TokenizerAlgorithm.SENTENCEPIECE,
    "longt5": TokenizerAlgorithm.SENTENCEPIECE,
    "umt5": TokenizerAlgorithm.SENTENCEPIECE,
    "byt5": TokenizerAlgorithm.SENTENCEPIECE,
}

# Model family display names
MODEL_FAMILY_MAP: dict[str, str] = {
    "roberta": "RoBERTa",
    "gpt2": "GPT-2",
    "gpt_neo": "GPT-Neo",
    "gpt_neox": "GPT-NeoX",
    "gptj": "GPT-J",
    "llama": "LLaMA",
    "mistral": "Mistral",
    "mixtral": "Mixtral",
    "falcon": "Falcon",
    "qwen3": "Qwen",
    "qwen2": "Qwen2",
    "qwen2_5": "Qwen2.5",
    "gemma": "Gemma",
    "gemma2": "Gemma 2",
    "phi3": "Phi-3",
    "phi4": "Phi-4",
    "cohere": "Cohere",
    "command-r": "Command R",
    "bloom": "BLOOM",
    "bigscience": "BigScience",
    "olmo": "OLMo",
    "olmoe": "OLMoE",
    "smollm": "SmolLM",
    "jais": "Jais",
    "nemotron": "Nemotron",
    "aya": "Aya",
    "bert": "BERT",
    "distilbert": "DistilBERT",
    "electra": "ELECTRA",
    "mobilebert": "MobileBERT",
    "deberta": "DeBERTa",
    "deberta-v2": "DeBERTa-v2",
    "big_bird": "BigBird",
    "convbert": "ConvBERT",
    "funnel": "Funnel Transformer",
    "layoutlm": "LayoutLM",
    "lxmert": "LXMERT",
    "xlm-roberta": "XLM-RoBERTa",
    "albert": "ALBERT",
    "t5": "T5",
    "mt5": "mT5",
    "longt5": "LongT5",
    "umt5": "UMT5",
    "byt5": "ByT5",
}

VULNERABLE_ALGORITHMS: set[TokenizerAlgorithm] = {
    TokenizerAlgorithm.BPE,
    TokenizerAlgorithm.WORDPIECE,
}

# Mapping from runtime ``tokenizers`` Rust model type name → algorithm
# (e.g., type(tokenizer._tokenizer.model).__name__ returns "BPE")
RUNTIME_MODEL_TYPE_MAP: dict[str, TokenizerAlgorithm] = {
    "BPE": TokenizerAlgorithm.BPE,
    "WordPiece": TokenizerAlgorithm.WORDPIECE,
    "Unigram": TokenizerAlgorithm.UNIGRAM,
    "WordLevel": TokenizerAlgorithm.WORDPIECE,  # Rare; treat as WordPiece-like
}



def detect_tokenizer_from_json(tokenizer_json: dict[str, Any]) -> TokenizerAlgorithm | None:
    """Detect tokenizer algorithm from tokenizer.json content.

    The tokenizer.json from the `tokenizers` library contains a 'type' key
    at the top level (e.g., 'BPE', 'WordPiece', 'Unigram').
    """
    model = tokenizer_json.get("model", {})
    tokenizer_type = model.get("type")

    if tokenizer_type is None:
        # Some older formats have type at top level
        tokenizer_type = tokenizer_json.get("type")

    if tokenizer_type is None:
        return None

    type_upper = str(tokenizer_type).upper()
    mapping: dict[str, TokenizerAlgorithm] = {
        "BPE": TokenizerAlgorithm.BPE,
        "WORDPIECE": TokenizerAlgorithm.WORDPIECE,
        "UNIGRAM": TokenizerAlgorithm.UNIGRAM,
        "SENTENCEPIECE": TokenizerAlgorithm.SENTENCEPIECE,
    }
    return mapping.get(type_upper)


def detect_tokenizer_from_config(config: dict[str, Any]) -> TokenizerAlgorithm | None:
    """Detect tokenizer algorithm from tokenizer_config.json content."""
    tokenizer_class = config.get("tokenizer_class", "")
    if tokenizer_class:
        result = TOKENIZER_CLASS_MAP.get(tokenizer_class)
        if result is not None:
            return result

    # Try to infer from model_type if present
    model_type = config.get("model_type", "")
    if model_type:
        return MODEL_TYPE_MAP.get(model_type)

    return None


def get_model_family(model_type: str) -> str:
    """Return human-readable model family name."""
    return MODEL_FAMILY_MAP.get(model_type, model_type.capitalize())


def is_vulnerable(algorithm: TokenizerAlgorithm) -> bool:
    """Return True if tokenizer algorithm is vulnerable to TokenBreak."""
    return algorithm in VULNERABLE_ALGORITHMS


def get_recommendation(algorithm: TokenizerAlgorithm) -> str:
    """Return remediation recommendation based on algorithm."""
    if algorithm == TokenizerAlgorithm.UNIGRAM:
        return (
            "Risk Assessment: SAFE (Cleared for Deployment). "
            "Model utilizes Unigram tokenization, which is structurally immune to TokenBreak character-level perturbation attacks. "
            "No further remediation is required for AI supply chain deployment."
        )
    if algorithm in (TokenizerAlgorithm.BPE, TokenizerAlgorithm.WORDPIECE):
        return (
            "Risk Assessment: CRITICAL VULNERABILITY DETECTED. "
            "The implemented BPE/WordPiece tokenization scheme exposes this model to known TokenBreak adversarial evasion attacks. "
            "MITIGATION ACTION: (1) Implement a Unigram-based token pre-processor to sanitize inputs (Pre-Mapping Defense), "
            "or (2) Migrate the system architecture to resilient alternatives (e.g., DeBERTa-v3 or XLM-RoBERTa) prior to production release."
        )
    if algorithm == TokenizerAlgorithm.SENTENCEPIECE:
        return (
            "Risk Assessment: CONDITIONAL EXPOSURE. "
            "SentencePiece encapsulation detected. "
            "ACTION REQUIRED: Manually audit the upstream `tokenizer.json` configuration to guarantee reliance on Unigram logic. "
            "Confirmed Unigram implementations inherently mitigate TokenBreak exploitation risks."
        )
    return (
        "Risk Assessment: UNVERIFIED EXPOSURE. "
        "Automated telemetry could not establish a definitive tokenization architecture fingerprint. "
        "ACTION REQUIRED: Mandate a manual architectural audit of `tokenizer.json` and model documentation "
        "to certify compliance with AI supply chain security standards."
    )


# ────────────────── Runtime Object Inspection ──────────────────

def detect_from_runtime_tokenizer(tokenizer: Any) -> tuple[TokenizerAlgorithm | None, str]:
    """Inspect a loaded :class:`transformers.PreTrainedTokenizer` object at runtime.

    Returns ``(algorithm, source_description)`` or ``(None, "")`` if inference fails.
    """
    if tokenizer is None:
        return None, ""

    # Signal A: Fast (Rust) tokenizer backend
    fast_backend: Optional[Any] = getattr(tokenizer, "_tokenizer", None)
    if fast_backend is not None:
        model: Any = getattr(fast_backend, "model", None)
        if model is not None:
            type_name: str = type(model).__name__
            algo = RUNTIME_MODEL_TYPE_MAP.get(type_name)
            if algo is not None:
                return algo, f"tokenizer._tokenizer.model type={type_name}"

    # Signal B: Class name
    cls_name = tokenizer.__class__.__name__
    algo = TOKENIZER_CLASS_MAP.get(cls_name)
    if algo is not None:
        return algo, f"tokenizer class name={cls_name}"

    return None, ""


# - Signature keywords used for source-code fingerprinting -
SOURCE_FINGERPRINTS: dict[tuple[str, ...], TokenizerAlgorithm] = {
    ("bpe", "byte-encoder", "merges"): TokenizerAlgorithm.BPE,
    ("wordpiece", "vocab", "##"): TokenizerAlgorithm.WORDPIECE,
    ("sentencepiece", "sp_model", "sp_processor"): TokenizerAlgorithm.SENTENCEPIECE,
    ("spm", "sp_processor"): TokenizerAlgorithm.SENTENCEPIECE,
}


def _source_score(source: str) -> tuple[TokenizerAlgorithm | None, float, str]:
    """Fingerprint a tokenizer source-code string."""
    lowered = source.lower()
    best_algo: Optional[TokenizerAlgorithm] = None
    best_score = 0.0
    best_reason = ""

    for keywords, algo in SOURCE_FINGERPRINTS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        score = hits / len(keywords)
        if score > best_score:
            best_score = score
            best_algo = algo
            best_reason = f"matched {hits}/{len(keywords)} keywords {keywords}"

    return best_algo, best_score, best_reason


def detect_from_source_code(tokenizer: Any) -> tuple[TokenizerAlgorithm | None, str]:
    """Try :func:`inspect.getsource` on the tokenizer class.

    Returns ``(algorithm, reason)`` or ``(None, "")`` if source is unavailable
    (built-in, compiled extension, or missing file).
    """
    try:
        import inspect
        source = inspect.getsource(tokenizer.__class__)
    except Exception:
        return None, ""

    algo, score, reason = _source_score(source)
    if algo is not None and score >= 0.5:
        return algo, f"source-code fingerprint: {reason}"
    return None, ""


def detect_from_remote_source(
    model_path: str | Path, *, trust_remote_code: bool = False,
) -> tuple[TokenizerAlgorithm | None, str]:
    """Download and inspect the tokenizer Python module from HF Hub.

    Useful when ``trust_remote_code=True`` and the model ships a custom
    ``tokenization_*.py`` file that overrides default behaviour.
    """
    import re
    from pathlib import Path

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None, ""

    local = Path(model_path)
    repo_id: str = ""
    if not local.is_dir():
        repo_id = str(model_path)
    else:
        # Try to read model_info.json or infer from directory structure
        model_info = local / "model_info.json"
        if model_info.exists():
            info = json.loads(model_info.read_text())
            repo_id = info.get("id", "")

    if not repo_id:
        return None, ""

    # Try to locate the tokenizer module
    tokenizer_file_candidates = ["tokenization_.py", "tokenizer.py", "tokenization_utils.py"]
    for candidate in tokenizer_file_candidates:
        try:
            tok_file = hf_hub_download(repo_id, candidate, local_files_only=False)
        except Exception:
            continue
        if tok_file and Path(tok_file).exists():
            source = Path(tok_file).read_text(encoding="utf-8")
            algo, score, reason = _source_score(source)
            if algo is not None and score >= 0.5:
                return algo, f"remote source file: {reason}"

    return None, ""
