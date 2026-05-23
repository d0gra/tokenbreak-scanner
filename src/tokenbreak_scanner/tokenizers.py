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
            "No action needed. This model uses Unigram tokenization, which is "
            "structurally resistant to TokenBreak character-level perturbation attacks. "
            "Safe to fine-tune and deploy."
        )
    if algorithm in (TokenizerAlgorithm.BPE, TokenizerAlgorithm.WORDPIECE):
        return (
            "This model uses BPE/WordPiece tokenization, which is vulnerable to "
            "TokenBreak adversarial evasion attacks. Before deploying in a "
            "security-sensitive context, consider: "
            "(1) Adding a Unigram-based input pre-processor to neutralize "
            "character-level perturbations, or "
            "(2) Evaluating resistant alternatives like DeBERTa-v3 or "
            "XLM-RoBERTa that use Unigram tokenization natively."
        )
    if algorithm == TokenizerAlgorithm.SENTENCEPIECE:
        return (
            "SentencePiece detected. SentencePiece can wrap either Unigram (resistant) or "
            "BPE (vulnerable) under the hood. Verify that the underlying algorithm is Unigram "
            "by checking tokenizer.json or model documentation before deploying."
        )
    return (
        "Could not determine the tokenization algorithm automatically. "
        "Manually inspect tokenizer.json and model documentation to confirm "
        "whether BPE or WordPiece is used before deploying."
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


# ────────────────── Diagnostic Tokenization Sensitivity Probe ──────────────────
#
# This is NOT a ground-truth vulnerability test.  It is a *diagnostic probe*
# that measures how the tokenizer reacts to stealthy character perturbations
# (invisible / zero-width Unicode characters).  Its purpose is to highlight
# *unexpected sensitivity* or to help disambiguate unknown tokenizers when no
# structural metadata is available.
#
# Results are labelled explicitly as "structural" (algorithm name) vs.
# "behavioral" (this diagnostic) so users never see contradictory evidence.

_TOKENBREAK_TEST_WORDS: list[str] = [
    "password",
    "reveal",
    "system",
    "ignore",
    "instructions",
    "secret",
    "bypass",
    "admin",
    "override",
    "confidential",
]

# Invisible / stealthy perturbation characters used by real-world TokenBreak
# attacks (zero-width spaces, soft hyphens, zero-width joiners, etc.).  These
# are more realistic adversarial inputs than visible ASCII prepend chars.
_STEALTH_PERTURBATIONS: list[str] = [
    "\u200b",   # zero-width space (U+200B)
    "\u200c",   # zero-width non-joiner (U+200C)
    "\u200d",   # zero-width joiner (U+200D)
    "\u2060",   # word joiner (U+2060)
    "\ufeff",   # zero-width no-break space (U+FEFF)
    "\u00ad",   # soft hyphen (U+00AD)
    "\u180e",   # Mongolian vowel separator (U+180E)
    "\u200e",   # left-to-right mark (U+200E)
    "\u200f",   # right-to-left mark (U+200F)
]

# Visible prepend characters for complementarity when we have NO structural
# signals and need to compare an unknown tokenizer against known baselines.
_VISIBLE_PREPEND_CHARS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def detect_from_tokenization_behavior(
    tokenizer: Any,
    *,
    use_invisible: bool = True,
) -> dict[str, Any]:
    """Diagnostic probe of tokenization sensitivity to character perturbations.

    Parameters
    ----------
    tokenizer : Any
        A tokenizer object with an ``encode`` method (transformers, tokenizers, etc.).
    use_invisible : bool, optional
        If True (default), use invisible / zero-width Unicode chars — realistic
        adversarial perturbations.  If False, fall back to visible ASCII chars.

    Returns
    -------
    dict[str, Any]
        A structured diagnostic result with keys:
        - ``shifted``: int — how many perturbation / word pairs caused a shift.
        - ``total``: int — total perturbation / word pairs tested.
        - ``fragility``: float — shifted / total (0.0 = fully stable).
        - ``inconsistent_with_unigram``: bool — True if fragility > 0 and the
          tokenizer context suggests Unigram (indicates further review).
        - ``detail``: str — human-readable summary.
    """
    if tokenizer is None:
        return {
            "shifted": 0,
            "total": 0,
            "fragility": 0.0,
            "inconsistent_with_unigram": False,
            "detail": "No tokenizer available for diagnostic probe",
        }

    perturbations = _STEALTH_PERTURBATIONS if use_invisible else list(_VISIBLE_PREPEND_CHARS)
    total_tests = 0
    shifted_tests = 0
    fragile_words: list[str] = []

    try:
        for word in _TOKENBREAK_TEST_WORDS:
            base_ids = tokenizer.encode(word, add_special_tokens=False)
            word_shifted = False

            for ch in perturbations:
                perturbed = ch + word
                perturbed_ids = tokenizer.encode(perturbed, add_special_tokens=False)

                # Skip if tokenizer collapsed to same length (e.g. NFKC norm stripped char)
                if len(perturbed_ids) <= len(base_ids):
                    total_tests += 1
                    continue

                # Compare the tail — did the *word portion* of the tokenization change?
                tail = perturbed_ids[-len(base_ids):]
                total_tests += 1
                if tail != base_ids:
                    shifted_tests += 1
                    word_shifted = True

            if word_shifted:
                fragile_words.append(word)

    except Exception as exc:
        logger.warning("Diagnostic tokenization probe failed: %s", exc)
        return {
            "shifted": 0,
            "total": 0,
            "fragility": 0.0,
            "inconsistent_with_unigram": False,
            "detail": f"Probe error: {exc}",
        }

    if total_tests == 0:
        return {
            "shifted": 0,
            "total": 0,
            "fragility": 0.0,
            "inconsistent_with_unigram": False,
            "detail": "No tests could be performed",
        }

    fragility = shifted_tests / total_tests if total_tests > 0 else 0.0
    detail = (
        f"Diagnostic probe ({len(perturbations)} perturbations x {len(_TOKENBREAK_TEST_WORDS)} words): "
        f"{shifted_tests}/{total_tests} tokenization shifts (fragility={fragility:.2f}). "
        + (f"Sensitive words: {', '.join(fragile_words[:5])}" + ("..." if len(fragile_words) > 5 else ""))
        if fragile_words
        else "No shifts detected — tokenization is stable under stealthy character perturbation."
    )

    return {
        "shifted": shifted_tests,
        "total": total_tests,
        "fragility": fragility,
        "inconsistent_with_unigram": False,  # Set by caller after structural detection
        "detail": detail,
    }


def _is_behavior_consistent_with_algorithm(
    structural_algorithm: TokenizerAlgorithm,
    fragility: float,
) -> bool:
    """Check if the diagnostic fragility score is consistent with the structurally-detected algorithm.

    Unigram tokenizers are *context-aware*, so some non-zero fragility (especially
    with visible prepends) is expected.  We flag inconsistency only when the amount
    of fragility is unexpectedly high for a supposedly resistant tokenizer.
    """
    if structural_algorithm == TokenizerAlgorithm.BPE:
        # BPE is expected to be fragile; any fragility is "consistent"
        return True
    if structural_algorithm == TokenizerAlgorithm.WORDPIECE:
        return True
    if structural_algorithm == TokenizerAlgorithm.UNIGRAM:
        # Unigram should resist invisible-char perturbations.
        # Some visible-char fragility is expected.  Threshold is heuristic.
        return fragility < 0.40
    if structural_algorithm == TokenizerAlgorithm.SENTENCEPIECE:
        return True  # Ambiguous by design; can't judge without resolution
    return False  # UNKNOWN


# ────────────────── SentencePiece Ambiguity Resolution ──────────────────

def resolve_sentencepiece_algorithm(
    model_path: str | Path,
) -> TokenizerAlgorithm | None:
    """Resolve whether a SentencePiece model uses Unigram or BPE underneath.

    SentencePiece can wrap either algorithm.  This loads the .model file
    via the ``sentencepiece`` library and inspects the trainer type to
    disambiguate.

    Returns
    -------
    TokenizerAlgorithm.UNIGRAM, TokenizerAlgorithm.BPE, or None if unresolvable.
    """
    from pathlib import Path as _Path

    model_dir = _Path(model_path)
    sp_model_path = None

    # Look for sentencepiece model file under common names
    for candidate in ("spiece.model", "tokenizer.model", "sentencepiece.bpe.model"):
        candidate_path = model_dir / candidate
        if candidate_path.exists():
            sp_model_path = candidate_path
            break

    if sp_model_path is None:
        # Also check one level up (some repos have it in parent dir)
        for candidate in ("spiece.model", "tokenizer.model", "sentencepiece.bpe.model"):
            candidate_path = model_dir.parent / candidate
            if candidate_path.exists():
                sp_model_path = candidate_path
                break

    if sp_model_path is None:
        return None

    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(sp_model_path))

        # Inspect the model proto to determine trainer algorithm
        # sentencepiece stores this in the model proto's trainer_spec
        model_proto = sp.SerializedModelProto()
        if b"unigram" in model_proto.lower():
            return TokenizerAlgorithm.UNIGRAM
        if b"bpe" in model_proto.lower():
            return TokenizerAlgorithm.BPE

        # Fallback: check vocab size patterns
        # BPE vocab typically has more single-char tokens
        vocab_size = sp.GetPieceSize()
        single_char_count = sum(
            1 for i in range(vocab_size)
            if len(sp.IdToPiece(i)) == 1
        )
        if single_char_count > vocab_size * 0.3:
            # High proportion of single-char tokens → likely BPE
            return TokenizerAlgorithm.BPE
        else:
            return TokenizerAlgorithm.UNIGRAM

    except Exception as exc:
        logger.debug("Could not resolve SentencePiece ambiguity: %s", exc)
        return None


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
