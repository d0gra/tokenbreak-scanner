"""TokenBreak Model File Scanner.

Audit NLP model artifacts for TokenBreak vulnerabilities by inspecting
tokenizer configurations and model architectures.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tokenbreak-scanner")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
