"""DocumentLanguageModel: a text file with a .dlm extension becomes a local, trainable LLM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("document-language-model")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
