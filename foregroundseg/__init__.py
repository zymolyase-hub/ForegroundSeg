"""
ForegroundSeg
==============

AI-powered foreground/background segmentation using PyTorch.
"""

from .version import __version__
from .predictor import ForegroundPredictor

__all__ = [
    "ForegroundPredictor",
    "__version__",
]
