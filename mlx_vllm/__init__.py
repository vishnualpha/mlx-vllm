"""
MLX-vLLM: High-performance LLM serving for Apple Silicon

A production-ready implementation of vLLM-style serving optimized for Apple Silicon
using the MLX framework. Provides OpenAI-compatible APIs with continuous batching,
speculative decoding, and Apple Silicon optimizations.
"""

__version__ = "0.1.0"
__author__ = "MLX-vLLM Team"

from .core.engine import MLXEngine
from .api.server import MLXServer
from .models.loader import ModelLoader

__all__ = [
    "MLXEngine",
    "MLXServer", 
    "ModelLoader",
]
