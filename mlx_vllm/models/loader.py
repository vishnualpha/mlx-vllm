"""
Model loading utilities for MLX-vLLM.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.models.base import BaseModelArgs
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from huggingface_hub import snapshot_download
import json

logger = logging.getLogger(__name__)


class MLXModel:
    """Wrapper for MLX models with vLLM-compatible interface."""
    
    def __init__(self, model, tokenizer, model_args: Optional[BaseModelArgs] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.vocab_size = getattr(model_args, 'vocab_size', 32000) if model_args else 32000
        
    def forward(self, input_ids: mx.array, cache=None, **kwargs) -> mx.array:
        """Forward pass through the model."""
        if hasattr(self.model, 'forward'):
            return self.model.forward(input_ids, cache=cache, **kwargs)
        else:
            return self.model(input_ids, cache=cache, **kwargs)
    
    def __call__(self, input_ids: mx.array, **kwargs) -> mx.array:
        """Make model callable."""
        return self.forward(input_ids, **kwargs)


class ModelLoader:
    """Loads and manages MLX models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mlx_vllm")
        self.loaded_models: Dict[str, MLXModel] = {}
        
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Please install MLX: pip install mlx mlx-lm"
            )
    
    def load_model(self, model_name: str, **kwargs) -> MLXModel:
        """
        Load a model from Hugging Face or local path.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional arguments for model loading
            
        Returns:
            MLXModel instance
        """
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return self.loaded_models[model_name]
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Use mlx-lm to load the model
            model, tokenizer = load(model_name, **kwargs)
            
            # Get model args if available
            model_args = None
            if hasattr(model, 'args'):
                model_args = model.args
            
            mlx_model = MLXModel(model, tokenizer, model_args)
            self.loaded_models[model_name] = mlx_model
            
            logger.info(f"Successfully loaded model: {model_name}")
            return mlx_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        try:
            # Download model config
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                allow_patterns=["config.json", "*.json"]
            )
            
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                return {
                    "model_name": model_name,
                    "vocab_size": config.get("vocab_size", 32000),
                    "hidden_size": config.get("hidden_size", 4096),
                    "num_layers": config.get("num_hidden_layers", 32),
                    "num_attention_heads": config.get("num_attention_heads", 32),
                    "max_position_embeddings": config.get("max_position_embeddings", 2048),
                    "model_type": config.get("model_type", "unknown"),
                }
            else:
                return {"model_name": model_name, "error": "Config not found"}
                
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {"model_name": model_name, "error": str(e)}
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")
        else:
            logger.warning(f"Model {model_name} not found in loaded models")
    
    def list_loaded_models(self) -> list[str]:
        """List currently loaded models."""
        return list(self.loaded_models.keys())
    
    def clear_cache(self):
        """Clear all loaded models."""
        self.loaded_models.clear()
        logger.info("Cleared all loaded models")


def get_recommended_models() -> Dict[str, list[str]]:
    """Get recommended MLX-community models by category."""
    return {
        "7B": [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        ],
        "14B": [
            "mlx-community/Qwen2.5-14B-Instruct-4bit",
            "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
        ],
        "32B": [
            "mlx-community/Qwen2.5-32B-Instruct-4bit",
            "mlx-community/Meta-Llama-3.1-70B-Instruct-8bit",
        ],
        "coding": [
            "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
            "mlx-community/CodeLlama-7b-Instruct-hf-4bit",
        ],
        "fast": [
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
        ]
    }
