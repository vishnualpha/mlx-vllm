#!/usr/bin/env python3
"""
MLX-vLLM: High-performance LLM serving for Apple Silicon
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx-vllm",
    version="0.1.0",
    author="MLX-vLLM Team",
    author_email="mlx-vllm@example.com",
    description="High-performance LLM serving for Apple Silicon with MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlx-vllm/mlx-vllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "mlx>=0.15.0",
        "mlx-lm>=0.15.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.35.0",
        "huggingface_hub>=0.17.0",
        "tokenizers>=0.14.0",
        "psutil>=5.9.0",
        "aiofiles>=23.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlx-vllm=mlx_vllm.cli:main",
        ],
    },
)
