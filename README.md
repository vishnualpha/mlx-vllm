# MLX-vLLM

**High-performance LLM serving for Apple Silicon with MLX**

MLX-vLLM is a production-ready implementation of vLLM-style serving optimized for Apple Silicon using the MLX framework. It provides OpenAI-compatible APIs with continuous batching and Apple Silicon optimizations.

## 🚀 Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **Apple Silicon Optimized** - Native MLX integration for M1/M2/M3/M4 chips
- **Continuous Batching** - Efficient request scheduling and batching
- **Streaming Support** - Real-time token streaming
- **Easy Model Loading** - Automatic MLX-community model support
- **Production Ready** - Comprehensive error handling and monitoring

## 📦 Installation

### Requirements

- **macOS** (Apple Silicon recommended)
- **Python 3.9+**
- **MLX framework**

### Install from PyPI (Coming Soon)

```bash
pip install mlx-vllm
```

### Install from Source

```bash
git clone https://github.com/mlx-vllm/mlx-vllm.git
cd mlx-vllm
pip install -e .
```

### Install MLX Dependencies

```bash
pip install mlx mlx-lm
```

## 🎯 Quick Start

### 1. Check System Compatibility

```bash
mlx-vllm check-system
```

### 2. List Recommended Models

```bash
mlx-vllm list-models
```

### 3. Start the Server

```bash
# Start with a 7B model (recommended for most systems)
mlx-vllm serve --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Start on all interfaces
mlx-vllm serve --host 0.0.0.0 --port 8000 --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

### 4. Test the API

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'

# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-model",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

## 🔧 API Reference

### Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List loaded models
- `GET /v1/models/recommended` - Get recommended models
- `POST /v1/models/load` - Load a model
- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)
- `POST /v1/completions` - Text completions (OpenAI compatible)
- `GET /v1/stats` - Server statistics

### Load a Model

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit"
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-model",
    "messages": [{"role": "user", "content": "Write a short story"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## 🤖 Supported Models

MLX-vLLM works with any model from the [mlx-community](https://huggingface.co/mlx-community) organization on Hugging Face.

### Recommended Models by Size

**7B Models (Most Systems):**
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`

**14B Models (32GB+ RAM):**
- `mlx-community/Qwen2.5-14B-Instruct-4bit`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`

**32B+ Models (64GB+ RAM):**
- `mlx-community/Qwen2.5-32B-Instruct-4bit`
- `mlx-community/Meta-Llama-3.1-70B-Instruct-8bit`

**Coding Models:**
- `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit`
- `mlx-community/CodeLlama-7b-Instruct-hf-4bit`

## 📊 Performance

### Expected Performance on Apple Silicon

| Model Size | M2 Max | M3 Max | M4 Max | Memory Usage |
|------------|--------|--------|--------|--------------|
| 7B-4bit    | 45 t/s | 65 t/s | 85 t/s | ~6 GB        |
| 14B-4bit   | 25 t/s | 35 t/s | 45 t/s | ~12 GB       |
| 32B-4bit   | 12 t/s | 18 t/s | 25 t/s | ~24 GB       |

*t/s = tokens per second for single request*

### Optimization Tips

1. **Use 4-bit quantized models** for best memory efficiency
2. **Close other applications** to free up memory
3. **Use smaller models** for better latency
4. **Enable Metal Performance Shaders** (automatic)

## 🛠 Development

### Project Structure

```
mlx_vllm/
├── core/           # Core engine and scheduling
│   ├── engine.py   # Main MLX engine
│   └── types.py    # Data structures
├── models/         # Model loading and management
│   └── loader.py   # MLX model loader
├── api/            # FastAPI server
│   └── server.py   # OpenAI-compatible API
└── cli.py          # Command-line interface
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=mlx_vllm
```

### Code Formatting

```bash
# Format code
black mlx_vllm/
isort mlx_vllm/

# Type checking
mypy mlx_vllm/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mlx-vllm/mlx-vllm.git
cd mlx-vllm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Apple MLX Team** - For the excellent MLX framework
- **vLLM Team** - For the original vLLM architecture and inspiration
- **Hugging Face** - For the transformers library and model hub
- **MLX-Community** - For providing optimized MLX models

## 📞 Support

- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - General questions and community support
- **Documentation** - Comprehensive guides and API reference

## 🔗 Links

- **GitHub**: https://github.com/mlx-vllm/mlx-vllm
- **Documentation**: https://mlx-vllm.readthedocs.io
- **PyPI**: https://pypi.org/project/mlx-vllm
- **MLX Framework**: https://ml-explore.github.io/mlx/
- **MLX Models**: https://huggingface.co/mlx-community

---

**Made with ❤️ for the Apple Silicon community**
