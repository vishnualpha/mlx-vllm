#!/usr/bin/env python3
"""
Command-line interface for MLX-vLLM.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .api.server import run_server
from .core.types import EngineConfig
from .models.loader import get_recommended_models


def setup_logging(level: str = "info"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def check_system():
    """Check system compatibility."""
    print("🔍 Checking system compatibility...")
    
    # Check MLX availability
    try:
        import mlx.core as mx
        print("✅ MLX is available")
    except ImportError:
        print("❌ MLX not found. Please install: pip install mlx mlx-lm")
        return False
    
    # Check platform
    if sys.platform != "darwin":
        print("⚠️  Warning: Not running on macOS. Performance may be limited.")
    else:
        print("✅ Running on macOS")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def list_models():
    """List recommended models."""
    print("📋 Recommended MLX Models:")
    print("=" * 50)
    
    models = get_recommended_models()
    for category, model_list in models.items():
        print(f"\n{category.upper()}:")
        for model in model_list:
            print(f"  • {model}")


async def serve_command(args):
    """Run the server."""
    if not check_system():
        sys.exit(1)
    
    print(f"🚀 Starting MLX-vLLM server on {args.host}:{args.port}")
    if args.model:
        print(f"📥 Loading model: {args.model}")
    
    try:
        await run_server(
            host=args.host,
            port=args.port,
            model_name=args.model,
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mlx-vllm",
        description="MLX-vLLM: High-performance LLM serving for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with automatic model selection
  mlx-vllm serve --model mlx-community/Qwen2.5-7B-Instruct-4bit
  
  # Start server on all interfaces
  mlx-vllm serve --host 0.0.0.0 --port 8000
  
  # List recommended models
  mlx-vllm list-models
  
  # Check system compatibility
  mlx-vllm check-system
        """
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MLX-vLLM server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--model",
        help="Model to load on startup (e.g., mlx-community/Qwen2.5-7B-Instruct-4bit)"
    )
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List recommended models")
    
    # Check system command
    check_parser = subparsers.add_parser("check-system", help="Check system compatibility")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle commands
    if args.command == "serve":
        asyncio.run(serve_command(args))
    elif args.command == "list-models":
        list_models()
    elif args.command == "check-system":
        if check_system():
            print("\n✅ System is compatible with MLX-vLLM")
        else:
            print("\n❌ System compatibility issues found")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
