#!/usr/bin/env python3
"""
Simple test script for MLX-vLLM server.
"""

import asyncio
import json
import aiohttp
import time


async def test_server():
    """Test the MLX-vLLM server."""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing MLX-vLLM server...")
        
        # Test health check
        print("\n1. Health check...")
        async with session.get(f"{base_url}/health") as resp:
            health = await resp.json()
            print(f"   Status: {health}")
        
        # Test model loading
        print("\n2. Loading model...")
        model_data = {
            "model_name": "mlx-community/Qwen2.5-3B-Instruct-4bit"
        }
        async with session.post(f"{base_url}/v1/models/load", json=model_data) as resp:
            result = await resp.json()
            print(f"   Result: {result}")
        
        # Wait a moment for model to load
        await asyncio.sleep(2)
        
        # Test list models
        print("\n3. List models...")
        async with session.get(f"{base_url}/v1/models") as resp:
            models = await resp.json()
            print(f"   Models: {models}")
        
        # Test completion
        print("\n4. Test completion...")
        completion_data = {
            "model": "mlx-model",
            "prompt": "Hello, how are you?",
            "max_tokens": 20,
            "temperature": 0.7
        }
        
        start_time = time.time()
        async with session.post(f"{base_url}/v1/completions", json=completion_data) as resp:
            completion = await resp.json()
            end_time = time.time()
            
            print(f"   Response: {completion}")
            print(f"   Time taken: {end_time - start_time:.2f}s")
        
        # Test chat completion
        print("\n5. Test chat completion...")
        chat_data = {
            "model": "mlx-model",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 30,
            "temperature": 0.7
        }
        
        start_time = time.time()
        async with session.post(f"{base_url}/v1/chat/completions", json=chat_data) as resp:
            chat_completion = await resp.json()
            end_time = time.time()
            
            print(f"   Response: {chat_completion}")
            print(f"   Time taken: {end_time - start_time:.2f}s")
        
        # Test streaming
        print("\n6. Test streaming...")
        stream_data = {
            "model": "mlx-model",
            "prompt": "Write a short poem about",
            "max_tokens": 50,
            "stream": True
        }
        
        print("   Streaming response: ", end="")
        async with session.post(f"{base_url}/v1/completions", json=stream_data) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'text' in delta:
                                print(delta['text'], end='', flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print("\n\n✅ All tests completed!")


if __name__ == "__main__":
    print("🚀 MLX-vLLM Server Test")
    print("Make sure the server is running with:")
    print("mlx-vllm serve --model mlx-community/Qwen2.5-3B-Instruct-4bit --host 0.0.0.0 --port 8000")
    print()
    
    asyncio.run(test_server())
