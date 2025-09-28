"""
FastAPI server with OpenAI-compatible endpoints.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.engine import MLXEngine
from ..core.types import EngineConfig, SamplingParams
from ..models.loader import get_recommended_models

logger = logging.getLogger(__name__)


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mlx-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "mlx-model"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ModelLoadRequest(BaseModel):
    model_name: str
    force_reload: Optional[bool] = False


class MLXServer:
    """MLX-vLLM FastAPI server."""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig(model_name="")
        self.engine: Optional[MLXEngine] = None
        self.app = FastAPI(
            title="MLX-vLLM Server",
            description="OpenAI-compatible API server for MLX models",
            version="0.1.0"
        )
        self._setup_routes()
        
        # Background task for processing requests
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup():
            """Start background processing."""
            self._background_task = asyncio.create_task(self._background_loop())
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Shutdown background processing."""
            self._shutdown_event.set()
            if self._background_task:
                await self._background_task
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model_loaded": self.engine is not None}
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            if self.engine is None:
                return {"object": "list", "data": []}
            
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mlx-vllm",
                    }
                ]
            }
        
        @self.app.get("/v1/models/recommended")
        async def get_recommended_models_endpoint():
            """Get recommended models."""
            return get_recommended_models()
        
        @self.app.post("/v1/models/load")
        async def load_model(request: ModelLoadRequest):
            """Load a model."""
            try:
                # Create engine if not exists
                if self.engine is None or request.force_reload:
                    config = EngineConfig(model_name=request.model_name)
                    self.engine = MLXEngine(config)
                    self.config = config
                
                await self.engine.load_model(request.model_name)
                
                return {
                    "status": "success",
                    "model_name": request.model_name,
                    "message": "Model loaded successfully"
                }
            except Exception as e:
                logger.error(f"Failed to load model {request.model_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Chat completions endpoint."""
            if self.engine is None:
                raise HTTPException(status_code=400, detail="No model loaded")
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(request.messages)
            
            # Convert to completion request
            completion_request = CompletionRequest(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                stop=request.stop,
            )
            
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(completion_request, is_chat=True),
                    media_type="text/plain"
                )
            else:
                return await self._complete(completion_request, is_chat=True)
        
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """Completions endpoint."""
            if self.engine is None:
                raise HTTPException(status_code=400, detail="No model loaded")
            
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(request),
                    media_type="text/plain"
                )
            else:
                return await self._complete(request)
        
        @self.app.get("/v1/stats")
        async def get_stats():
            """Get server statistics."""
            if self.engine is None:
                return {"error": "No model loaded"}
            
            stats = self.engine.get_stats()
            return {
                "engine_stats": stats,
                "model_config": self.engine.get_model_config(),
            }
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a prompt."""
        prompt_parts = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def _complete(self, request: CompletionRequest, is_chat: bool = False) -> Dict[str, Any]:
        """Handle non-streaming completion."""
        request_id = str(uuid.uuid4())
        
        # Handle multiple prompts
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        results = []
        for prompt in prompts:
            # Create sampling params
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens or 16,
                temperature=request.temperature or 1.0,
                top_p=request.top_p or 1.0,
                stop=self._parse_stop(request.stop),
            )
            
            # Add request to engine
            self.engine.add_request(request_id, prompt, sampling_params)
            
            # Wait for completion
            result = await self._wait_for_completion(request_id)
            results.append(result)
        
        # Format response
        choices = []
        for i, result in enumerate(results):
            choice = {
                "index": i,
                "finish_reason": "stop",
                "logprobs": None,
            }
            
            if is_chat:
                choice["message"] = {
                    "role": "assistant",
                    "content": result["text"]
                }
            else:
                choice["text"] = result["text"]
            
            choices.append(choice)
        
        return {
            "id": request_id,
            "object": "chat.completion" if is_chat else "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": sum(r["prompt_tokens"] for r in results),
                "completion_tokens": sum(r["completion_tokens"] for r in results),
                "total_tokens": sum(r["total_tokens"] for r in results),
            }
        }
    
    async def _stream_completion(self, request: CompletionRequest, is_chat: bool = False) -> AsyncGenerator[str, None]:
        """Handle streaming completion."""
        request_id = str(uuid.uuid4())
        
        # Handle only first prompt for streaming
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens or 16,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            stop=self._parse_stop(request.stop),
        )
        
        # Add request to engine
        self.engine.add_request(request_id, prompt, sampling_params)
        
        # Stream tokens
        async for token in self._stream_tokens(request_id):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk" if is_chat else "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token} if is_chat else {"text": token},
                    "finish_reason": None,
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk" if is_chat else "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _parse_stop(self, stop: Optional[Union[str, List[str]]]) -> List[str]:
        """Parse stop sequences."""
        if stop is None:
            return []
        elif isinstance(stop, str):
            return [stop]
        else:
            return stop
    
    async def _wait_for_completion(self, request_id: str) -> Dict[str, Any]:
        """Wait for a request to complete."""
        # This is a simplified implementation
        # In practice, you'd track requests and their outputs
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "text": "This is a placeholder response.",
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
        }
    
    async def _stream_tokens(self, request_id: str) -> AsyncGenerator[str, None]:
        """Stream tokens for a request."""
        # This is a simplified implementation
        tokens = ["This", " is", " a", " placeholder", " response", "."]
        
        for token in tokens:
            await asyncio.sleep(0.1)  # Simulate generation time
            yield token
    
    async def _background_loop(self):
        """Background loop for processing requests."""
        while not self._shutdown_event.is_set():
            try:
                if self.engine:
                    await self.engine.step()
                await asyncio.sleep(0.01)  # 10ms loop
            except Exception as e:
                logger.error(f"Error in background loop: {e}")
                await asyncio.sleep(0.1)


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    """Create FastAPI app."""
    server = MLXServer(config)
    return server.app


async def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: Optional[str] = None,
    **kwargs
):
    """Run the server."""
    config = EngineConfig(model_name=model_name or "")
    app = create_app(config)
    
    # Load model if specified
    if model_name:
        server = MLXServer(config)
        server.engine = MLXEngine(config)
        await server.engine.load_model(model_name)
    
    # Run server
    config_uvicorn = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        **kwargs
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()
