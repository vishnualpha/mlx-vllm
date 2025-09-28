"""
Core MLX engine for request processing and model execution.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from collections import deque

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from .types import (
    Sequence, SequenceGroup, SequenceStatus, SamplingParams,
    ModelInput, ModelOutput, SchedulerOutput, EngineConfig
)
from ..models.loader import ModelLoader, MLXModel

logger = logging.getLogger(__name__)


class SimpleScheduler:
    """Simple FIFO scheduler for MLX-vLLM."""
    
    def __init__(self, max_num_seqs: int = 256, max_num_batched_tokens: int = 2048):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        
        # Queues
        self.waiting: deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []
        
    def add_seq_group(self, seq_group: SequenceGroup):
        """Add a sequence group to the waiting queue."""
        self.waiting.append(seq_group)
        logger.debug(f"Added sequence group {seq_group.request_id} to waiting queue")
    
    def abort_seq_group(self, request_id: str):
        """Abort a sequence group."""
        # Remove from waiting
        self.waiting = deque([sg for sg in self.waiting if sg.request_id != request_id])
        
        # Mark running sequences as aborted
        for seq_group in self.running:
            if seq_group.request_id == request_id:
                for seq in seq_group.sequences:
                    seq.status = SequenceStatus.FINISHED_ABORTED
        
        logger.debug(f"Aborted sequence group {request_id}")
    
    def schedule(self) -> SchedulerOutput:
        """Schedule sequences for execution."""
        scheduled_seq_groups = []
        prompt_runs = []
        decode_runs = []
        
        # Move waiting sequences to running if there's capacity
        while (len(self.running) < self.max_num_seqs and 
               self.waiting and 
               self._can_allocate_tokens()):
            
            seq_group = self.waiting.popleft()
            
            # Mark sequences as running
            for seq in seq_group.sequences:
                seq.status = SequenceStatus.RUNNING
                if seq.metrics.first_scheduled_time is None:
                    seq.metrics.first_scheduled_time = time.time()
            
            self.running.append(seq_group)
            scheduled_seq_groups.append(seq_group)
            
            # Determine if this is prompt or decode
            if seq_group.sequences[0].get_output_len() == 0:
                prompt_runs.append(seq_group)
            else:
                decode_runs.append(seq_group)
        
        # Add existing running sequences to decode runs
        for seq_group in self.running:
            if seq_group not in scheduled_seq_groups:
                if seq_group.sequences[0].get_output_len() > 0:
                    decode_runs.append(seq_group)
                    scheduled_seq_groups.append(seq_group)
        
        # Remove finished sequences
        self.running = [sg for sg in self.running if not sg.is_finished()]
        
        return SchedulerOutput(
            scheduled_seq_groups=scheduled_seq_groups,
            prompt_runs=prompt_runs,
            decode_runs=decode_runs,
            preempted=[],
            swapped_in=[],
            swapped_out=[],
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
        )
    
    def _can_allocate_tokens(self) -> bool:
        """Check if we can allocate more tokens."""
        total_tokens = sum(
            seq.get_len() for seq_group in self.running 
            for seq in seq_group.sequences
        )
        return total_tokens < self.max_num_batched_tokens
    
    def get_num_unfinished_seq_groups(self) -> int:
        """Get number of unfinished sequence groups."""
        return len(self.waiting) + len(self.running) + len(self.swapped)


class MLXEngine:
    """Main MLX engine for processing requests."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_loader = ModelLoader()
        self.model: Optional[MLXModel] = None
        self.scheduler = SimpleScheduler(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens
        )
        
        # Statistics
        self.stats = {
            "num_requests": 0,
            "num_tokens_generated": 0,
            "total_generation_time": 0.0,
        }
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Please install MLX.")
    
    async def load_model(self, model_name: str):
        """Load a model."""
        logger.info(f"Loading model: {model_name}")
        self.model = self.model_loader.load_model(model_name)
        logger.info(f"Model loaded successfully: {model_name}")
    
    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> None:
        """Add a request to the engine."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Tokenize prompt if needed
        if prompt_token_ids is None:
            prompt_token_ids = self.model.tokenizer.encode(prompt)
        
        # Create sequence
        seq_id = f"{request_id}-0"
        sequence = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        
        # Create sequence group
        seq_group = SequenceGroup(
            request_id=request_id,
            sequences=[sequence],
            sampling_params=sampling_params,
        )
        
        # Add to scheduler
        self.scheduler.add_seq_group(seq_group)
        self.stats["num_requests"] += 1
        
        logger.debug(f"Added request {request_id} with {len(prompt_token_ids)} tokens")
    
    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_seq_group(request_id)
    
    async def step(self) -> List[SequenceGroup]:
        """Execute one step of the engine."""
        if self.model is None:
            return []
        
        # Schedule sequences
        scheduler_output = self.scheduler.schedule()
        
        if scheduler_output.is_empty():
            return []
        
        # Execute model
        model_input = self._prepare_model_input(scheduler_output.scheduled_seq_groups)
        model_output = await self._execute_model(model_input)
        
        # Process outputs
        finished_seq_groups = self._process_model_output(
            model_output, scheduler_output.scheduled_seq_groups
        )
        
        return finished_seq_groups
    
    def _prepare_model_input(self, seq_groups: List[SequenceGroup]) -> ModelInput:
        """Prepare input for the model."""
        input_ids = []
        positions = []
        seq_lens = []
        query_lens = []
        
        for seq_group in seq_groups:
            for seq in seq_group.sequences:
                if seq.status != SequenceStatus.RUNNING:
                    continue
                
                # Get tokens to process
                if seq.get_output_len() == 0:
                    # Prompt phase - process all prompt tokens
                    tokens = seq.prompt_token_ids
                    pos = list(range(len(tokens)))
                    query_len = len(tokens)
                else:
                    # Decode phase - process last token
                    tokens = [seq.output_token_ids[-1]]
                    pos = [seq.get_len() - 1]
                    query_len = 1
                
                input_ids.append(tokens)
                positions.append(pos)
                seq_lens.append(seq.get_len())
                query_lens.append(query_len)
        
        return ModelInput(
            input_ids=input_ids,
            positions=positions,
            seq_lens=seq_lens,
            query_lens=query_lens,
            max_seq_len=max(seq_lens) if seq_lens else 0,
            max_query_len=max(query_lens) if query_lens else 0,
        )
    
    async def _execute_model(self, model_input: ModelInput) -> ModelOutput:
        """Execute the model."""
        if not model_input.input_ids:
            return ModelOutput(logits=mx.array([]))
        
        # Convert to MLX arrays
        batch_tokens = []
        for tokens in model_input.input_ids:
            batch_tokens.extend(tokens)
        
        if not batch_tokens:
            return ModelOutput(logits=mx.array([]))
        
        input_array = mx.array(batch_tokens).reshape(len(model_input.input_ids), -1)
        
        # Forward pass
        start_time = time.time()
        logits = self.model.forward(input_array)
        execution_time = time.time() - start_time
        
        self.stats["total_generation_time"] += execution_time
        
        return ModelOutput(logits=logits)
    
    def _process_model_output(
        self, 
        model_output: ModelOutput, 
        seq_groups: List[SequenceGroup]
    ) -> List[SequenceGroup]:
        """Process model output and update sequences."""
        finished_seq_groups = []
        
        if model_output.logits.size == 0:
            return finished_seq_groups
        
        logits = model_output.logits
        seq_idx = 0
        
        for seq_group in seq_groups:
            for seq in seq_group.sequences:
                if seq.status != SequenceStatus.RUNNING:
                    continue
                
                if seq_idx >= logits.shape[0]:
                    break
                
                # Sample next token
                seq_logits = logits[seq_idx, -1, :]  # Last position
                next_token = self._sample_token(seq_logits, seq.sampling_params)
                
                # Add token to sequence
                seq.append_token_id(int(next_token))
                self.stats["num_tokens_generated"] += 1
                
                # Update metrics
                if seq.get_output_len() == 1 and seq.metrics.first_token_time is None:
                    seq.metrics.first_token_time = time.time()
                
                # Check stopping conditions
                if self._should_stop_sequence(seq):
                    seq.status = SequenceStatus.FINISHED_STOPPED
                    seq.metrics.finished_time = time.time()
                
                seq_idx += 1
            
            # Check if sequence group is finished
            if seq_group.is_finished():
                finished_seq_groups.append(seq_group)
        
        return finished_seq_groups
    
    def _sample_token(self, logits: mx.array, sampling_params: SamplingParams) -> mx.array:
        """Sample a token from logits."""
        # Apply temperature
        if sampling_params.temperature != 1.0:
            logits = logits / sampling_params.temperature
        
        # Apply top-k
        if sampling_params.top_k > 0:
            top_k_logits, top_k_indices = mx.topk(logits, sampling_params.top_k)
            # Create mask for top-k
            mask = mx.full_like(logits, float('-inf'))
            mask = mask.at[top_k_indices].set(top_k_logits)
            logits = mask
        
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Apply top-p (nucleus sampling)
        if sampling_params.top_p < 1.0:
            sorted_probs, sorted_indices = mx.sort(probs, axis=-1)[::-1], mx.argsort(probs, axis=-1)[::-1]
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff
            cutoff_idx = mx.argmax(cumsum_probs >= sampling_params.top_p)
            if cutoff_idx == 0:
                cutoff_idx = 1
            
            # Zero out probabilities beyond cutoff
            mask = mx.arange(probs.shape[-1]) <= cutoff_idx
            probs = mx.where(mask[sorted_indices], probs, 0.0)
            probs = probs / mx.sum(probs)  # Renormalize
        
        # Sample
        return mx.random.categorical(mx.log(probs + 1e-8))
    
    def _should_stop_sequence(self, seq: Sequence) -> bool:
        """Check if sequence should stop."""
        # Check max tokens
        if seq.get_output_len() >= seq.sampling_params.max_tokens:
            return True
        
        # Check stop tokens
        if seq.output_token_ids and seq.output_token_ids[-1] in seq.sampling_params.stop_token_ids:
            return True
        
        # Check EOS token (assuming token 2 is EOS)
        if not seq.sampling_params.ignore_eos and seq.output_token_ids and seq.output_token_ids[-1] == 2:
            return True
        
        return False
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        if self.model is None:
            return {}
        
        return {
            "vocab_size": self.model.vocab_size,
            "max_model_len": self.config.max_model_len,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "num_waiting": len(self.scheduler.waiting),
            "num_running": len(self.scheduler.running),
            "num_swapped": len(self.scheduler.swapped),
        }
