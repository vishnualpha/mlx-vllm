"""
Core data types and structures for MLX-vLLM.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import time


class SequenceStatus(Enum):
    """Status of a sequence in the system."""
    WAITING = "waiting"
    RUNNING = "running" 
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_IGNORED = "finished_ignored"


@dataclass
class SamplingParams:
    """Parameters for sampling during generation."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    arrival_time: float = field(default_factory=time.time)
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finished_time: Optional[float] = None
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time from arrival to first token."""
        if self.first_token_time and self.arrival_time:
            return self.first_token_time - self.arrival_time
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        """Total time from arrival to completion."""
        if self.finished_time and self.arrival_time:
            return self.finished_time - self.arrival_time
        return None


@dataclass
class Sequence:
    """A single sequence being generated."""
    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    metrics: RequestMetrics = field(default_factory=lambda: RequestMetrics(""))
    
    def __post_init__(self):
        if not self.metrics.request_id:
            self.metrics.request_id = self.seq_id
    
    @property
    def get_len(self) -> int:
        """Get total length of sequence."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)
    
    @property
    def get_prompt_len(self) -> int:
        """Get length of prompt."""
        return len(self.prompt_token_ids)
    
    @property
    def get_output_len(self) -> int:
        """Get length of generated output."""
        return len(self.output_token_ids)
    
    def append_token_id(self, token_id: int):
        """Append a token to the output."""
        self.output_token_ids.append(token_id)
    
    def is_finished(self) -> bool:
        """Check if sequence is finished."""
        return self.status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]


@dataclass
class SequenceGroup:
    """A group of sequences that share the same prompt (for beam search, etc)."""
    request_id: str
    sequences: List[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    
    def get_max_num_running_seqs(self) -> int:
        """Get maximum number of running sequences."""
        return len(self.sequences)
    
    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        """Get sequences with optional status filter."""
        if status is None:
            return self.sequences
        return [seq for seq in self.sequences if seq.status == status]
    
    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        """Get number of sequences with optional status filter."""
        return len(self.get_seqs(status))
    
    def is_finished(self) -> bool:
        """Check if all sequences in group are finished."""
        return all(seq.is_finished() for seq in self.sequences)


@dataclass
class ModelInput:
    """Input to the model for a batch."""
    input_ids: List[List[int]]
    positions: List[List[int]]
    seq_lens: List[int]
    query_lens: List[int]
    max_seq_len: int
    max_query_len: int
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.input_ids)


@dataclass
class ModelOutput:
    """Output from the model."""
    logits: Any  # MLX array
    hidden_states: Optional[Any] = None
    
    
@dataclass
class SchedulerOutput:
    """Output from the scheduler."""
    scheduled_seq_groups: List[SequenceGroup]
    prompt_runs: List[SequenceGroup]
    decode_runs: List[SequenceGroup]
    preempted: List[SequenceGroup]
    swapped_in: List[SequenceGroup]
    swapped_out: List[SequenceGroup]
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    num_lookahead_slots: int = 0
    
    def is_empty(self) -> bool:
        """Check if scheduler output is empty."""
        return (
            len(self.scheduled_seq_groups) == 0
            and len(self.prompt_runs) == 0
            and len(self.decode_runs) == 0
        )


@dataclass
class EngineConfig:
    """Configuration for the MLX engine."""
    model_name: str
    max_model_len: int = 2048
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    block_size: int = 16
    enable_chunked_prefill: bool = True
    max_num_on_the_fly: int = 1
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_log_stats: bool = False
    
    # Apple Silicon specific
    enable_metal_performance_shaders: bool = True
    enable_neural_engine: bool = True
    memory_pool_size_gb: float = 8.0
    
    # Speculative decoding
    enable_speculative_decoding: bool = True
    draft_model_name: Optional[str] = None
    num_speculative_tokens: int = 4
    speculative_acceptance_threshold: float = 0.8
