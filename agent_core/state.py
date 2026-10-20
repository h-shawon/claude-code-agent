from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class AgentPhase(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    TOOL_DISPATCH = "TOOL_DISPATCH"
    SELF_CORRECTION = "SELF_CORRECTION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    is_error: bool = False
    estimated_tokens: int = 0

    def __post_init__(self) -> None:
        if self.estimated_tokens == 0:
            raw_len = len(self.content or "")
            if self.tool_calls:
                raw_len += sum(len(tc.name) + len(str(tc.arguments)) for tc in self.tool_calls)
            self.estimated_tokens = max(1, (raw_len // 4) + 8)


@dataclass
class StepRecord:
    step_index: int
    phase: AgentPhase
    thought: str
    tool_invoked: Optional[str] = None
    tool_error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ExecutionTrace:
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    total_self_corrections: int = 0
    final_status: AgentPhase = AgentPhase.IDLE
    start_time: float = field(default_factory=time.perf_counter)

    def add_step(
        self,
        phase: AgentPhase,
        thought: str,
        tool_invoked: Optional[str] = None,
        tool_error: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.steps.append(
            StepRecord(
                step_index=len(self.steps) + 1,
                phase=phase,
                thought=thought,
                tool_invoked=tool_invoked,
                tool_error=tool_error,
                latency_ms=round(latency_ms, 2),
            )
        )
