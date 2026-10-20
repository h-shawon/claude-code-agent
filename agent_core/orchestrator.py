import time
import uuid
from pathlib import Path
from typing import Optional

from .guardrails import SecurityGuardrail, SecurityViolationError
from .memory import ContextWindowManager
from .providers import LLMProvider
from .state import AgentPhase, ExecutionTrace, Message, Role
from .tools import ToolRegistry, ToolValidationError


SYSTEM_DIRECTIVE = (
    "You are an autonomous engineering agent operating inside a deterministic execution graph. "
    "When a tool execution fails or returns a validation error, analyze the traceback carefully "
    "and issue a corrected tool call. Never guess file contents."
)


class AgentOrchestrator:
    """
    State-machine orchestrator supporting multi-step tool dispatch and
    automatic self-correcting reasoning loops on tool failure.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        workspace: Optional[Path] = None,
        max_steps: int = 8,
        max_retries_per_step: int = 3,
        token_budget: int = 6000,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.guardrail = SecurityGuardrail(workspace or Path.cwd())
        self.memory = ContextWindowManager(max_token_budget=token_budget)
        self.max_steps = max_steps
        self.max_retries_per_step = max_retries_per_step

    async def run(self, user_goal: str) -> tuple[str, ExecutionTrace]:
        trace = ExecutionTrace(task_id=str(uuid.uuid4())[:8])

        sanitized_goal = self.guardrail.validate_user_prompt(user_goal)

        self.memory.append(Message(role=Role.SYSTEM, content=SYSTEM_DIRECTIVE))
        self.memory.append(Message(role=Role.USER, content=sanitized_goal))

        consecutive_errors = 0

        for _ in range(self.max_steps):
            step_start = time.perf_counter()
            current_phase = (
                AgentPhase.SELF_CORRECTION
                if consecutive_errors > 0
                else AgentPhase.PLANNING
            )

            assistant_msg = await self.provider.generate(
                messages=self.memory.get_context(),
                tools=self.registry.get_schemas(),
            )
            self.memory.append(assistant_msg)

            if not assistant_msg.tool_calls:
                latency = (time.perf_counter() - step_start) * 1000
                trace.add_step(
                    phase=AgentPhase.COMPLETED,
                    thought=assistant_msg.content,
                    latency_ms=latency,
                )
                trace.final_status = AgentPhase.COMPLETED
                return assistant_msg.content, trace

            for tool_call in assistant_msg.tool_calls:
                tc_start = time.perf_counter()
                try:
                    self.guardrail.validate_tool_arguments(
                        tool_call.name, tool_call.arguments
                    )
                    output = await self.registry.execute(
                        tool_call.name, tool_call.arguments
                    )
                    self.memory.append(
                        Message(
                            role=Role.TOOL,
                            content=output,
                            tool_call_id=tool_call.id,
                            is_error=False,
                        )
                    )
                    trace.add_step(
                        phase=current_phase,
                        thought=assistant_msg.content or f"Invoke {tool_call.name}",
                        tool_invoked=tool_call.name,
                        latency_ms=(time.perf_counter() - tc_start) * 1000,
                    )
                    consecutive_errors = 0

                except (ToolValidationError, SecurityViolationError, Exception) as exc:
                    consecutive_errors += 1
                    trace.total_self_corrections += 1
                    error_feedback = (
                        f"[TOOL_EXECUTION_ERROR] {type(exc).__name__}: {exc}. "
                        f"Reflect on the required schema/constraints and retry."
                    )
                    self.memory.append(
                        Message(
                            role=Role.TOOL,
                            content=error_feedback,
                            tool_call_id=tool_call.id,
                            is_error=True,
                        )
                    )
                    trace.add_step(
                        phase=AgentPhase.SELF_CORRECTION,
                        thought=assistant_msg.content or f"Failed call to {tool_call.name}",
                        tool_invoked=tool_call.name,
                        tool_error=str(exc),
                        latency_ms=(time.perf_counter() - tc_start) * 1000,
                    )

                    if consecutive_errors >= self.max_retries_per_step:
                        trace.final_status = AgentPhase.FAILED
                        raise RuntimeError(
                            f"Agent exceeded maximum self-correction retries ({self.max_retries_per_step})."
                        ) from exc

        trace.final_status = AgentPhase.FAILED
        raise RuntimeError(f"Agent exceeded max reasoning steps ({self.max_steps}).")
