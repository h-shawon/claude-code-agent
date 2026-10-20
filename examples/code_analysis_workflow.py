import asyncio
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import (
    AgentOrchestrator,
    MockDeterministicProvider,
    Message,
    Role,
    ToolCall,
    ToolRegistry,
    tool,
)

registry = ToolRegistry()


@tool(registry)
def analyze_python_ast(code: str, max_complexity: int) -> str:
    """
    Parses Python source code into an Abstract Syntax Tree (AST) and computes
    function count and branching complexity against a threshold.
    """
    tree = ast.parse(code)
    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    branches = sum(
        1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))
    )
    status = "PASS" if branches <= max_complexity else "EXCEEDS_THRESHOLD"
    return f"Functions={functions} | BranchCount={branches} | Status={status}"


async def main() -> None:
    sample_snippet = (
        "def process_queue(items):\n"
        "    for item in items:\n"
        "        if item > 0:\n"
        "            print(item)\n"
    )

    scripted_turns = [
        Message(
            role=Role.ASSISTANT,
            content="I will inspect the AST complexity of the provided snippet.",
            tool_calls=[
                ToolCall(
                    id="call_01",
                    name="analyze_python_ast",
                    arguments={"code": sample_snippet},
                )
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content="Self-correcting: 'max_complexity' is required by schema. Retrying with max_complexity=5.",
            tool_calls=[
                ToolCall(
                    id="call_02",
                    name="analyze_python_ast",
                    arguments={"code": sample_snippet, "max_complexity": 5},
                )
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content="AST Analysis Complete: `process_queue` contains 2 branches, well within the complexity threshold of 5.",
        ),
    ]

    provider = MockDeterministicProvider(scripted_turns)
    orchestrator = AgentOrchestrator(
        provider=provider,
        registry=registry,
        workspace=Path.cwd(),
    )

    final_output, trace = await orchestrator.run(
        "Analyze the AST complexity of `process_queue`."
    )

    print("=== FINAL AGENT OUTPUT ===")
    print(final_output)
    print("\n=== DETERMINISTIC EXECUTION TRACE ===")
    for step in trace.steps:
        print(
            f"[Step {step.step_index}] Phase={step.phase.value:<16} | "
            f"Tool={str(step.tool_invoked):<20} | "
            f"Error={str(step.tool_error):<45} | "
            f"Latency={step.latency_ms}ms"
        )
    print(f"\nTotal Self-Corrections Triggered: {trace.total_self_corrections}")


if __name__ == "__main__":
    asyncio.run(main())
