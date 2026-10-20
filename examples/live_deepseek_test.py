import asyncio
import ast
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import (
    AgentOrchestrator,
    DeepSeekProvider,
    ToolRegistry,
    tool,
)

load_dotenv()

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


@tool(registry)
def compute_modular_hash(payload: str, prime_mod: int) -> int:
    """Computes a polynomial rolling hash modulo prime_mod."""
    h = 0
    for ch in payload:
        h = (h * 31 + ord(ch)) % prime_mod
    return h


async def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEFAULT_MODEL", "deepseek-v4-flash")

    if not api_key or "your-deepseek-key" in api_key:
        print("[!] Error: Set a valid DEEPSEEK_API_KEY in your .env file first.")
        return

    print(f"[*] Initializing DeepSeekProvider with model: {model}")
    provider = DeepSeekProvider(api_key=api_key, model=model)
    orchestrator = AgentOrchestrator(provider=provider, registry=registry)

    prompt = (
        "1. Use `analyze_python_ast` with max_complexity=4 to check this code: "
        "'def check(x):\\n    if x > 0:\\n        for i in range(x):\\n            print(i)'\n"
        "2. Then use `compute_modular_hash` to hash the function name 'check' with prime_mod=997.\n"
        "Report both results clearly."
    )

    final_answer, trace = await orchestrator.run(prompt)

    print("\n=== LIVE DEEPSEEK-V4-FLASH RESPONSE ===")
    print(final_answer)

    print("\n=== EXECUTION TRACE ===")
    for step in trace.steps:
        print(
            f"[Step {step.step_index}] Phase={step.phase.value:<16} | "
            f"Tool={str(step.tool_invoked):<22} | "
            f"Error={str(step.tool_error):<15} | "
            f"Latency={step.latency_ms}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
