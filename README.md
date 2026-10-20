# Claude Code Agent — Modular LLM Tooling & Workflow Pipeline

![CI](https://github.com/h-shawon/claude-code-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight, first-principles agentic execution engine in Python 3.11+ designed for deterministic multi-step reasoning, schema-validated tool dispatch, and automated self-correcting execution loops across **Anthropic (Claude)** and **DeepSeek** models.

Built **without heavy orchestration wrappers** (zero LangChain/LlamaIndex dependencies) to maintain full control over state transitions, context-window token pruning, and security boundaries.

## Key Architectural Highlights

- **Deterministic State-Machine Orchestrator (`agent_core/orchestrator.py`)**: Explicit phase transitions (`PLANNING` -> `TOOL_DISPATCH` -> `SELF_CORRECTION` -> `COMPLETED`) with full execution trace telemetry.
- **Self-Correcting Reasoning Loop**: Automatically intercepts tool validation failures, schema mismatches, and runtime exceptions, feeding structured error traces back into the model context for autonomous recovery up to configurable retry bounds.
- **Zero-Dependency Reflection Tool Registry (`agent_core/tools.py`)**: Uses Python's native `inspect` and type-hinting AST to dynamically compile Python functions into Anthropic and OpenAI-compatible JSON Schemas with strict runtime type coercion.
- **OWASP LLM Top-10 Guardrails (`agent_core/guardrails.py`)**: Mitigates **LLM01 (Prompt Injection)** via regex heuristics and **LLM07 (Insecure Plugin Design)** via workspace path-traversal sandboxing and shell-command filtering.
- **Sliding-Window Context Manager (`agent_core/memory.py`)**: Preserves the system directive and pinned user objective while deterministically evicting stale intermediate tool turns to stay within token budgets.

## Quickstart

### 1. Installation
```bash
git clone [https://github.com/h-shawon/claude-code-agent.git](https://github.com/h-shawon/claude-code-agent.git)
cd claude-code-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### 2. Run the Self-Correcting Workflow Demo (No API Key Required)
```bash
python3 examples/code_analysis_workflow.py
```

### 3. Run the Test Suite
```bash
pytest -v
```

## License
MIT
