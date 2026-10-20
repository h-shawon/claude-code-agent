import re
from pathlib import Path
from typing import Any, Dict, List


class SecurityViolationError(Exception):
    """Raised when an input or tool call violates OWASP LLM security boundaries."""
    pass


class SecurityGuardrail:
    """
    Implements defense-in-depth against OWASP LLM01 (Prompt Injection)
    and OWASP LLM07 (Insecure Plugin/Tool Design).
    """

    INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"system\s+override", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
        re.compile(r"<\|im_start\|>|<\|endoftext\|>", re.IGNORECASE),
    ]

    FORBIDDEN_SHELL_PATTERNS: List[re.Pattern] = [
        re.compile(r"(rm\s+-rf|mkfs|dd\s+if=|:(){ :\|:& };:)", re.IGNORECASE),
        re.compile(r"(/etc/shadow|/etc/passwd|\.env)", re.IGNORECASE),
        re.compile(r"(curl|wget)\s+.*\|\s*(sh|bash)", re.IGNORECASE),
    ]

    def __init__(self, allowed_workspace: Path) -> None:
        self.allowed_workspace = allowed_workspace.resolve()

    def validate_user_prompt(self, prompt: str) -> str:
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(prompt):
                raise SecurityViolationError(
                    f"Prompt rejected by OWASP LLM01 Guardrail: matched pattern '{pattern.pattern}'"
                )
        return prompt.strip()

    def validate_tool_arguments(self, tool_name: str, args: Dict[str, Any]) -> None:
        serialized = str(args)
        for pattern in self.FORBIDDEN_SHELL_PATTERNS:
            if pattern.search(serialized):
                raise SecurityViolationError(
                    f"Tool '{tool_name}' blocked: dangerous argument pattern detected."
                )

        if "filepath" in args or "path" in args:
            raw_path = args.get("filepath") or args.get("path")
            target = (self.allowed_workspace / str(raw_path)).resolve()
            if not str(target).startswith(str(self.allowed_workspace)):
                raise SecurityViolationError(
                    f"Path traversal blocked: '{raw_path}' escapes sandbox '{self.allowed_workspace}'."
                )
