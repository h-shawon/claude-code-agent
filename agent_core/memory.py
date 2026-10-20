from typing import List
from .state import Message, Role


class ContextWindowManager:
    """
    Deterministic sliding-window context manager.
    Preserves the System Prompt and pinned initial user task while pruning
    older intermediate tool turns when approaching the token budget limit.
    """

    def __init__(self, max_token_budget: int = 6000) -> None:
        self.max_token_budget = max_token_budget
        self._messages: List[Message] = []

    def append(self, message: Message) -> None:
        self._messages.append(message)
        self._enforce_budget()

    def get_context(self) -> List[Message]:
        return list(self._messages)

    @property
    def current_token_usage(self) -> int:
        return sum(m.estimated_tokens for m in self._messages)

    def _enforce_budget(self) -> None:
        while self.current_token_usage > self.max_token_budget and len(self._messages) > 4:
            evicted = self._messages.pop(2)
            if evicted.role == Role.ASSISTANT and evicted.tool_calls and len(self._messages) > 2:
                if self._messages[2].role == Role.TOOL:
                    self._messages.pop(2)
