import abc
import json
import os
from typing import Any, Dict, List
import httpx

from .state import Message, Role, ToolCall


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self, messages: List[Message], tools: List[Dict[str, Any]]
    ) -> Message:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Direct HTTPX implementation for Anthropic Messages API with tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-latest",
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.endpoint = "[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)"

    async def generate(
        self, messages: List[Message], tools: List[Dict[str, Any]]
    ) -> Message:
        system_prompt = next((m.content for m in messages if m.role == Role.SYSTEM), "")
        chat_payload = []

        for m in messages:
            if m.role == Role.SYSTEM:
                continue
            if m.role == Role.TOOL:
                chat_payload.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                        "is_error": m.is_error,
                    }],
                })
            elif m.role == Role.ASSISTANT and m.tool_calls:
                blocks: List[Dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                chat_payload.append({"role": "assistant", "content": blocks})
            else:
                chat_payload.append({"role": m.role.value, "content": m.content})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": chat_payload,
        }
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        return Message(
            role=Role.ASSISTANT,
            content="\n".join(text_parts),
            tool_calls=tool_calls or None,
        )


class DeepSeekProvider(LLMProvider):
    """Direct HTTPX implementation for DeepSeek OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.endpoint = "[https://api.deepseek.com/chat/completions](https://api.deepseek.com/chat/completions)"

    async def generate(
        self, messages: List[Message], tools: List[Dict[str, Any]]
    ) -> Message:
        formatted_messages = []
        for m in messages:
            if m.role == Role.TOOL:
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                })
            elif m.role == Role.ASSISTANT and m.tool_calls:
                formatted_messages.append({
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                })
            else:
                formatted_messages.append({"role": m.role.value, "content": m.content})

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if openai_tools:
            body["tools"] = openai_tools

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]

        tool_calls = None
        if choice.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for tc in choice["tool_calls"]
            ]

        return Message(
            role=Role.ASSISTANT,
            content=choice.get("content") or "",
            tool_calls=tool_calls,
        )


class MockDeterministicProvider(LLMProvider):
    """
    Deterministic scripted LLM provider for offline execution and unit testing.
    """

    def __init__(self, scripted_responses: List[Message]) -> None:
        self._responses = list(scripted_responses)
        self._cursor = 0

    async def generate(
        self, messages: List[Message], tools: List[Dict[str, Any]]
    ) -> Message:
        if self._cursor >= len(self._responses):
            return Message(role=Role.ASSISTANT, content="Workflow completed.")
        reply = self._responses[self._cursor]
        self._cursor += 1
        return reply
