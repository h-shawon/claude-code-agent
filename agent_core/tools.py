import inspect
from typing import Any, Callable, Dict, List, get_type_hints


PYTHON_TO_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolValidationError(Exception):
    """Raised when LLM-supplied arguments fail schema validation."""
    pass


class ToolRegistry:
    """
    First-principles function registry that introspects Python type annotations
    to build JSON Schemas and validates LLM arguments before execution.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        name = func.__name__
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        doc = (inspect.getdoc(func) or "No description provided.").strip()

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            ann = type_hints.get(param_name, str)
            json_type = PYTHON_TO_JSON_TYPES.get(ann, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": doc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        return func

    def get_schemas(self) -> List[Dict[str, Any]]:
        return list(self._schemas.values())

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._tools:
            raise ToolValidationError(
                f"Unknown tool '{name}'. Available tools: {list(self._tools.keys())}"
            )

        func = self._tools[name]
        schema = self._schemas[name]["input_schema"]
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        for req in schema["required"]:
            if req not in arguments:
                raise ToolValidationError(
                    f"Missing required parameter '{req}' for tool '{name}'."
                )

        validated_kwargs: Dict[str, Any] = {}
        for k, v in arguments.items():
            if k not in sig.parameters:
                raise ToolValidationError(
                    f"Unexpected parameter '{k}' passed to tool '{name}'."
                )
            expected_type = type_hints.get(k, str)
            if not isinstance(v, expected_type):
                try:
                    v = expected_type(v)
                except (ValueError, TypeError) as exc:
                    raise ToolValidationError(
                        f"Parameter '{k}' must be of type {expected_type.__name__}, got {type(v).__name__}."
                    ) from exc
            validated_kwargs[k] = v

        if inspect.iscoroutinefunction(func):
            result = await func(**validated_kwargs)
        else:
            result = func(**validated_kwargs)

        return str(result)


def tool(registry: ToolRegistry) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return registry.register(func)
    return decorator
