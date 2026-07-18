"""Provider adapters: one normalized step interface over Anthropic and Gemini.

The ReAct loop in ``agent.py`` only sees the normalized per-step surface
(``StepResult`` / ``ToolCall`` / ``ToolResult``). Each adapter owns its
conversation history in its provider's native format — for Anthropic this is
required, because adaptive thinking emits signed thinking blocks that must be
replayed verbatim during tool-use loops.

Gemini function calls carry no ids, so ``parse_gemini_response`` synthesizes
them (``call_0``, ``call_1``, ...) and results are matched back by function
name; parallel calls to the *same* tool in one step rely on ordering.
"""

from __future__ import annotations

from typing import Protocol

import anthropic
from google import genai
from google.genai import types as genai_types

ANTHROPIC_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 16000


class ToolCall:
    """A single tool invocation requested by the model."""

    def __init__(self, id: str, name: str, args: dict) -> None:
        self.id = id
        self.name = name
        self.args = args


class ToolResult:
    """The outcome of one tool call, ready to feed back to the model."""

    def __init__(self, id: str, name: str, content: str, is_error: bool) -> None:
        self.id = id  # Anthropic tool_use_id
        self.name = name  # Gemini functionResponse key
        self.content = content
        self.is_error = is_error


class StepResult:
    """Normalized outcome of one model call.

    ``status`` is one of:
      - ``"tool_use"``  the model wants tools run (see ``tool_calls``)
      - ``"continue"``  re-send to resume a server-side continuation
      - ``"done"``      final answer available in ``text``
    """

    def __init__(self, status: str, text: str, tool_calls: list[ToolCall]) -> None:
        self.status = status
        self.text = text
        self.tool_calls = tool_calls


class Provider(Protocol):
    """What the agent loop needs from a model backend."""

    def add_user_message(self, text: str) -> None: ...

    def step(self) -> StepResult: ...

    def add_tool_results(self, results: list[ToolResult]) -> None: ...


# ---------------------------------------------------------------------------
# response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------


def parse_anthropic_response(response) -> StepResult:
    """Normalize an Anthropic Messages API response into a StepResult."""
    tool_calls = [
        ToolCall(block.id, block.name, block.input)
        for block in response.content
        if block.type == "tool_use"
    ]
    text = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if response.stop_reason == "tool_use":
        status = "tool_use"
    elif response.stop_reason == "pause_turn":
        status = "continue"
    else:
        status = "done"
    return StepResult(status, text, tool_calls)


def parse_gemini_response(response) -> StepResult:
    """Normalize a Gemini generate_content response into a StepResult."""
    candidates = getattr(response, "candidates", None) or []
    parts = []
    if candidates and candidates[0].content is not None:
        parts = candidates[0].content.parts or []

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            tool_calls.append(
                ToolCall(
                    f"call_{len(tool_calls)}",
                    function_call.name,
                    dict(function_call.args or {}),
                )
            )
        elif getattr(part, "text", None):
            texts.append(part.text)

    status = "tool_use" if tool_calls else "done"
    return StepResult(status, "\n".join(texts).strip(), tool_calls)


def to_gemini_declarations(tool_schemas: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool schemas to Gemini function declarations."""
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        }
        for schema in tool_schemas
    ]


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Anthropic Messages API adapter; history is raw content-block messages."""

    def __init__(
        self, api_key: str, model: str, system_prompt: str, tool_schemas: list[dict]
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.messages: list[dict] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> StepResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=self.system_prompt,
            tools=self.tool_schemas,
            messages=self.messages,
        )
        # Raw content blocks preserve signed thinking blocks for replay.
        self.messages.append({"role": "assistant", "content": response.content})
        return parse_anthropic_response(response)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in results
                ],
            }
        )


class GeminiProvider:
    """Gemini API adapter; history is a list of ``types.Content`` objects."""

    def __init__(
        self, api_key: str, model: str, system_prompt: str, tool_schemas: list[dict]
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[
                genai_types.Tool(
                    function_declarations=to_gemini_declarations(tool_schemas)
                )
            ],
            max_output_tokens=MAX_TOKENS,
        )
        self.contents: list[genai_types.Content] = []

    def add_user_message(self, text: str) -> None:
        self.contents.append(
            genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
        )

    def step(self) -> StepResult:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.contents,
            config=self.config,
        )
        candidates = response.candidates or []
        if candidates and candidates[0].content is not None:
            self.contents.append(candidates[0].content)
        return parse_gemini_response(response)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        parts = [
            genai_types.Part.from_function_response(
                name=result.name,
                response=(
                    {"error": result.content}
                    if result.is_error
                    else {"result": result.content}
                ),
            )
            for result in results
        ]
        self.contents.append(genai_types.Content(role="user", parts=parts))


def build_provider(
    provider: str,
    api_key: str,
    model: str | None,
    system_prompt: str,
    tool_schemas: list[dict],
) -> Provider:
    """Construct the requested provider adapter with its default model."""
    if provider == "anthropic":
        return AnthropicProvider(
            api_key, model or ANTHROPIC_MODEL, system_prompt, tool_schemas
        )
    if provider == "gemini":
        return GeminiProvider(
            api_key, model or GEMINI_MODEL, system_prompt, tool_schemas
        )
    raise ValueError(f"unknown provider: {provider!r}")
