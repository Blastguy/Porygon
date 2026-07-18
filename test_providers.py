"""Unit tests for the provider adapter layer (offline — no API calls).

Response parsing and history-shape logic are tested with fakes; the live
request paths of both providers are covered by a manual smoke test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import providers
import tools


def _block(**kwargs):
    return SimpleNamespace(**kwargs)


# --- schema conversion -----------------------------------------------------


def test_to_gemini_declarations_covers_all_tools():
    decls = providers.to_gemini_declarations(tools.TOOL_SCHEMAS)
    assert [d["name"] for d in decls] == [s["name"] for s in tools.TOOL_SCHEMAS]
    for decl, schema in zip(decls, tools.TOOL_SCHEMAS):
        assert decl["description"] == schema["description"]
        assert decl["parameters"] == schema["input_schema"]


# --- Anthropic response parsing --------------------------------------------


def test_parse_anthropic_tool_use():
    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            _block(type="text", text="Let me check."),
            _block(type="tool_use", id="t1", name="calculator", input={"expression": "1+1"}),
        ],
    )
    result = providers.parse_anthropic_response(response)
    assert result.status == "tool_use"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert (call.id, call.name, call.args) == ("t1", "calculator", {"expression": "1+1"})


def test_parse_anthropic_end_turn():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="Hello"), _block(type="text", text="world")],
    )
    result = providers.parse_anthropic_response(response)
    assert result.status == "done"
    assert result.text == "Hello\nworld"
    assert result.tool_calls == []


def test_parse_anthropic_pause_turn():
    response = SimpleNamespace(stop_reason="pause_turn", content=[])
    assert providers.parse_anthropic_response(response).status == "continue"


# --- Gemini response parsing -----------------------------------------------


def _gemini_response(parts):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )


def test_parse_gemini_function_calls_get_synthesized_ids():
    response = _gemini_response([
        _block(text=None, function_call=SimpleNamespace(name="calculator", args={"expression": "2*2"})),
        _block(text=None, function_call=SimpleNamespace(name="memory", args={"operation": "list"})),
    ])
    result = providers.parse_gemini_response(response)
    assert result.status == "tool_use"
    assert [c.id for c in result.tool_calls] == ["call_0", "call_1"]
    assert result.tool_calls[0].args == {"expression": "2*2"}
    assert result.tool_calls[1].name == "memory"


def test_parse_gemini_text_only():
    response = _gemini_response([_block(text="The answer is 4.", function_call=None)])
    result = providers.parse_gemini_response(response)
    assert result.status == "done"
    assert result.text == "The answer is 4."
    assert result.tool_calls == []


def test_parse_gemini_empty_candidates():
    result = providers.parse_gemini_response(SimpleNamespace(candidates=[]))
    assert result.status == "done"
    assert result.text == ""


# --- provider history shapes -----------------------------------------------


def _anthropic_provider():
    return providers.AnthropicProvider(
        "mock_key_for_testing", providers.ANTHROPIC_MODEL, "system", tools.TOOL_SCHEMAS
    )


def test_anthropic_provider_history_shapes():
    provider = _anthropic_provider()
    provider.add_user_message("hi")
    assert provider.messages == [{"role": "user", "content": "hi"}]

    provider.add_tool_results(
        [providers.ToolResult("t1", "calculator", "2", False)]
    )
    assert provider.messages[-1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "2", "is_error": False}
        ],
    }


def test_gemini_provider_history_shapes():
    provider = providers.GeminiProvider(
        "mock_key_for_testing", providers.GEMINI_MODEL, "system", tools.TOOL_SCHEMAS
    )
    provider.add_user_message("hi")
    assert provider.contents[-1].role == "user"
    assert provider.contents[-1].parts[0].text == "hi"

    provider.add_tool_results(
        [
            providers.ToolResult("call_0", "calculator", "4", False),
            providers.ToolResult("call_1", "memory", "Error: nope", True),
        ]
    )
    parts = provider.contents[-1].parts
    assert len(parts) == 2
    assert parts[0].function_response.name == "calculator"
    assert parts[0].function_response.response == {"result": "4"}
    assert parts[1].function_response.response == {"error": "Error: nope"}


# --- build_provider --------------------------------------------------------


def test_build_provider_defaults_and_overrides():
    provider = providers.build_provider(
        "anthropic", "mock_key_for_testing", None, "system", tools.TOOL_SCHEMAS
    )
    assert isinstance(provider, providers.AnthropicProvider)
    assert provider.model == providers.ANTHROPIC_MODEL

    provider = providers.build_provider(
        "gemini", "mock_key_for_testing", "gemini-2.5-pro", "system", tools.TOOL_SCHEMAS
    )
    assert isinstance(provider, providers.GeminiProvider)
    assert provider.model == "gemini-2.5-pro"


def test_build_provider_unknown():
    with pytest.raises(ValueError):
        providers.build_provider("openai", "k", None, "system", [])
