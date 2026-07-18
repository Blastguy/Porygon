"""The porygon ReAct agent: a hand-written agentic loop over the Anthropic API.

The loop *is* the ReAct engine. Each turn, the model reasons (adaptive thinking),
acts (emits ``tool_use`` blocks), and we feed back observations (``tool_result``
blocks) until it produces a final answer (``stop_reason == "end_turn"``).
"""

from __future__ import annotations

from pathlib import Path

import anthropic

import tools

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000
MAX_ITERATIONS = 10  # tool rounds per user turn, so a stuck model can't loop forever

SYSTEM_PROMPT = (
    "You are porygon, a concise ReAct-style assistant. You have three tools:\n"
    "  - calculator: for exact arithmetic.\n"
    "  - wikipedia_search: for factual/encyclopedic lookups.\n"
    "  - file_io: to read, write, and list files in your sandboxed working directory.\n"
    "Reason about what the user needs, call a tool when it helps, and use the tool "
    "results to answer. Prefer tools over guessing for calculations and facts. When "
    "you have enough information, give a direct final answer."
)


class Agent:
    """Holds the API client and conversation state, and runs the agentic loop."""

    def __init__(self, client: anthropic.Anthropic, workdir: Path) -> None:
        self.client = client
        self.workdir = workdir
        self.messages: list[dict] = []

    def run_turn(self, user_input: str) -> str:
        """Run one user turn to completion; return the final answer text."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_ITERATIONS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=tools.TOOL_SCHEMAS,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                self._run_tools(response.content)
                continue

            if response.stop_reason == "pause_turn":
                # Server-side continuation; re-send to resume.
                continue

            return _text_of(response.content)

        return "(stopped: reached the tool-call limit for this turn.)"

    def _run_tools(self, content: list) -> None:
        """Execute every tool_use block and append the results as one user turn."""
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({_fmt_input(block.input)})")
            result, is_error = tools.dispatch(block.name, block.input, self.workdir)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )
        self.messages.append({"role": "user", "content": results})


def _text_of(content: list) -> str:
    """Join the text blocks of a response into a single string."""
    parts = [block.text for block in content if block.type == "text"]
    return "\n".join(parts).strip() or "(no answer)"


def _fmt_input(tool_input: dict) -> str:
    """Compact one-line rendering of tool input for the trace line."""
    items = []
    for key, value in tool_input.items():
        text = str(value).replace("\n", " ")
        if len(text) > 60:
            text = text[:57] + "..."
        items.append(f"{key}={text!r}")
    return ", ".join(items)
