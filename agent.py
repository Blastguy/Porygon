"""The porygon ReAct agent: a hand-written agentic loop over a model provider.

The loop *is* the ReAct engine. Each turn, the model reasons, acts (requests
tool calls), and we feed back observations until it produces a final answer.
Provider specifics (Anthropic vs Gemini message formats, tool-calling schemas)
live behind the ``providers.Provider`` interface.
"""

from __future__ import annotations

from pathlib import Path

import providers
import tools

MAX_ITERATIONS = 10  # tool rounds per user turn, so a stuck model can't loop forever

SYSTEM_PROMPT = (
    "You are porygon, a concise ReAct-style assistant. You have five tools:\n"
    "  - calculator: for exact arithmetic.\n"
    "  - wikipedia_search: for factual/encyclopedic lookups.\n"
    "  - file_io: to read, write, and list files in your sandboxed working directory.\n"
    "  - scratchpad: private working notes for the current task; append intermediate "
    "results as you gather them and read them back before answering.\n"
    "  - memory: long-term memory files that persist across sessions. When a task "
    "might depend on saved context, 'list' the memory files first, then 'read' only "
    "the relevant ones. Save durable facts the user asks you to remember with "
    "'write' (read the file first when updating it).\n"
    "Reason about what the user needs, call a tool when it helps, and use the tool "
    "results to answer. Prefer tools over guessing for calculations and facts. When "
    "you have enough information, give a direct final answer."
)


class Agent:
    """Holds the provider and tool context, and runs the agentic loop."""

    def __init__(
        self, provider: providers.Provider, workdir: Path, memory_dir: Path
    ) -> None:
        self.provider = provider
        self.ctx = tools.ToolContext(workdir, memory_dir)

    def run_turn(self, user_input: str) -> str:
        """Run one user turn to completion; return the final answer text."""
        self.provider.add_user_message(user_input)

        for _ in range(MAX_ITERATIONS):
            result = self.provider.step()

            if result.status == "tool_use":
                self.provider.add_tool_results(self._run_tools(result.tool_calls))
                continue

            if result.status == "continue":
                # Server-side continuation; re-send to resume.
                continue

            return result.text or "(no answer)"

        return "(stopped: reached the tool-call limit for this turn.)"

    def _run_tools(
        self, tool_calls: list[providers.ToolCall]
    ) -> list[providers.ToolResult]:
        """Execute every requested tool call and collect the results."""
        results = []
        for call in tool_calls:
            print(f"  [tool] {call.name}({_fmt_input(call.args)})")
            result, is_error = tools.dispatch(call.name, call.args, self.ctx)
            results.append(providers.ToolResult(call.id, call.name, result, is_error))
        return results


def _fmt_input(tool_input: dict) -> str:
    """Compact one-line rendering of tool input for the trace line."""
    items = []
    for key, value in tool_input.items():
        text = str(value).replace("\n", " ")
        if len(text) > 60:
            text = text[:57] + "..."
        items.append(f"{key}={text!r}")
    return ", ".join(items)
