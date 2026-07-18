"""porygon: a simple from-scratch ReAct agent.

Usage:
    uv run python main.py [--provider anthropic|gemini] [--api-key ...]
                          [--model ...] [--workdir ./workspace] [--memory-dir ./memory]

The API key may also come from ANTHROPIC_API_KEY or GEMINI_API_KEY,
depending on the selected provider.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic
from google.genai import errors as genai_errors

import providers
import tools
from agent import SYSTEM_PROMPT, Agent

API_KEY_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="porygon",
        description="A simple ReAct agent with calculator, Wikipedia, and file I/O tools.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="Model provider to use (default: anthropic).",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "API key for the selected provider. Falls back to the "
            "ANTHROPIC_API_KEY or GEMINI_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--model",
        help="Model name override (default: the provider's default model).",
    )
    parser.add_argument(
        "--workdir",
        default="./workspace",
        type=Path,
        help="Sandbox directory for the file_io tool (default: ./workspace).",
    )
    parser.add_argument(
        "--memory-dir",
        default="./memory",
        type=Path,
        help="Directory for long-term memory files (default: ./memory).",
    )
    return parser.parse_args(argv)


def repl(agent: Agent) -> None:
    """Read user input, run a turn, print the answer; loop until exit/EOF."""
    print("porygon ReAct agent. Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n(interrupted)")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            answer = agent.run_turn(user_input)
        except anthropic.AuthenticationError:
            print("Error: authentication failed — check your API key.")
            break
        except anthropic.RateLimitError:
            print("Error: rate limited. Wait a moment and try again.")
            continue
        except (anthropic.APIError, genai_errors.APIError) as exc:
            print(f"Error: API request failed: {exc}")
            continue

        print(f"\n{answer}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.environ.get(API_KEY_ENV_VARS[args.provider])
    if not api_key:
        print(
            f"Error: no API key. Pass --api-key or set "
            f"{API_KEY_ENV_VARS[args.provider]}."
        )
        return 1

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    memory_dir = args.memory_dir
    memory_dir.mkdir(parents=True, exist_ok=True)

    provider = providers.build_provider(
        args.provider, api_key, args.model, SYSTEM_PROMPT, tools.TOOL_SCHEMAS
    )
    agent = Agent(provider, workdir, memory_dir)

    repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
