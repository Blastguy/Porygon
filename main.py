"""porygon: a simple from-scratch ReAct agent.

Usage:
    uv run python main.py --api-key sk-ant-... [--workdir ./workspace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from agent import Agent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="porygon",
        description="A simple ReAct agent with calculator, Wikipedia, and file I/O tools.",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Anthropic API key.",
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
            print("Error: authentication failed — check your --api-key.")
            break
        except anthropic.RateLimitError:
            print("Error: rate limited. Wait a moment and try again.")
            continue
        except anthropic.APIError as exc:
            print(f"Error: API request failed: {exc}")
            continue

        print(f"\n{answer}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    memory_dir = args.memory_dir
    memory_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=args.api_key)
    agent = Agent(client, workdir, memory_dir)

    repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
