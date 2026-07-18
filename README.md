# porygon

A simple from-scratch ReAct agent that runs on the Anthropic or Gemini API. It
runs a hand-written agentic loop — the model reasons, calls tools, and observes
the results until it can answer — exposed as an interactive REPL.

## Tools

- **calculator** — exact arithmetic via a safe AST evaluator (no `eval`).
- **wikipedia_search** — intro summary of the best-matching Wikipedia article.
- **file_io** — read / write / list files inside a sandboxed working directory.
- **scratchpad** — in-memory working notes that persist across ReAct steps
  within a session (read / write / append).
- **memory** — long-term memory files that persist across sessions
  (list / read / write).

## Usage

```sh
uv sync
uv run python main.py [--api-key ...] [--workdir ./workspace] [--memory-dir ./memory]
```

Type questions at the `>` prompt; `exit` or Ctrl-D to quit. Tool calls are printed
as a `[tool]` trace line so you can watch the ReAct loop work.

## Providers

The agent runs on Anthropic (default) or Google's Gemini API:

```sh
uv run python main.py --provider anthropic   # ANTHROPIC_API_KEY or --api-key
uv run python main.py --provider gemini      # GEMINI_API_KEY or --api-key
uv run python main.py --provider gemini --model gemini-2.5-pro
```

The API key comes from `--api-key` or the provider's environment variable.
`--model` overrides the provider's default model (`claude-sonnet-4-6` /
`gemini-2.5-flash`). The ReAct loop and all tool definitions are
provider-agnostic; the Anthropic/Gemini differences live in the adapter layer
in `providers.py`.

## Long-term memory

Memory lives as markdown files in `./memory/` (gitignored — it is
user-specific). Each file starts with a one-line summary the agent sees when it
lists memory, so it can read only the relevant files:

```markdown
Summary: One line describing this memory.

The full markdown body.
```

To add a memory by hand, drop a `<name>.md` file in that format into `memory/`
— or just ask the agent to remember something.

## Evals

A 10-task eval suite (easy/medium/hard) lives in `evals/tasks.jsonl`. Each task
runs on a fresh, isolated agent (temp workspace + temp memory dir) and is
graded deterministically (exact / contains / numeric-tolerance, plus
expected-tool and file checks):

```sh
uv run python evals/run_evals.py                    # full suite
uv run python evals/run_evals.py --task calc-basic  # one task by id
uv run python evals/run_evals.py --jobs 4           # run tasks in parallel
uv run python evals/run_evals.py --provider gemini  # against Gemini
```

A summary is printed to stdout and the full record (answers, tool traces,
timings) is written to `evals/results/<timestamp>.json` (gitignored). The
process exits 0 only if every selected task passed. Some tasks need live
Wikipedia access.

To add a task, append a JSON line to `evals/tasks.jsonl` with `id`,
`difficulty`, `prompt`, `expected_tools`, `expected_output`, `grading`
(`{"method": "exact" | "contains" | "numeric", "expected": ..., "tolerance": ...}`),
and `grading_notes`; optional `setup_memory` seeds memory files before the run
and `file_checks` asserts on files afterwards. `uv run pytest test_evals.py`
validates the task file.

## Tests

```sh
uv run pytest
```

Covers the deterministic logic (tools, provider adapters, eval grading). The
network tool and the live loop are verified by manual smoke testing with a real
API key, and end-to-end behavior by the eval suite.
