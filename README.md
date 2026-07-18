# porygon

A simple from-scratch ReAct agent built on the Anthropic API (Sonnet). It runs a
hand-written agentic loop — the model reasons, calls tools, and observes the
results until it can answer — exposed as an interactive REPL.

## Tools

- **calculator** — exact arithmetic via a safe AST evaluator (no `eval`).
- **wikipedia_search** — intro summary of the best-matching Wikipedia article.
- **file_io** — read / write / list files inside a sandboxed working directory.

## Usage

```sh
uv sync
uv run python main.py --api-key sk-ant-... [--workdir ./workspace]
```

Type questions at the `>` prompt; `exit` or Ctrl-D to quit. Tool calls are printed
as a `[tool]` trace line so you can watch the ReAct loop work.

## Tests

```sh
uv run pytest
```

Covers the deterministic logic (calculator, file_io sandbox, dispatch). The
network tool and the live loop are verified by manual smoke testing with a real
API key.
