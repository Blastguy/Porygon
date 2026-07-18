"""Tool registry for the porygon ReAct agent.

Exposes ``TOOL_SCHEMAS`` (the JSON schemas handed to the Anthropic API) and
``dispatch`` (runs a tool by name and returns ``(result_text, is_error)``).

Three tools are provided:
  - ``calculator``        evaluate an arithmetic expression (safe AST walk, no eval)
  - ``wikipedia_search``  fetch the intro summary of the best-matching Wikipedia page
  - ``file_io``           read / write / list files inside a sandboxed working dir
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Tool schemas (passed verbatim to client.messages.create(tools=...))
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the numeric result. "
            "Supports + - * / // % ** and parentheses over numbers only. "
            "Use this whenever the user needs an exact calculation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression, e.g. '2 ** 10 + 5 * 3'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for a topic and return the introductory summary of the "
            "best-matching article. Use this for factual/encyclopedic questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up, e.g. 'ReAct prompting' or 'Tokyo'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "file_io",
        "description": (
            "Read, write, or list files in the agent's sandboxed working directory. "
            "All paths are relative to that directory; paths escaping it are rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "The file operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the working directory. For 'list', use '.' "
                        "for the working directory root."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Text to write (required for 'write', ignored otherwise).",
                },
            },
            "required": ["operation", "path"],
        },
    },
]


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate ``expression`` safely; return the result or an error message."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except SyntaxError:
        raise ValueError(f"could not parse expression: {expression!r}")
    return str(result)


# ---------------------------------------------------------------------------
# wikipedia_search
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "porygon-react-agent/0.1 (educational project)"}
_WIKI_TIMEOUT = 15


def wikipedia_search(query: str) -> str:
    """Return the intro summary of the best-matching Wikipedia article."""
    search = requests.get(
        _WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers=_WIKI_HEADERS,
        timeout=_WIKI_TIMEOUT,
    )
    search.raise_for_status()
    hits = search.json().get("query", {}).get("search", [])
    if not hits:
        return f"No Wikipedia article found for {query!r}."
    title = hits[0]["title"]

    extract = requests.get(
        _WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        headers=_WIKI_HEADERS,
        timeout=_WIKI_TIMEOUT,
    )
    extract.raise_for_status()
    pages = extract.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    summary = (page.get("extract") or "").strip()
    if not summary:
        return f"Found article '{title}' but it has no summary text."
    return f"{title}\n\n{summary}"


# ---------------------------------------------------------------------------
# file_io
# ---------------------------------------------------------------------------


def _resolve_in_sandbox(path: str, workdir: Path) -> Path:
    """Resolve ``path`` under ``workdir``; raise if it escapes the sandbox."""
    root = workdir.resolve()
    target = (root / path).resolve()
    if target != root and not target.is_relative_to(root):
        raise ValueError(f"path {path!r} escapes the working directory")
    return target


def file_io(operation: str, path: str, content: str | None, workdir: Path) -> str:
    """Perform a sandboxed read / write / list operation."""
    target = _resolve_in_sandbox(path, workdir)

    if operation == "read":
        if not target.is_file():
            raise ValueError(f"no such file: {path!r}")
        return target.read_text()

    if operation == "write":
        if content is None:
            raise ValueError("'content' is required for a write operation")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} characters to {path!r}."

    if operation == "list":
        if not target.is_dir():
            raise ValueError(f"not a directory: {path!r}")
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        if not entries:
            return f"{path!r} is empty."
        return "\n".join(entries)

    raise ValueError(f"unknown operation: {operation!r}")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def dispatch(name: str, tool_input: dict, workdir: Path) -> tuple[str, bool]:
    """Run tool ``name`` with ``tool_input``.

    Returns ``(result_text, is_error)``. Any handler exception is caught and
    returned as an error string so the agent can recover instead of crashing.
    """
    try:
        if name == "calculator":
            return calculator(tool_input["expression"]), False
        if name == "wikipedia_search":
            return wikipedia_search(tool_input["query"]), False
        if name == "file_io":
            return (
                file_io(
                    tool_input["operation"],
                    tool_input["path"],
                    tool_input.get("content"),
                    workdir,
                ),
                False,
            )
        return f"Error: unknown tool {name!r}.", True
    except Exception as exc:  # noqa: BLE001 - surface any failure to the model
        return f"Error: {exc}", True
