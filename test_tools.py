"""Unit tests for the deterministic tool logic (calculator, file_io, dispatch).

The network tool (wikipedia_search) and the live agent loop are covered by a
manual smoke test, not mocked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools


# --- calculator ------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 ** 10 + 5 * 3", "1039"),
        ("(1 + 2) * 3", "9"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-5 + 2", "-3"),
        ("10 / 4", "2.5"),
    ],
)
def test_calculator_valid(expr, expected):
    assert tools.calculator(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "os.getcwd()",
        "x + 1",  # bare name
        "abs(-1)",  # function call
        "(1).__class__",  # attribute access
        "1 + ",  # syntax error
        "True + 1",  # bool is not a number
    ],
)
def test_calculator_rejects_injection(expr):
    with pytest.raises(ValueError):
        tools.calculator(expr)


# --- file_io sandbox -------------------------------------------------------


def test_file_io_write_read_list(tmp_path: Path):
    assert "Wrote" in tools.file_io("write", "notes.txt", "hello", tmp_path)
    assert tools.file_io("read", "notes.txt", None, tmp_path) == "hello"
    assert "notes.txt" in tools.file_io("list", ".", None, tmp_path)


def test_file_io_write_creates_parent_dirs(tmp_path: Path):
    tools.file_io("write", "sub/dir/a.txt", "x", tmp_path)
    assert (tmp_path / "sub" / "dir" / "a.txt").read_text() == "x"


def test_file_io_read_missing(tmp_path: Path):
    with pytest.raises(ValueError):
        tools.file_io("read", "nope.txt", None, tmp_path)


def test_file_io_write_requires_content(tmp_path: Path):
    with pytest.raises(ValueError):
        tools.file_io("write", "a.txt", None, tmp_path)


@pytest.mark.parametrize("path", ["../escape.txt", "../../etc/passwd", "/etc/passwd"])
def test_file_io_rejects_traversal(tmp_path: Path, path):
    with pytest.raises(ValueError):
        tools.file_io("write", path, "x", tmp_path)
    # Nothing should have been written outside the sandbox.
    assert not (tmp_path.parent / "escape.txt").exists()


# --- dispatch --------------------------------------------------------------


def test_dispatch_unknown_tool(tmp_path: Path):
    result, is_error = tools.dispatch("nope", {}, tmp_path)
    assert is_error
    assert "unknown tool" in result


def test_dispatch_calculator_ok(tmp_path: Path):
    result, is_error = tools.dispatch("calculator", {"expression": "1 + 1"}, tmp_path)
    assert (result, is_error) == ("2", False)


def test_dispatch_catches_handler_error(tmp_path: Path):
    # Bad expression -> handler raises -> dispatch returns an error result.
    result, is_error = tools.dispatch("calculator", {"expression": "1 +"}, tmp_path)
    assert is_error
    assert result.startswith("Error:")


def test_dispatch_traversal_is_error(tmp_path: Path):
    result, is_error = tools.dispatch(
        "file_io", {"operation": "read", "path": "../x"}, tmp_path
    )
    assert is_error
    assert "escapes" in result
