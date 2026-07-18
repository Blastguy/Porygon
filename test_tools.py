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


# --- scratchpad ------------------------------------------------------------


def _ctx(tmp_path: Path) -> tools.ToolContext:
    return tools.ToolContext(tmp_path / "workspace", tmp_path / "memory")


def test_scratchpad_starts_empty(tmp_path: Path):
    ctx = _ctx(tmp_path)
    assert "empty" in tools.scratchpad("read", None, ctx)


def test_scratchpad_write_read_roundtrip(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tools.scratchpad("write", "step 1: found the year 1889", ctx)
    assert tools.scratchpad("read", None, ctx) == "step 1: found the year 1889"


def test_scratchpad_write_replaces(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tools.scratchpad("write", "old", ctx)
    tools.scratchpad("write", "new", ctx)
    assert tools.scratchpad("read", None, ctx) == "new"


def test_scratchpad_append_accumulates_lines(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tools.scratchpad("append", "144", ctx)
    tools.scratchpad("append", "169", ctx)
    assert tools.scratchpad("read", None, ctx) == "144\n169"


def test_scratchpad_write_requires_content(tmp_path: Path):
    with pytest.raises(ValueError):
        tools.scratchpad("write", None, _ctx(tmp_path))


def test_scratchpad_unknown_operation(tmp_path: Path):
    with pytest.raises(ValueError):
        tools.scratchpad("erase", None, _ctx(tmp_path))


# --- memory ----------------------------------------------------------------


def _mem_dir(tmp_path: Path) -> Path:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


def test_memory_write_creates_summary_file(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    tools.memory("write", None, "project", "Project codename", "Codename BLUEBIRD.", memory_dir)
    text = (memory_dir / "project.md").read_text()
    assert text.startswith("Summary: Project codename\n")
    assert "Codename BLUEBIRD." in text


def test_memory_write_accepts_md_suffix(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    tools.memory("write", None, "notes.md", "Some notes", "body", memory_dir)
    assert (memory_dir / "notes.md").exists()
    assert not (memory_dir / "notes.md.md").exists()


def test_memory_write_overwrites_as_update(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    tools.memory("write", None, "list", "Shopping list", "apples, bread", memory_dir)
    tools.memory("write", None, "list", "Shopping list", "apples, bread, milk", memory_dir)
    text = (memory_dir / "list.md").read_text()
    assert "milk" in text
    assert text.count("Summary:") == 1


def test_memory_write_requires_fields(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    with pytest.raises(ValueError):
        tools.memory("write", None, None, "summary", "body", memory_dir)
    with pytest.raises(ValueError):
        tools.memory("write", None, "name", None, "body", memory_dir)
    with pytest.raises(ValueError):
        tools.memory("write", None, "name", "summary", None, memory_dir)


def test_memory_list_shows_names_and_summaries(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    tools.memory("write", None, "beta", "Second fact", "b", memory_dir)
    tools.memory("write", None, "alpha", "First fact", "a", memory_dir)
    listing = tools.memory("list", None, None, None, None, memory_dir)
    lines = listing.splitlines()
    assert lines[0].startswith("alpha") and "First fact" in lines[0]
    assert lines[1].startswith("beta") and "Second fact" in lines[1]


def test_memory_list_empty(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    assert "No memory files" in tools.memory("list", None, None, None, None, memory_dir)


def test_memory_read_single_and_multiple(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    tools.memory("write", None, "a", "A", "alpha body", memory_dir)
    tools.memory("write", None, "b", "B", "beta body", memory_dir)
    single = tools.memory("read", ["a"], None, None, None, memory_dir)
    assert "alpha body" in single
    both = tools.memory("read", ["a", "b"], None, None, None, memory_dir)
    assert "alpha body" in both and "beta body" in both
    assert "## a" in both and "## b" in both


def test_memory_read_missing_raises(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    with pytest.raises(ValueError):
        tools.memory("read", ["nope"], None, None, None, memory_dir)


def test_memory_read_requires_names(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    with pytest.raises(ValueError):
        tools.memory("read", None, None, None, None, memory_dir)


@pytest.mark.parametrize("name", ["../escape", "../../etc/passwd", "/etc/passwd"])
def test_memory_rejects_traversal(tmp_path: Path, name):
    memory_dir = _mem_dir(tmp_path)
    with pytest.raises(ValueError):
        tools.memory("write", None, name, "s", "b", memory_dir)
    with pytest.raises(ValueError):
        tools.memory("read", [name], None, None, None, memory_dir)


def test_memory_unknown_operation(tmp_path: Path):
    memory_dir = _mem_dir(tmp_path)
    with pytest.raises(ValueError):
        tools.memory("delete", None, None, None, None, memory_dir)


# --- dispatch --------------------------------------------------------------


def test_dispatch_unknown_tool(tmp_path: Path):
    result, is_error = tools.dispatch("nope", {}, _ctx(tmp_path))
    assert is_error
    assert "unknown tool" in result


def test_dispatch_calculator_ok(tmp_path: Path):
    result, is_error = tools.dispatch(
        "calculator", {"expression": "1 + 1"}, _ctx(tmp_path)
    )
    assert (result, is_error) == ("2", False)


def test_dispatch_catches_handler_error(tmp_path: Path):
    # Bad expression -> handler raises -> dispatch returns an error result.
    result, is_error = tools.dispatch(
        "calculator", {"expression": "1 +"}, _ctx(tmp_path)
    )
    assert is_error
    assert result.startswith("Error:")


def test_dispatch_traversal_is_error(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ctx.workdir.mkdir(parents=True)
    result, is_error = tools.dispatch(
        "file_io", {"operation": "read", "path": "../x"}, ctx
    )
    assert is_error
    assert "escapes" in result


def test_dispatch_scratchpad_roundtrip(tmp_path: Path):
    ctx = _ctx(tmp_path)
    result, is_error = tools.dispatch(
        "scratchpad", {"operation": "write", "content": "note"}, ctx
    )
    assert not is_error
    result, is_error = tools.dispatch("scratchpad", {"operation": "read"}, ctx)
    assert (result, is_error) == ("note", False)


def test_dispatch_scratchpad_error(tmp_path: Path):
    result, is_error = tools.dispatch(
        "scratchpad", {"operation": "write"}, _ctx(tmp_path)
    )
    assert is_error
    assert result.startswith("Error:")


def test_dispatch_memory_roundtrip(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ctx.memory_dir.mkdir(parents=True)
    result, is_error = tools.dispatch(
        "memory",
        {"operation": "write", "name": "fact", "summary": "A fact", "content": "42"},
        ctx,
    )
    assert not is_error
    result, is_error = tools.dispatch("memory", {"operation": "read", "names": ["fact"]}, ctx)
    assert not is_error
    assert "42" in result


def test_dispatch_memory_error(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ctx.memory_dir.mkdir(parents=True)
    result, is_error = tools.dispatch(
        "memory", {"operation": "read", "names": ["missing"]}, ctx
    )
    assert is_error
    assert result.startswith("Error:")
