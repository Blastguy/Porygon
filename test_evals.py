"""Unit tests for the eval suite (offline — task file validation, grading,
and the agent's trace capture). Live eval runs happen via evals/run_evals.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "evals"))

import run_evals  # noqa: E402
from agent import Agent  # noqa: E402

TASKS_FILE = Path(__file__).resolve().parent / "evals" / "tasks.jsonl"


# --- task file -------------------------------------------------------------


def test_load_tasks_shape():
    tasks = run_evals.load_tasks(TASKS_FILE)
    assert len(tasks) == 10
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == 10
    for task in tasks:
        assert task["difficulty"] in {"easy", "medium", "hard"}
        assert task["prompt"]
        assert isinstance(task["expected_tools"], list) and task["expected_tools"]
        assert task["expected_output"]
        assert task["grading"]["method"] in {"exact", "contains", "numeric"}
        assert task["grading_notes"]


def test_tasks_cover_all_difficulties():
    tasks = run_evals.load_tasks(TASKS_FILE)
    assert {t["difficulty"] for t in tasks} == {"easy", "medium", "hard"}


# --- number extraction -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 2467.", [2467.0]),
        ("2467.0", [2467.0]),
        ("It's -3.5 degrees", [-3.5]),
        ("65,536 bytes", [65536.0]),
        ("1989 - 1912 = 77", [1989.0, 1912.0, 77.0]),
        ("no numbers here", []),
    ],
)
def test_numbers_in(text, expected):
    assert run_evals._numbers_in(text) == expected


# --- grading ---------------------------------------------------------------


def _task(**overrides):
    task = {
        "id": "t",
        "expected_tools": ["calculator"],
        "grading": {"method": "contains", "expected": "yes"},
    }
    task.update(overrides)
    return task


TRACE = [{"tool": "calculator", "input": {}, "result": "2", "is_error": False}]


def test_grade_contains_pass_and_fail(tmp_path: Path):
    task = _task()
    assert run_evals.grade(task, "well YES indeed", TRACE, tmp_path, tmp_path)["passed"]
    result = run_evals.grade(task, "no", TRACE, tmp_path, tmp_path)
    assert not result["passed"]
    assert any("output" in f for f in result["failures"])


def test_grade_exact(tmp_path: Path):
    task = _task(grading={"method": "exact", "expected": "42"})
    assert run_evals.grade(task, "  42\n", TRACE, tmp_path, tmp_path)["passed"]
    assert not run_evals.grade(task, "The answer is 42", TRACE, tmp_path, tmp_path)["passed"]


def test_grade_numeric_tolerance(tmp_path: Path):
    task = _task(grading={"method": "numeric", "expected": 2467, "tolerance": 0.001})
    assert run_evals.grade(task, "The answer is 2467.0", TRACE, tmp_path, tmp_path)["passed"]
    assert not run_evals.grade(task, "about 2468", TRACE, tmp_path, tmp_path)["passed"]
    assert not run_evals.grade(task, "no idea", TRACE, tmp_path, tmp_path)["passed"]


def test_grade_missing_expected_tool_fails(tmp_path: Path):
    task = _task(expected_tools=["calculator", "wikipedia_search"])
    result = run_evals.grade(task, "yes", TRACE, tmp_path, tmp_path)
    assert not result["passed"]
    assert any("wikipedia_search" in f for f in result["failures"])


def test_grade_file_checks(tmp_path: Path):
    workdir = tmp_path / "work"
    memory_dir = tmp_path / "mem"
    workdir.mkdir()
    memory_dir.mkdir()
    (workdir / "out.txt").write_text("hello world")
    (memory_dir / "fact.md").write_text("Summary: s\n\n42\n")
    task = _task(
        file_checks=[
            {"location": "workspace", "path": "out.txt", "contains": "hello"},
            {"location": "memory", "path": "fact.md", "contains": "42"},
        ]
    )
    assert run_evals.grade(task, "yes", TRACE, workdir, memory_dir)["passed"]

    task = _task(
        file_checks=[{"location": "workspace", "path": "missing.txt", "contains": "x"}]
    )
    result = run_evals.grade(task, "yes", TRACE, workdir, memory_dir)
    assert not result["passed"]
    assert any("missing.txt" in f for f in result["failures"])


# --- agent trace capture ---------------------------------------------------


class _FakeProvider:
    """Scripted provider: one tool round, then a final answer."""

    def __init__(self):
        import providers

        self.steps = [
            providers.StepResult(
                "tool_use",
                "",
                [providers.ToolCall("t1", "calculator", {"expression": "1+1"})],
            ),
            providers.StepResult("done", "2", []),
        ]

    def add_user_message(self, text):
        pass

    def step(self):
        return self.steps.pop(0)

    def add_tool_results(self, results):
        pass


def test_agent_records_trace(tmp_path: Path):
    agent = Agent(_FakeProvider(), tmp_path / "work", tmp_path / "mem")
    answer = agent.run_turn("what is 1+1?")
    assert answer == "2"
    assert agent.trace == [
        {"tool": "calculator", "input": {"expression": "1+1"}, "result": "2", "is_error": False}
    ]
