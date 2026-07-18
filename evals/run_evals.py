"""Eval runner for the porygon agent.

Runs each task in evals/tasks.jsonl against a fresh, fully isolated agent
(temporary workspace and memory dirs, fresh provider client), captures the
tool-call trace, grades the final answer deterministically, and writes a
results file to evals/results/<timestamp>.json.

Usage:
    uv run python evals/run_evals.py                    # full suite
    uv run python evals/run_evals.py --task calc-basic  # one task by id
    uv run python evals/run_evals.py --jobs 4 --provider gemini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import providers  # noqa: E402
import tools  # noqa: E402
from agent import SYSTEM_PROMPT, Agent  # noqa: E402
from main import API_KEY_ENV_VARS  # noqa: E402

TASKS_FILE = REPO_ROOT / "evals" / "tasks.jsonl"
RESULTS_DIR = REPO_ROOT / "evals" / "results"

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_tasks(path: Path) -> list[dict]:
    """Load one task per non-empty line of a JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numbers_in(text: str) -> list[float]:
    """Extract every number in ``text`` (commas stripped) as floats."""
    return [float(m) for m in _NUMBER_RE.findall(text.replace(",", ""))]


def grade(
    task: dict, answer: str, trace: list[dict], workdir: Path, memory_dir: Path
) -> dict:
    """Grade a final answer; returns ``{"passed": bool, "failures": [...]}``."""
    failures = []

    grading = task["grading"]
    expected = grading["expected"]
    method = grading["method"]
    if method == "exact":
        if answer.strip() != str(expected):
            failures.append(f"output: expected exactly {expected!r}, got {answer!r}")
    elif method == "contains":
        if str(expected).lower() not in answer.lower():
            failures.append(f"output: expected to contain {expected!r}, got {answer!r}")
    elif method == "numeric":
        tolerance = grading.get("tolerance", 0.001)
        if not any(abs(n - expected) <= tolerance for n in _numbers_in(answer)):
            failures.append(f"output: no number within {tolerance} of {expected} in {answer!r}")
    else:
        failures.append(f"grading: unknown method {method!r}")

    used_tools = {entry["tool"] for entry in trace}
    for missing in sorted(set(task["expected_tools"]) - used_tools):
        failures.append(f"tools: expected tool {missing!r} was never called")

    roots = {"workspace": workdir, "memory": memory_dir}
    for check in task.get("file_checks", []):
        target = roots[check["location"]] / check["path"]
        if not target.is_file():
            failures.append(f"files: {check['location']}/{check['path']} does not exist")
        elif check["contains"] not in target.read_text():
            failures.append(
                f"files: {check['location']}/{check['path']} missing {check['contains']!r}"
            )

    return {"passed": not failures, "failures": failures}


def run_task(task: dict, provider_name: str, model: str | None, api_key: str) -> dict:
    """Run one task on a fresh isolated agent and grade the result."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "workspace"
        memory_dir = Path(tmp) / "memory"
        workdir.mkdir()
        memory_dir.mkdir()
        for seed in task.get("setup_memory", []):
            tools.memory(
                "write", None, seed["name"], seed["summary"], seed["content"], memory_dir
            )

        provider = providers.build_provider(
            provider_name, api_key, model, SYSTEM_PROMPT, tools.TOOL_SCHEMAS
        )
        agent = Agent(provider, workdir, memory_dir)

        started = time.monotonic()
        try:
            answer = agent.run_turn(task["prompt"])
            result = grade(task, answer, agent.trace, workdir, memory_dir)
        except Exception as exc:  # noqa: BLE001 - one task crashing must not kill the run
            answer = ""
            result = {"passed": False, "failures": [f"exception: {exc}"]}

        return {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "passed": result["passed"],
            "failures": result["failures"],
            "answer": answer,
            "trace": agent.trace,
            "seconds": round(time.monotonic() - started, 2),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_evals", description="Run the porygon eval suite."
    )
    parser.add_argument("--task", help="Run only the task with this id.")
    parser.add_argument(
        "--tasks-file", default=TASKS_FILE, type=Path, help="Path to the tasks JSONL."
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "gemini"], default="anthropic"
    )
    parser.add_argument("--model", help="Model override (default: provider default).")
    parser.add_argument(
        "--api-key", help="API key; falls back to the provider's env var."
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Tasks to run in parallel (default: 1)."
    )
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get(API_KEY_ENV_VARS[args.provider])
    if not api_key:
        print(
            f"Error: no API key. Pass --api-key or set "
            f"{API_KEY_ENV_VARS[args.provider]}."
        )
        return 1

    tasks = load_tasks(args.tasks_file)
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"Error: no task with id {args.task!r}.")
            return 1

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(
            pool.map(
                lambda t: run_task(t, args.provider, args.model, api_key), tasks
            )
        )

    print()
    for record in records:
        status = "PASS" if record["passed"] else "FAIL"
        print(f"{status}  {record['id']} ({record['difficulty']}, {record['seconds']}s)")
        for failure in record["failures"]:
            print(f"      - {failure}")

    passed = sum(r["passed"] for r in records)
    print(f"\n{passed}/{len(records)} passed")
    for difficulty in ("easy", "medium", "hard"):
        bucket = [r for r in records if r["difficulty"] == difficulty]
        if bucket:
            print(f"  {difficulty}: {sum(r['passed'] for r in bucket)}/{len(bucket)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    results_file.write_text(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "passed": passed,
                "total": len(records),
                "tasks": records,
            },
            indent=2,
        )
    )
    print(f"\nResults written to {results_file.relative_to(REPO_ROOT)}")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
