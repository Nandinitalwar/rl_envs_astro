import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).parents[1]
TASKS = ROOT / "tasks"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oracle(task: Path) -> dict[str, list[int]]:
    for line in (task / "solution" / "solve.sh").read_text().splitlines():
        if line.startswith('{"actions"'):
            return json.loads(line)
    raise AssertionError("oracle answer JSON not found")


def test_harbor_task_layout_and_executables():
    required = (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/scenario.json",
        "environment/astro_task.py",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/grader.py",
    )
    for task_name in ("zodiac-alignment", "transit-timing"):
        task = TASKS / task_name
        assert all((task / relative).is_file() for relative in required)
        assert os.access(task / "environment" / "astro_task.py", os.X_OK)
        assert os.access(task / "solution" / "solve.sh", os.X_OK)
        assert os.access(task / "tests" / "test.sh", os.X_OK)


def test_zodiac_harbor_verifier_oracle_and_nop():
    task = TASKS / "zodiac-alignment"
    grader = load_module("zodiac_grader", task / "tests" / "grader.py")
    scenario = json.loads((task / "environment" / "scenario.json").read_text())

    oracle_metrics, _ = grader.grade(scenario, load_oracle(task))
    nop_metrics, _ = grader.grade(scenario, {"actions": []})

    assert oracle_metrics == {
        "reward": 1.0,
        "success": 1.0,
        "distance_progress": 1.0,
        "efficiency": 1.0,
    }
    assert nop_metrics["reward"] == 0.0
    assert nop_metrics["success"] == 0.0


def test_transit_harbor_verifier_oracle_and_nop():
    task = TASKS / "transit-timing"
    grader = load_module("transit_grader", task / "tests" / "grader.py")
    scenario = json.loads((task / "environment" / "scenario.json").read_text())

    oracle_metrics, _ = grader.grade(scenario, load_oracle(task))
    nop_metrics, _ = grader.grade(scenario, {"actions": []})

    assert oracle_metrics == {
        "reward": 1.0,
        "success": 1.0,
        "return_ratio": 1.0,
        "timing_quality": 1.0,
    }
    assert nop_metrics["reward"] == 0.0
    assert nop_metrics["success"] == 0.0
