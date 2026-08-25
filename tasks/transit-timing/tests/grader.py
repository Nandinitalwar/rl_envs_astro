"""Harbor verifier for the Transit Timing task."""

import json
from pathlib import Path
from typing import Any

ANSWER_PATH = Path("/app/answer.json")
SCENARIO_PATH = Path("/app/scenario.json")
LOG_DIR = Path("/logs/verifier")


def circular_distance(left: int, right: int) -> int:
    direct = abs(left - right)
    return min(direct, 12 - direct)


def grade(scenario: dict[str, Any], answer: Any) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(answer, dict) or set(answer) != {"actions"}:
        raise ValueError('answer must contain only the key "actions"')
    actions = answer["actions"]
    if not isinstance(actions, list) or len(actions) > scenario["horizon"]:
        raise ValueError("actions must be a list no longer than the horizon")

    positions = list(scenario["positions"])
    targets = scenario["targets"]
    resolved = [False] * len(positions)
    timing_scores: list[float] = []
    raw_return = 0.0
    executed = 0
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("all actions must be integers")
        if not 0 <= action <= len(positions):
            raise ValueError(f"illegal action: {action}")
        if action == 0:
            raw_return -= scenario["wait_cost"]
        else:
            task = action - 1
            if resolved[task]:
                raw_return -= scenario["repeat_action_penalty"]
            else:
                score = 1.0 - circular_distance(positions[task], targets[task]) / 6.0
                timing_scores.append(score)
                raw_return += score
                resolved[task] = True
        executed += 1
        positions = [
            (position + speed) % 12
            for position, speed in zip(positions, scenario["speeds"])
        ]
        if all(resolved):
            break

    success = all(resolved)
    if not success and executed >= scenario["horizon"]:
        raw_return -= scenario["missed_task_penalty"] * resolved.count(False)
    return_ratio = max(0.0, min(1.0, raw_return / scenario["optimal_return"]))
    reward = return_ratio if success else min(0.5, return_ratio)
    timing_quality = sum(timing_scores) / len(targets)
    metrics = {
        "reward": round(reward, 6),
        "success": float(success),
        "return_ratio": round(return_ratio, 6),
        "timing_quality": round(timing_quality, 6),
    }
    feedback = {
        "executed_actions": executed,
        "raw_return": round(raw_return, 6),
        "resolved": resolved,
        "timing_scores": timing_scores,
        "metrics": metrics,
    }
    return metrics, feedback


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        metrics, feedback = grade(
            json.loads(SCENARIO_PATH.read_text()),
            json.loads(ANSWER_PATH.read_text()),
        )
    except Exception as error:
        metrics = {
            "reward": 0.0,
            "success": 0.0,
            "return_ratio": 0.0,
            "timing_quality": 0.0,
        }
        feedback = {"error": str(error), "metrics": metrics}
    (LOG_DIR / "reward.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    (LOG_DIR / "feedback.json").write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
