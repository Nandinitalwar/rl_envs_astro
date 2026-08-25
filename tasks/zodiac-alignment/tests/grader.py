"""Harbor verifier for the Zodiac Alignment task."""

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
    initial = list(scenario["positions"])
    targets = scenario["targets"]
    initial_distance = sum(circular_distance(a, b) for a, b in zip(initial, targets))
    if not isinstance(answer, dict) or set(answer) != {"actions"}:
        raise ValueError('answer must contain only the key "actions"')
    actions = answer["actions"]
    if not isinstance(actions, list) or len(actions) > scenario["max_steps"]:
        raise ValueError("actions must be a list no longer than max_steps")

    positions = initial.copy()
    success = False
    executed = 0
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("all actions must be integers")
        if not 0 <= action < 2 * len(positions):
            raise ValueError(f"illegal action: {action}")
        body = action // 2
        positions[body] = (positions[body] + (-1 if action % 2 == 0 else 1)) % 12
        executed += 1
        if positions == targets:
            success = True
            break

    final_distance = sum(circular_distance(a, b) for a, b in zip(positions, targets))
    progress = max(0.0, 1.0 - final_distance / initial_distance)
    efficiency = 0.0
    if success:
        efficiency = min(1.0, scenario["optimal_steps"] / max(executed, 1))
        reward = 0.5 + 0.5 * efficiency
    else:
        reward = 0.5 * progress
    metrics = {
        "reward": round(reward, 6),
        "success": float(success),
        "distance_progress": round(progress, 6),
        "efficiency": round(efficiency, 6),
    }
    feedback = {
        "executed_actions": executed,
        "final_distance": final_distance,
        "final_positions": positions,
        "initial_distance": initial_distance,
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
            "distance_progress": 0.0,
            "efficiency": 0.0,
        }
        feedback = {"error": str(error), "metrics": metrics}
    (LOG_DIR / "reward.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    (LOG_DIR / "feedback.json").write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
