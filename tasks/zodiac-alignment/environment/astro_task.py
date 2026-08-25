#!/usr/bin/env python3
"""Inspect or evaluate a candidate Zodiac Alignment trajectory."""

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIO_PATH = Path("/app/scenario.json")


def circular_distance(left: int, right: int) -> int:
    direct = abs(left - right)
    return min(direct, 12 - direct)


def evaluate(scenario: dict[str, Any], actions: Any) -> dict[str, Any]:
    if not isinstance(actions, list):
        raise ValueError("actions must be a JSON list")
    if len(actions) > scenario["max_steps"]:
        raise ValueError("trajectory exceeds max_steps")

    positions = list(scenario["positions"])
    targets = scenario["targets"]
    total_reward = 0.0
    terminated = False
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("every action must be an integer")
        if not 0 <= action < 2 * len(positions):
            raise ValueError(f"illegal action: {action}")
        previous = sum(circular_distance(a, b) for a, b in zip(positions, targets))
        body = action // 2
        delta = -1 if action % 2 == 0 else 1
        positions[body] = (positions[body] + delta) % 12
        distance = sum(circular_distance(a, b) for a, b in zip(positions, targets))
        total_reward += previous - distance - scenario["step_cost"]
        terminated = distance == 0
        if terminated:
            total_reward += scenario["success_bonus"]
            break

    final_distance = sum(circular_distance(a, b) for a, b in zip(positions, targets))
    return {
        "final_distance": final_distance,
        "positions": positions,
        "raw_reward": round(total_reward, 6),
        "steps": len(actions),
        "success": terminated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("observe", "evaluate"))
    parser.add_argument("actions", nargs="?")
    args = parser.parse_args()
    scenario = json.loads(SCENARIO_PATH.read_text())
    if args.command == "observe":
        print(json.dumps(scenario, indent=2, sort_keys=True))
        return
    if args.actions is None:
        parser.error("evaluate requires a JSON action list")
    try:
        result = evaluate(scenario, json.loads(args.actions))
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
