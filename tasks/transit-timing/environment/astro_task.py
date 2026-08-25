#!/usr/bin/env python3
"""Inspect or evaluate a candidate Transit Timing trajectory."""

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
    if len(actions) > scenario["horizon"]:
        raise ValueError("trajectory exceeds horizon")

    positions = list(scenario["positions"])
    targets = scenario["targets"]
    speeds = scenario["speeds"]
    resolved = [False] * len(positions)
    total_reward = 0.0
    timing_scores: list[float] = []
    terminated = False
    executed = 0
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("every action must be an integer")
        if not 0 <= action <= len(positions):
            raise ValueError(f"illegal action: {action}")
        if action == 0:
            total_reward -= scenario["wait_cost"]
        else:
            task = action - 1
            if resolved[task]:
                total_reward -= scenario["repeat_action_penalty"]
            else:
                score = 1.0 - circular_distance(positions[task], targets[task]) / 6.0
                timing_scores.append(score)
                total_reward += score
                resolved[task] = True
        executed += 1
        positions = [(position + speed) % 12 for position, speed in zip(positions, speeds)]
        terminated = all(resolved)
        if terminated:
            break
    if not terminated and executed >= scenario["horizon"]:
        total_reward -= scenario["missed_task_penalty"] * resolved.count(False)
    return {
        "positions": positions,
        "raw_reward": round(total_reward, 6),
        "resolved": resolved,
        "steps": executed,
        "success": terminated,
        "timing_scores": timing_scores,
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
