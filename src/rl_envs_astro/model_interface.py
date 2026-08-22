"""A small JSON protocol for language-model policies.

The environments themselves remain ordinary Gymnasium environments.  This
module only turns a state into a self-contained model prompt and validates the
single action returned by a model.
"""

import json
from typing import Any, Dict, Mapping, Optional

import gymnasium as gym
import numpy as np

from rl_envs_astro.envs import TransitTimingEnv, ZodiacAlignmentEnv


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _environment_description(env: gym.Env) -> Dict[str, Any]:
    base_env = env.unwrapped
    if isinstance(base_env, ZodiacAlignmentEnv):
        return {
            "name": "ZodiacAlignment",
            "objective": (
                "Move every body to its target on a cyclic 12-sign ring in as "
                "few steps as possible. Each move changes one position by one."
            ),
            "dynamics": {
                "sign_indices": "0 through 11; movement wraps modulo 12",
                "action_encoding": "2 * body_index + direction",
                "direction_values": {
                    "even_action": "counterclockwise (-1)",
                    "odd_action": "clockwise (+1)",
                },
            },
            "config": {
                "n_bodies": base_env.n_bodies,
                "max_steps": base_env.max_steps,
                "step_cost": base_env.step_cost,
                "success_bonus": base_env.success_bonus,
            },
        }
    if isinstance(base_env, TransitTimingEnv):
        return {
            "name": "TransitTiming",
            "objective": (
                "Resolve every task as near its target sign as possible before "
                "the horizon. Only one action can be taken per day."
            ),
            "dynamics": {
                "sign_indices": "0 through 11; positions wrap modulo 12",
                "wait_action": 0,
                "resolve_action": "action i + 1 resolves task i",
                "timing_score": "1 - circular_distance / 6",
                "daily_update": (
                    "After each action, add the corresponding transit speed to "
                    "every position modulo 12."
                ),
            },
            "config": {
                "n_tasks": base_env.n_tasks,
                "horizon": base_env.horizon,
                "transit_speeds": base_env.transit_speeds.tolist(),
                "wait_cost": base_env.wait_cost,
                "repeat_action_penalty": base_env.repeat_action_penalty,
                "missed_task_penalty": base_env.missed_task_penalty,
            },
        }
    raise TypeError("unsupported environment type: %s" % type(base_env).__name__)


def build_model_prompt(
    env: gym.Env,
    observation: Mapping[str, Any],
    *,
    previous_reward: Optional[float] = None,
    previous_info: Optional[Mapping[str, Any]] = None,
) -> str:
    """Serialize one decision turn into a self-contained JSON prompt."""

    base_env = env.unwrapped
    action_meanings = getattr(base_env, "action_meanings", None)
    if action_meanings is None:
        raise TypeError("environment does not expose action_meanings")

    payload: Dict[str, Any] = {
        "protocol": "rl-envs-astro/model-action-v1",
        "instruction": (
            "Choose exactly one legal action. Respond only with a JSON object "
            "of the form {\"action\": integer}."
        ),
        "environment": _environment_description(env),
        "actions": [
            {"action": index, "meaning": meaning}
            for index, meaning in enumerate(action_meanings)
        ],
        "observation": _json_value(observation),
    }
    if previous_reward is not None:
        payload["previous_transition"] = {
            "reward": float(previous_reward),
            "info": _json_value(previous_info or {}),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def parse_model_action(env: gym.Env, response: str) -> int:
    """Parse and validate a model's ``{"action": integer}`` response."""

    candidate = response.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0] in ("```", "```json"):
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("model response must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        raise ValueError('model response must contain only the key "action"')
    action = payload["action"]
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError("action must be an integer")
    if not env.action_space.contains(action):
        raise ValueError("action %d is outside the environment action space" % action)
    return action
