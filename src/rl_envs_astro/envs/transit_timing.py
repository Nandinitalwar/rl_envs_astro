"""A compact optimal-timing environment with synthetic transit mechanics."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_envs_astro.constants import CLASSICAL_BODIES, ZODIAC_GLYPHS, ZODIAC_SIGNS
from rl_envs_astro.envs._shared import circular_distances, option_signs


class TransitTimingEnv(gym.Env):
    """Resolve tasks near their target signs before the episode horizon.

    Action 0 waits. Action ``i + 1`` resolves task ``i`` at its current timing
    score. Every body advances by its configured speed after each action.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        n_tasks: int = 3,
        horizon: int = 24,
        transit_speeds: Sequence[int] = (1, 5, 7, 11, 1, 5, 7),
        wait_cost: float = 0.02,
        repeat_action_penalty: float = 0.25,
        missed_task_penalty: float = 0.5,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not 1 <= n_tasks <= len(CLASSICAL_BODIES):
            raise ValueError("n_tasks must be between 1 and 7")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(transit_speeds) < n_tasks:
            raise ValueError("transit_speeds must provide at least n_tasks values")
        speeds = np.asarray(transit_speeds[:n_tasks])
        if not np.issubdtype(speeds.dtype, np.integer) or np.any(speeds <= 0):
            raise ValueError("transit_speeds must contain positive integers")
        if wait_cost < 0 or repeat_action_penalty < 0 or missed_task_penalty < 0:
            raise ValueError("costs and penalties cannot be negative")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")

        self.n_tasks = int(n_tasks)
        self.horizon = int(horizon)
        self.transit_speeds = speeds.astype(np.int64, copy=True)
        self.wait_cost = float(wait_cost)
        self.repeat_action_penalty = float(repeat_action_penalty)
        self.missed_task_penalty = float(missed_task_penalty)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.n_tasks + 1)
        self.observation_space = spaces.Dict(
            {
                "positions": spaces.MultiDiscrete([12] * self.n_tasks),
                "targets": spaces.MultiDiscrete([12] * self.n_tasks),
                "resolved": spaces.MultiBinary(self.n_tasks),
                "days_remaining": spaces.Discrete(self.horizon + 1),
            }
        )

        self._positions = np.zeros(self.n_tasks, dtype=np.int64)
        self._targets = np.zeros(self.n_tasks, dtype=np.int64)
        self._resolved = np.zeros(self.n_tasks, dtype=np.int8)
        self._elapsed_days = 0

    @property
    def action_meanings(self) -> Tuple[str, ...]:
        return ("wait",) + tuple(
            "resolve %s task" % body for body in CLASSICAL_BODIES[: self.n_tasks]
        )

    def _observation(self) -> Dict[str, Any]:
        return {
            "positions": self._positions.copy(),
            "targets": self._targets.copy(),
            "resolved": self._resolved.copy(),
            "days_remaining": self.horizon - self._elapsed_days,
        }

    def _info(self) -> Dict[str, Any]:
        distances = circular_distances(self._positions, self._targets)
        return {
            "timing_distances": distances.astype(np.int64, copy=True),
            "resolved_tasks": int(self._resolved.sum()),
            "unresolved_tasks": int(self.n_tasks - self._resolved.sum()),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        positions = option_signs(options, "positions", self.n_tasks)
        targets = option_signs(options, "targets", self.n_tasks)
        self._positions = (
            positions
            if positions is not None
            else self.np_random.integers(0, 12, size=self.n_tasks, dtype=np.int64)
        )
        self._targets = (
            targets
            if targets is not None
            else self.np_random.integers(0, 12, size=self.n_tasks, dtype=np.int64)
        )
        self._resolved.fill(0)
        self._elapsed_days = 0
        observation, info = self._observation(), self._info()
        if self.render_mode == "human":
            self._render_human()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(
                "action must be an integer in [0, %d]" % (self.action_space.n - 1)
            )

        action_value = int(action)
        timing_score = 0.0
        wait_component = 0.0
        repeat_component = 0.0
        if action_value == 0:
            wait_component = -self.wait_cost
        else:
            task_index = action_value - 1
            if self._resolved[task_index]:
                repeat_component = -self.repeat_action_penalty
            else:
                distance = int(
                    circular_distances(
                        self._positions[task_index : task_index + 1],
                        self._targets[task_index : task_index + 1],
                    )[0]
                )
                timing_score = 1.0 - distance / 6.0
                self._resolved[task_index] = 1

        self._elapsed_days += 1
        self._positions = (self._positions + self.transit_speeds) % 12

        terminated = bool(np.all(self._resolved))
        truncated = self._elapsed_days >= self.horizon and not terminated
        missed_component = 0.0
        if truncated:
            missed_component = -self.missed_task_penalty * int(
                self.n_tasks - self._resolved.sum()
            )
        reward = timing_score + wait_component + repeat_component + missed_component

        info = self._info()
        info.update(
            {
                "timing_score": timing_score,
                "reward_components": {
                    "timing_score": timing_score,
                    "wait_cost": wait_component,
                    "repeat_action_penalty": repeat_component,
                    "missed_task_penalty": missed_component,
                },
            }
        )
        if self.render_mode == "human":
            self._render_human()
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        lines = ["Transit Timing — day %d/%d" % (self._elapsed_days, self.horizon)]
        distances = circular_distances(self._positions, self._targets)
        for index, body in enumerate(CLASSICAL_BODIES[: self.n_tasks]):
            position = int(self._positions[index])
            target = int(self._targets[index])
            status = "resolved" if self._resolved[index] else "open"
            lines.append(
                "% -8s %s %-11s target=%s %-11s distance=%d  %s"
                % (
                    body,
                    ZODIAC_GLYPHS[position],
                    ZODIAC_SIGNS[position],
                    ZODIAC_GLYPHS[target],
                    ZODIAC_SIGNS[target],
                    int(distances[index]),
                    status,
                )
            )
        output = "\n".join(lines)
        if self.render_mode == "human":
            return None
        return output

    def _render_human(self) -> None:
        original_mode = self.render_mode
        self.render_mode = "ansi"
        try:
            print(self.render())
        finally:
            self.render_mode = original_mode
