"""A goal-conditioned control task on the zodiac ring."""

from typing import Any, Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_envs_astro.constants import CLASSICAL_BODIES, ZODIAC_GLYPHS, ZODIAC_SIGNS
from rl_envs_astro.envs._shared import circular_distances, option_signs


class ZodiacAlignmentEnv(gym.Env):
    """Move symbolic bodies clockwise/counterclockwise to target signs.

    Actions are encoded as ``2 * body_index + direction`` where direction 0
    moves counterclockwise and direction 1 moves clockwise.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        n_bodies: int = 7,
        max_steps: int = 64,
        step_cost: float = 0.01,
        success_bonus: float = 10.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not 1 <= n_bodies <= len(CLASSICAL_BODIES):
            raise ValueError("n_bodies must be between 1 and 7")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if step_cost < 0:
            raise ValueError("step_cost cannot be negative")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")

        self.n_bodies = int(n_bodies)
        self.max_steps = int(max_steps)
        self.step_cost = float(step_cost)
        self.success_bonus = float(success_bonus)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.n_bodies * 2)
        self.observation_space = spaces.Dict(
            {
                "positions": spaces.MultiDiscrete([12] * self.n_bodies),
                "targets": spaces.MultiDiscrete([12] * self.n_bodies),
                "steps_remaining": spaces.Discrete(self.max_steps + 1),
            }
        )

        self._positions = np.zeros(self.n_bodies, dtype=np.int64)
        self._targets = np.zeros(self.n_bodies, dtype=np.int64)
        self._elapsed_steps = 0

    @property
    def action_meanings(self) -> Tuple[str, ...]:
        meanings = []
        for body in CLASSICAL_BODIES[: self.n_bodies]:
            meanings.extend(("%s counterclockwise" % body, "%s clockwise" % body))
        return tuple(meanings)

    def _observation(self) -> Dict[str, Any]:
        return {
            "positions": self._positions.copy(),
            "targets": self._targets.copy(),
            "steps_remaining": self.max_steps - self._elapsed_steps,
        }

    def _info(self) -> Dict[str, Any]:
        distances = circular_distances(self._positions, self._targets)
        return {
            "total_distance": int(distances.sum()),
            "aligned_bodies": int(np.count_nonzero(distances == 0)),
            "is_success": bool(np.all(distances == 0)),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        positions = option_signs(options, "positions", self.n_bodies)
        targets = option_signs(options, "targets", self.n_bodies)
        self._positions = (
            positions
            if positions is not None
            else self.np_random.integers(0, 12, size=self.n_bodies, dtype=np.int64)
        )
        self._targets = (
            targets
            if targets is not None
            else self.np_random.integers(0, 12, size=self.n_bodies, dtype=np.int64)
        )

        # Avoid a randomly sampled episode that is already solved.
        if (
            positions is None
            and targets is None
            and np.array_equal(self._positions, self._targets)
        ):
            self._targets[0] = (self._targets[0] + 1) % 12

        self._elapsed_steps = 0
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

        previous_distance = int(circular_distances(self._positions, self._targets).sum())
        action_value = int(action)
        body_index = action_value // 2
        delta = -1 if action_value % 2 == 0 else 1
        self._positions[body_index] = (self._positions[body_index] + delta) % 12
        self._elapsed_steps += 1

        info = self._info()
        distance_gain = previous_distance - info["total_distance"]
        reward = float(distance_gain) - self.step_cost
        terminated = bool(info["is_success"])
        if terminated:
            reward += self.success_bonus
        truncated = self._elapsed_steps >= self.max_steps and not terminated

        info.update(
            {
                "moved_body": CLASSICAL_BODIES[body_index],
                "direction": "counterclockwise" if delta < 0 else "clockwise",
                "reward_components": {
                    "distance_gain": float(distance_gain),
                    "step_cost": -self.step_cost,
                    "success_bonus": self.success_bonus if terminated else 0.0,
                },
            }
        )
        if self.render_mode == "human":
            self._render_human()
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        lines = ["Zodiac Alignment — step %d/%d" % (self._elapsed_steps, self.max_steps)]
        distances = circular_distances(self._positions, self._targets)
        for index, body in enumerate(CLASSICAL_BODIES[: self.n_bodies]):
            position = int(self._positions[index])
            target = int(self._targets[index])
            lines.append(
                "% -8s %s %-11s → %s %-11s  distance=%d"
                % (
                    body,
                    ZODIAC_GLYPHS[position],
                    ZODIAC_SIGNS[position],
                    ZODIAC_GLYPHS[target],
                    ZODIAC_SIGNS[target],
                    int(distances[index]),
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
