"""Gymnasium registration for the astrology-themed environments."""

from gymnasium.envs.registration import register, registry

from rl_envs_astro.envs import TransitTimingEnv, ZodiacAlignmentEnv
from rl_envs_astro.model_interface import build_model_prompt, parse_model_action

ALIGNMENT_ENV_ID = "Astro/ZodiacAlignment-v0"
TRANSIT_ENV_ID = "Astro/TransitTiming-v0"


def _register_environments() -> None:
    if ALIGNMENT_ENV_ID not in registry:
        register(
            id=ALIGNMENT_ENV_ID,
            entry_point="rl_envs_astro.envs:ZodiacAlignmentEnv",
        )
    if TRANSIT_ENV_ID not in registry:
        register(
            id=TRANSIT_ENV_ID,
            entry_point="rl_envs_astro.envs:TransitTimingEnv",
        )


_register_environments()

__all__ = [
    "ALIGNMENT_ENV_ID",
    "TRANSIT_ENV_ID",
    "TransitTimingEnv",
    "ZodiacAlignmentEnv",
    "build_model_prompt",
    "parse_model_action",
]
