import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rl_envs_astro
from rl_envs_astro.constants import ZODIAC_SIGNS
from rl_envs_astro.envs import TransitTimingEnv, ZodiacAlignmentEnv
from rl_envs_astro.model_interface import build_model_prompt, parse_model_action


@pytest.mark.parametrize(
    "env_id",
    [rl_envs_astro.ALIGNMENT_ENV_ID, rl_envs_astro.TRANSIT_ENV_ID],
)
def test_gymnasium_contract(env_id):
    env = gym.make(env_id).unwrapped
    check_env(env)
    env.close()


@pytest.mark.parametrize(
    "env_id",
    [rl_envs_astro.ALIGNMENT_ENV_ID, rl_envs_astro.TRANSIT_ENV_ID],
)
def test_registered_environment_runs(env_id):
    env = gym.make(env_id)
    observation, info = env.reset(seed=10)
    assert env.observation_space.contains(observation)

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_alignment_seed_is_deterministic():
    env = ZodiacAlignmentEnv()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first["positions"], second["positions"])
    np.testing.assert_array_equal(first["targets"], second["targets"])


def test_alignment_distance_reward_and_success_bonus():
    env = ZodiacAlignmentEnv(n_bodies=1, max_steps=4, step_cost=0.01, success_bonus=10.0)
    observation, _ = env.reset(options={"positions": [0], "targets": [1]})
    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["positions"].tolist() == [1]
    assert reward == pytest.approx(10.99)
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True


def test_alignment_truncates_at_configured_limit():
    env = ZodiacAlignmentEnv(n_bodies=1, max_steps=1)
    env.reset(options={"positions": [0], "targets": [6]})
    _, _, terminated, truncated, _ = env.step(1)
    assert terminated is False
    assert truncated is True


def test_transit_waiting_can_improve_timing():
    env = TransitTimingEnv(n_tasks=1, horizon=3, transit_speeds=(1,))
    observation, _ = env.reset(options={"positions": [0], "targets": [1]})

    observation, wait_reward, terminated, truncated, _ = env.step(0)
    assert observation["positions"].tolist() == [1]
    assert wait_reward == pytest.approx(-0.02)
    assert not terminated and not truncated

    _, action_reward, terminated, truncated, info = env.step(1)
    assert action_reward == pytest.approx(1.0)
    assert info["timing_score"] == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False


def test_transit_penalizes_unfinished_tasks_at_horizon():
    env = TransitTimingEnv(
        n_tasks=2,
        horizon=1,
        transit_speeds=(1, 1),
        wait_cost=0.02,
        missed_task_penalty=0.5,
    )
    env.reset(options={"positions": [0, 0], "targets": [0, 0]})
    _, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(-1.02)
    assert terminated is False
    assert truncated is True
    assert info["unresolved_tasks"] == 2


@pytest.mark.parametrize("env_class", [ZodiacAlignmentEnv, TransitTimingEnv])
def test_ansi_render_contains_environment_state(env_class):
    env = env_class(render_mode="ansi")
    env.reset(seed=2)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "Sun" in rendered
    assert any(sign in rendered for sign in ZODIAC_SIGNS)


@pytest.mark.parametrize("key", ["positions", "targets"])
def test_reset_rejects_invalid_signs(key):
    env = ZodiacAlignmentEnv(n_bodies=2)
    with pytest.raises(ValueError):
        env.reset(options={key: [0, 12]})


def test_model_prompt_includes_transit_config_and_actions():
    env = gym.make(rl_envs_astro.TRANSIT_ENV_ID, transit_speeds=(1, 5, 7))
    observation, _ = env.reset(seed=7)
    prompt = build_model_prompt(env, observation)

    assert '"protocol": "rl-envs-astro/model-action-v1"' in prompt
    assert '"transit_speeds"' in prompt
    assert '"resolve Mercury task"' in prompt
    assert '"positions"' in prompt


@pytest.mark.parametrize("response", ['{"action": 1}', '```json\n{"action": 1}\n```'])
def test_model_action_parser_accepts_valid_json(response):
    env = ZodiacAlignmentEnv(n_bodies=1)
    assert parse_model_action(env, response) == 1


@pytest.mark.parametrize(
    "response",
    [
        "action 1",
        '{"action": true}',
        '{"action": 2}',
        '{"action": 1, "reason": "clockwise"}',
    ],
)
def test_model_action_parser_rejects_invalid_responses(response):
    env = ZodiacAlignmentEnv(n_bodies=1)
    with pytest.raises(ValueError):
        parse_model_action(env, response)
