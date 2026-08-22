# rl-envs-astro

Two Gymnasium environments with synthetic astrology mechanics. They are RL test
problems, not claims about astronomy, personality, or prediction.

## What an RL environment looks like

An environment is a state machine. The agent observes state, chooses an action,
receives a reward, and repeats until the episode ends:

```text
agent -- action --> environment
agent <-- observation, reward, terminated/truncated -- environment
```

Gymnasium exposes that interaction through two methods:

```python
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(action)
```

- **Observation:** information available to the policy.
- **Action:** a legal decision from `env.action_space`.
- **Reward:** scalar feedback for the last transition.
- **Terminated:** the task reached a terminal state.
- **Truncated:** the time limit ended the episode.
- **Info:** diagnostics that are not part of the policy input.

## Environments

### `Astro/ZodiacAlignment-v0`

A goal-reaching problem on a cyclic 12-position ring.

- Observation: body `positions`, `targets`, and `steps_remaining`.
- Action space: `Discrete(2 * n_bodies)`; select a body and move it one position
  clockwise or counterclockwise.
- Reward: reduction in total circular distance minus `step_cost`, plus
  `success_bonus` when all bodies reach their targets.
- End condition: all bodies aligned, or `max_steps` exhausted.

### `Astro/TransitTiming-v0`

An optimal-stopping and scheduling problem.

- Observation: transit `positions`, task `targets`, `resolved` mask, and
  `days_remaining`.
- Action space: `Discrete(n_tasks + 1)`; action `0` waits and action `i + 1`
  resolves task `i`.
- Resolution score: `1 - circular_distance(position, target) / 6`.
- Reward: resolution score minus wait/repeated-action costs; unresolved tasks
  receive a penalty at the horizon.
- End condition: all tasks resolved, or `horizon` exhausted.

Both environments support deterministic seeding, configurable horizons and
rewards, controlled reset states, Gymnasium registration, and ANSI/human
rendering.

## Install

```bash
uv sync --extra dev
source .venv/bin/activate
```

Alternatively:

```bash
python -m pip install -e '.[dev]'
```

## Run an episode

```python
import gymnasium as gym
import rl_envs_astro  # registers Astro/* environment IDs

env = gym.make("Astro/ZodiacAlignment-v0", render_mode="ansi")
observation, info = env.reset(seed=7)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

print(env.render())
env.close()
```

Run the demos and tests:

```bash
python examples/random_agent.py --env Astro/TransitTiming-v0 --seed 7
pytest
```

## Model policies

`build_model_prompt()` serializes the rules, configuration, legal actions, and
observation. `parse_model_action()` accepts only a valid JSON action before it
reaches `env.step()`:

```python
prompt = rl_envs_astro.build_model_prompt(env, observation)
response = '{"action": 0}'  # replace with model inference
action = rl_envs_astro.parse_model_action(env, response)
```

The protocol is provider-independent and does not require API credentials.

## Source layout

- [`envs/zodiac_alignment.py`](src/rl_envs_astro/envs/zodiac_alignment.py):
  alignment dynamics and reward.
- [`envs/transit_timing.py`](src/rl_envs_astro/envs/transit_timing.py): timing
  dynamics and reward.
- [`model_interface.py`](src/rl_envs_astro/model_interface.py): JSON model
  protocol.
- [`tests/test_environments.py`](tests/test_environments.py): Gymnasium contract,
  behavior, seeding, registration, rendering, and protocol tests.
