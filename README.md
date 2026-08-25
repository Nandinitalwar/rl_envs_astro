# rl-envs-astro

Two deterministic astrology-themed reinforcement-learning tasks packaged for
[Harbor](https://github.com/harbor-framework/harbor), plus their reusable
Gymnasium environments. The mechanics are synthetic test problems, not claims
about astronomy, personality, or prediction.

## Harbor tasks

| Task | Objective | Reward |
| --- | --- | --- |
| `zodiac-alignment` | Move seven bodies around a cyclic 12-position ring until each reaches its target. | Partial credit for distance reduction; `1.0` for an optimal 25-action solution. |
| `transit-timing` | Wait or resolve three moving transits at favorable positions. | Normalized schedule return; `1.0` for the optimal timing policy. |

Each task follows Harbor schema `1.4`:

```text
tasks/<task>/
├── task.toml                 # metadata, limits, artifact contract
├── instruction.md            # prompt shown to the agent
├── environment/              # Dockerfile, scenario, inspection CLI
├── solution/solve.sh         # oracle policy
└── tests/test.sh             # independent verifier → reward.json
```

The agent inspects the scenario with `astro-task observe`, evaluates candidate
trajectories with `astro-task evaluate`, and writes `{"actions":[...]}` to
`/app/answer.json`. The verifier independently replays that trajectory and
writes scalar and diagnostic metrics to `/logs/verifier/reward.json`.

## Install and run

Requirements: Docker, `uv`, and Python 3.9+.

```bash
uv tool install harbor
uv sync --extra dev
```

Run the known-good oracle and the zero-action negative control:

```bash
harbor run -p tasks/zodiac-alignment -a oracle -e docker -n 1
harbor run -p tasks/zodiac-alignment -a nop -e docker -n 1
harbor run -p tasks/transit-timing -a oracle -e docker -n 1
harbor run -p tasks/transit-timing -a nop -e docker -n 1
```

Run an agent/model supported by Harbor:

```bash
harbor run -p tasks/zodiac-alignment -a codex -m <provider/model> -e docker -n 1
```

The root `dataset.toml` groups both content-addressed task packages.

## Gymnasium API

For online RL, the same mechanics remain available as step-by-step Gymnasium
environments:

```python
import gymnasium as gym
import rl_envs_astro  # registers Astro/* IDs

env = gym.make("Astro/ZodiacAlignment-v0")
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(1)
```

- `Astro/ZodiacAlignment-v0`: `Discrete(2 * n_bodies)` movement actions.
- `Astro/TransitTiming-v0`: wait (`0`) or resolve task `i` (`i + 1`).

An RL environment is a state machine: the policy receives an observation,
chooses an action, and receives a reward and terminal flags. Harbor wraps a
complete agent episode in an isolated container and scores its submitted
artifact; Gymnasium exposes each transition directly to a training loop.

## Test

```bash
uv run pytest
```

The suite checks Gymnasium contracts, deterministic dynamics, model-action JSON,
Harbor package layout, and oracle/NOP verifier behavior.

## Source layout

- `tasks/`: Harbor task packages and deterministic verifiers.
- `src/rl_envs_astro/envs/`: Gymnasium environment implementations.
- `src/rl_envs_astro/model_interface.py`: provider-independent JSON action
  protocol.
- `tests/`: unit and integration-level contract tests.
