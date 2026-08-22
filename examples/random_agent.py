"""Run one episode with a random policy."""

import argparse

import gymnasium as gym

import rl_envs_astro  # noqa: F401 - importing registers the environments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="Astro/ZodiacAlignment-v0",
        choices=("Astro/ZodiacAlignment-v0", "Astro/TransitTiming-v0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="ansi")
    observation, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(env.render())
    print("episode reward: %.3f" % total_reward)
    print("final info:", info)
    env.close()


if __name__ == "__main__":
    main()
