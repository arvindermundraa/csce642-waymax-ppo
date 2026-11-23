"""Trains a PPO driving policy.

Example:
    python scripts/train.py --reward balanced --seed 0 --out runs/balanced_s0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml  # noqa: E402

from safedrive import data  # noqa: E402
from safedrive.env import REWARD_PRESETS, DrivingEnv  # noqa: E402
from safedrive.ppo import PPOConfig, train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ppo.yaml")
    p.add_argument("--bank", default="data/cache/train.pkl")
    p.add_argument("--reward", default="balanced", choices=sorted(REWARD_PRESETS))
    p.add_argument("--traffic", default="replay", choices=["replay", "idm"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = PPOConfig(**yaml.safe_load(f))
    if args.steps:
        cfg.total_timesteps = args.steps

    bank = data.load_bank(args.bank)
    env = DrivingEnv(bank, reward=REWARD_PRESETS[args.reward], traffic=args.traffic)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(dict(vars(args), ppo=cfg.__dict__, num_scenarios=env.num_scenarios), f, indent=2)
    print(f"training on {env.num_scenarios} scenarios, {cfg.num_updates} updates")
    train(env, cfg, args.seed, args.out)


if __name__ == "__main__":
    main()
