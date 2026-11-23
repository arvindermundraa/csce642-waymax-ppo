"""Evaluates methods on the held-out scenario bank.

Methods are `expert`, `idm_ego`, `pure_pursuit`, or a PPO run directory.

Example:
    python scripts/evaluate.py --traffic replay --out results/replay \
        --methods expert idm_ego pure_pursuit runs/balanced_s0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from safedrive import data  # noqa: E402
from safedrive.env import REWARD_PRESETS, DrivingEnv  # noqa: E402
from safedrive.evaluate import episode_table, make_actor, rollout, summarize  # noqa: E402
from safedrive.ppo import load_checkpoint, make_policy  # noqa: E402


def evaluate_method(bank, method, traffic, reward, batch):
    policy = None
    if os.path.isdir(method):
        policy = make_policy(load_checkpoint(os.path.join(method, "checkpoint.pkl")))
        kind, dyn = "ppo", "bicycle"
    else:
        kind = method
        dyn = "state" if method in ("expert", "idm_ego") else "bicycle"
    env = DrivingEnv(bank, reward=REWARD_PRESETS[reward], traffic=traffic, ego_dynamics=dyn)
    act = make_actor(env, kind, policy)
    n = env.num_scenarios
    parts = []
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        if len(idx) < batch:                     # pad so every chunk has the same shape
            idx = np.concatenate([idx, np.full(batch - len(idx), idx[-1])])
        roll = rollout(env, act, idx)
        keep = min(batch, n - start)
        parts.append(episode_table({k: v[:keep] for k, v in roll.items()},
                                   bank["ids"][start:start + keep]))
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="data/cache/val.pkl")
    p.add_argument("--traffic", default="replay", choices=["replay", "idm"])
    p.add_argument("--reward", default="balanced", help="reward used to report return")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=56)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    bank = data.load_bank(args.bank)
    if args.limit:
        bank = data.split_bank(bank, np.arange(args.limit))
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for m in args.methods:
        t0 = time.time()
        name = os.path.basename(os.path.normpath(m))
        df = evaluate_method(bank, m, args.traffic, args.reward, args.batch)
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        summary[name] = summarize(df)
        s = summary[name]
        print(f"{name:16s} coll={s['collision']:.3f} off={s['offroad']:.3f} "
              f"prog={s['progress']:.3f} succ={s['success']:.3f} "
              f"ret={s['ep_return']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(args.out, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(summary)
    with open(path, "w") as f:
        json.dump(old, f, indent=2)


if __name__ == "__main__":
    main()
