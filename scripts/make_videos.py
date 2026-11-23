"""Renders episodes to mp4 for the failure analysis.

Picks scenarios from an evaluation CSV: a couple of successes and one
example of each failure type that occurred.

Example:
    python scripts/make_videos.py --method runs/balanced_s0 \
        --eval_csv results/replay/balanced_s0.csv --out videos/balanced_s0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax  # noqa: E402
import mediapy  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from safedrive import data  # noqa: E402
from safedrive.env import DrivingEnv  # noqa: E402
from safedrive.evaluate import make_actor  # noqa: E402
from safedrive.ppo import load_checkpoint, make_policy  # noqa: E402
from safedrive.render import draw_frame  # noqa: E402


def pick_scenarios(df, n_success=2):
    picks = []
    ok = df[df.success]
    picks += [(i, "success") for i in ok.index[:n_success]]
    for ftype in sorted(set(df.failure) - {"none"}):
        rows = df[df.failure == ftype]
        picks.append((rows.index[0], ftype))
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="data/cache/val.pkl")
    p.add_argument("--method", required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--traffic", default="replay")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    bank = data.load_bank(args.bank)
    df = pd.read_csv(args.eval_csv)
    if os.path.isdir(args.method):
        policy = make_policy(load_checkpoint(os.path.join(args.method, "checkpoint.pkl")))
        kind, dyn = "ppo", "bicycle"
    else:
        policy, kind = None, args.method
        dyn = "state" if kind in ("expert", "idm_ego") else "bicycle"
    env = DrivingEnv(bank, traffic=args.traffic, ego_dynamics=dyn)
    act = jax.jit(make_actor(env, kind, policy))
    step = jax.jit(env.step_waymax)
    name = os.path.basename(os.path.normpath(args.method))
    os.makedirs(args.out, exist_ok=True)

    for idx, label in pick_scenarios(df):
        es = env.reset_idx(idx)
        obs = env.observe(es)
        route = np.asarray(bank["routes"]["route"][idx])
        valid = np.asarray(bank["routes"]["route_valid"][idx])
        frames, status = [], ""
        for t in range(80):
            es, obs, _, done, info = step(es, act(es, obs))
            if bool(info["collision"]):
                status = "COLLISION"
            elif bool(info["offroad"]):
                status = "OFF-ROAD"
            sim = jax.tree_util.tree_map(np.asarray, jax.device_get(es.sim))
            frames.append(draw_frame(
                sim, route[valid], title=f"{name} | {label} | {bank['ids'][idx]}",
                status=f"t={t * 0.1:.1f}s  v={float(info['speed']):.1f} m/s  {status}"))
            if bool(done):
                frames += [frames[-1]] * 10
                break
        path = os.path.join(args.out, f"{label}_{idx}.mp4")
        mediapy.write_video(path, frames, fps=10)
        print("wrote", path)


if __name__ == "__main__":
    main()
