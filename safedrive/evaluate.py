"""Deterministic evaluation rollouts, episode metrics and failure types."""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from safedrive.env import EPISODE_STEPS, DrivingEnv

SUCCESS_PROGRESS = 0.8     # fraction of the logged route that counts as "close to the log"
STALL_PROGRESS = 0.5
CURVE_KAPPA = 0.02         # 1/m, roughly a 50 m radius


def make_actor(env: DrivingEnv, method, policy=None):
    """Returns act(es, obs) -> waymax Action for a method name."""
    if method == "ppo":
        return lambda es, obs: env.bicycle_action(policy(obs))
    if method == "pure_pursuit":
        return lambda es, obs: env.bicycle_action(env.pure_pursuit(es))
    if method == "idm_ego":
        return lambda es, obs: env.idm_ego_action(es)
    if method == "expert":
        return lambda es, obs: env.expert_action(es)
    raise ValueError(method)


def rollout(env, act, idx):
    """Runs every scenario in idx for the full horizon and returns per-step
    arrays of shape (num_scenarios, EPISODE_STEPS). `alive` marks steps taken
    before (and including) the step that ended the episode."""

    @jax.jit
    def run(idx):
        es = jax.vmap(env.reset_idx)(idx)
        obs = jax.vmap(env.observe)(es)

        def body(carry, _):
            es, obs, alive = carry
            action = jax.vmap(act)(es, obs)
            es2, obs2, _, done, info = jax.vmap(env.step_waymax)(es, action)
            info = dict(info, alive=alive)
            return (es2, obs2, alive & ~done), info

        alive0 = jnp.ones(idx.shape, bool)
        _, infos = jax.lax.scan(body, (es, obs, alive0), None, EPISODE_STEPS)
        return jax.tree_util.tree_map(lambda a: jnp.swapaxes(a, 0, 1), infos)

    out = jax.device_get(run(jnp.asarray(idx)))
    return {k: np.asarray(v) for k, v in out.items()}


def classify_failure(r, t, collision, offroad, progress):
    """Sorts one episode into a failure type using the state at the first
    failure step t."""
    if collision:
        dx, dy = r["near_dx"][t], r["near_dy"][t]
        turning = (abs(r["route_kappa"][t]) > CURVE_KAPPA) or (abs(r["kappa"][t]) > CURVE_KAPPA)
        if dx < 0 and abs(dy) < 2.0:
            return "struck_from_behind"
        if turning:
            return "collision_turning"
        if dx > 0 and abs(dy) < 2.0:
            return "rear_end"
        return "collision_other"
    if offroad:
        return "offroad_curve" if abs(r["route_kappa"][t]) > CURVE_KAPPA else "offroad_straight"
    if progress < STALL_PROGRESS:
        return "stalled"
    return "none"


def episode_table(roll, ids):
    rows = []
    n = roll["alive"].shape[0]
    for i in range(n):
        r = {k: v[i] for k, v in roll.items()}
        alive = r["alive"]
        steps = int(alive.sum())
        last = steps - 1
        coll = bool((r["collision"] & alive).any())
        off = bool((r["offroad"] & alive).any())
        progress = float(r["progress"][last])
        # skip the first two steps: jerk needs two previous accelerations
        smooth = alive.copy()
        smooth[:2] = False
        smooth_n = max(int(smooth.sum()), 1)
        rows.append(dict(
            scenario=ids[i],
            steps=steps,
            ep_return=float((r["reward"] * alive).sum()),
            collision=coll,
            offroad=off,
            progress=progress,
            success=(not coll) and (not off) and progress >= SUCCESS_PROGRESS,
            mean_abs_jerk=float(np.abs(r["jerk"] * smooth).sum() / smooth_n),
            mean_abs_dkappa=float(np.abs(r["dkappa"] * smooth).sum() / smooth_n),
            mean_speed=float((r["speed"] * alive).sum() / max(steps, 1)),
            failure=classify_failure(r, last, coll, off, progress),
            fail_step=last if (coll or off) else -1,
        ))
    return pd.DataFrame(rows)


SUMMARY_COLS = ["ep_return", "collision", "offroad", "progress", "success",
                "mean_abs_jerk", "mean_abs_dkappa", "mean_speed"]


def summarize(df):
    return {c: float(df[c].mean()) for c in SUMMARY_COLS}
