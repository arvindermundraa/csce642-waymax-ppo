"""Loading WOMD scenarios, filtering them and caching a fixed scenario bank.

Reading tfrecords through Waymax's dataloader is slow, so we do it once,
keep only what training needs and pickle the result. Training and
evaluation then load the pickle and never touch TensorFlow.
"""
import dataclasses
import os
import pickle

import numpy as np

from safedrive import compat  # noqa: F401

ROUTE_POINTS = 400       # resampled route length (1 m spacing)
ROUTE_SPACING = 1.0
ROUTE_EXTENSION = 30.0   # straight extension past the end of the logged path
INIT_STEP = 10           # Waymax's first controllable timestep (init_steps=11)
LAST_STEP = 90


def scenario_iterator(path, max_num_objects=32):
    """Yields raw SimulatorStates from one tf_example tfrecord file."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from waymax import config, dataloader

    cfg = dataclasses.replace(
        config.WOD_1_1_0_VALIDATION, path=path,
        max_num_objects=max_num_objects, repeat=1, shuffle_seed=None)
    yield from dataloader.simulator_state_generator(cfg)


def to_numpy(tree):
    import jax
    return jax.tree_util.tree_map(np.asarray, jax.device_get(tree))


def sdc_index(state):
    return int(np.argmax(np.asarray(state.object_metadata.is_sdc)))


def keep_scenario(state, min_path_length=20.0):
    """True if the ego is tracked for the whole scene and actually moves."""
    if not bool(np.asarray(state.object_metadata.is_sdc).any()):
        return False
    i = sdc_index(state)
    valid = np.asarray(state.log_trajectory.valid[i])
    if not valid.all():
        return False
    x = np.asarray(state.log_trajectory.x[i, INIT_STEP:])
    y = np.asarray(state.log_trajectory.y[i, INIT_STEP:])
    return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) >= min_path_length


def build_route(state):
    """Resamples the logged ego path to evenly spaced points.

    Returns a dict with the route polyline, its curvature, and the arc
    length of the ego at the first controlled step and at the end of the log.
    """
    i = sdc_index(state)
    x = np.asarray(state.log_trajectory.x[i], dtype=np.float64)
    y = np.asarray(state.log_trajectory.y[i], dtype=np.float64)
    vx = np.asarray(state.log_trajectory.vel_x[i])
    vy = np.asarray(state.log_trajectory.vel_y[i])

    pts = np.stack([x, y], -1)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 0.05])      # drop repeated points while parked
    pts_k = pts[keep]
    s_k = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts_k, axis=0).T))])
    # arc length of every original timestep
    s_t = np.interp(np.arange(len(pts)), np.flatnonzero(keep), s_k)

    # straight extension along the final heading
    if len(pts_k) >= 2:
        d = pts_k[-1] - pts_k[-2]
    else:
        d = np.array([1.0, 0.0])
    d = d / (np.linalg.norm(d) + 1e-9)
    ext_s = np.arange(1, int(ROUTE_EXTENSION) + 1, dtype=np.float64)
    pts_e = np.concatenate([pts_k, pts_k[-1] + ext_s[:, None] * d])
    s_e = np.concatenate([s_k, s_k[-1] + ext_s])

    s_grid = np.arange(ROUTE_POINTS) * ROUTE_SPACING
    rx = np.interp(s_grid, s_e, pts_e[:, 0])
    ry = np.interp(s_grid, s_e, pts_e[:, 1])
    route = np.stack([rx, ry], -1)
    route_valid = s_grid <= s_e[-1]

    heading = np.unwrap(np.arctan2(np.gradient(ry), np.gradient(rx)))
    kappa = np.gradient(heading) / ROUTE_SPACING
    kappa = np.convolve(kappa, np.ones(5) / 5, mode="same")

    speed = np.hypot(vx, vy)
    return dict(
        route=route.astype(np.float32),
        route_valid=route_valid,
        route_kappa=kappa.astype(np.float32),
        s_init=np.float32(s_t[INIT_STEP]),
        s_end=np.float32(s_t[LAST_STEP]),
        log_speed=speed.astype(np.float32),
    )


def crop_roadgraph(state, route, num_points):
    """Keeps the roadgraph points closest to the logged ego path.

    Off-road checks only ever look near the ego, and 20k points per scene
    make the metric slow, so we keep the nearest `num_points`.
    """
    rg = state.roadgraph_points
    px, py = np.asarray(rg.x), np.asarray(rg.y)
    valid = np.asarray(rg.valid)
    ref = route[::5]
    d2 = ((px[:, None] - ref[None, :, 0]) ** 2 +
          (py[:, None] - ref[None, :, 1]) ** 2).min(axis=1)
    d2 = np.where(valid, d2, np.inf)
    idx = np.sort(np.argsort(d2)[:num_points])
    import jax
    return state.replace(
        roadgraph_points=jax.tree_util.tree_map(lambda a: np.asarray(a)[idx], rg))


def build_bank(paths, num_scenarios, rg_points=3000, max_num_objects=32,
               min_path_length=20.0, verbose=True):
    """Reads scenarios from tfrecords until `num_scenarios` pass the filter."""
    import jax
    states, routes, ids = [], [], []
    seen = 0
    for path in paths:
        for raw in scenario_iterator(path, max_num_objects):
            seen += 1
            s = to_numpy(raw)
            if not keep_scenario(s, min_path_length):
                continue
            r = build_route(s)
            s = crop_roadgraph(s, r["route"], rg_points)
            states.append(s)
            routes.append(r)
            ids.append(f"{os.path.basename(path)}#{seen - 1}")
            if verbose and len(states) % 50 == 0:
                print(f"  kept {len(states)} / read {seen}")
            if len(states) >= num_scenarios:
                break
        if len(states) >= num_scenarios:
            break
    stacked = jax.tree_util.tree_map(lambda *a: np.stack(a), *states)
    route_arrays = {k: np.stack([r[k] for r in routes]) for k in routes[0]}
    if verbose:
        print(f"kept {len(states)} of {seen} scenarios read")
    return dict(states=stacked, routes=route_arrays, ids=ids)


def save_bank(bank, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bank(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_bank(bank, idx):
    import jax
    idx = np.asarray(idx)
    return dict(
        states=jax.tree_util.tree_map(lambda a: a[idx], bank["states"]),
        routes={k: v[idx] for k, v in bank["routes"].items()},
        ids=[bank["ids"][i] for i in idx],
    )
