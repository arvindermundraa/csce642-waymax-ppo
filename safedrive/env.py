"""Single-agent driving environment on top of Waymax's PlanningAgentEnvironment.

The ego car is driven either through the bicycle model (acceleration,
curvature) or, for the reference methods, through state dynamics. Other
cars replay their logs or follow Waymax's IDM route policy.
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from safedrive import compat  # noqa: F401
from safedrive import geometry as geo
from safedrive.data import INIT_STEP, LAST_STEP

from waymax import agents, config as wcfg, datatypes, dynamics
from waymax import env as wenv
from waymax.metrics import roadgraph as rg_metrics

DT = 0.1
MAX_ACCEL = 6.0      # bicycle model bounds (InvertibleBicycleModel defaults)
MAX_KAPPA = 0.3
EPISODE_STEPS = LAST_STEP - INIT_STEP     # 80 control steps
LOOKAHEAD = (5.0, 10.0, 15.0, 20.0, 25.0)
NEIGHBOR_RANGE = 50.0


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    w_progress: float = 1.0
    w_collision: float = 25.0
    w_offroad: float = 25.0
    w_control: float = 0.05


REWARD_PRESETS = {
    "progress": RewardConfig(1.0, 10.0, 10.0, 0.02),
    "balanced": RewardConfig(1.0, 50.0, 50.0, 0.05),
    "safety": RewardConfig(1.0, 200.0, 200.0, 0.1),
}
PROGRESS_LEAD = 5.0   # m the ego may get ahead of the logged car and still earn progress


@struct.dataclass
class EnvState:
    sim: Any
    idx: jax.Array        # scenario index in the bank
    s: jax.Array          # ego arc length along the logged route
    speed: jax.Array
    accel: jax.Array      # realised longitudinal acceleration of the last step
    kappa: jax.Array      # realised path curvature of the last step
    ep_return: jax.Array
    ep_steps: jax.Array


class DrivingEnv:

    def __init__(self, bank, reward=REWARD_PRESETS["balanced"], traffic="replay",
                 ego_dynamics="bicycle", k_neighbors=5, idm_speed=15.0):
        self.states = jax.tree_util.tree_map(jnp.asarray, bank["states"])
        self.routes = {k: jnp.asarray(v) for k, v in bank["routes"].items()}
        self.num_scenarios = int(self.routes["route"].shape[0])
        self.num_objects = int(self.states.log_trajectory.x.shape[1])
        self.reward_cfg = reward
        self.traffic = traffic
        self.ego_dynamics = ego_dynamics
        self.k = k_neighbors
        self.idm_speed = idm_speed

        cfg = dataclasses.replace(
            wcfg.EnvironmentConfig(),
            max_num_objects=self.num_objects,
            init_steps=INIT_STEP + 1,
            controlled_object=wcfg.ObjectType.SDC,
            metrics=wcfg.MetricsConfig(metrics_to_run=("overlap",)),
        )
        dyn = (dynamics.InvertibleBicycleModel() if ego_dynamics == "bicycle"
               else dynamics.StateDynamics())
        actors = []
        if traffic == "idm":
            actors = [agents.IDMRoutePolicy(
                is_controlled_func=lambda s: ~s.object_metadata.is_sdc,
                desired_vel=idm_speed)]
        elif traffic != "replay":
            raise ValueError(f"unknown traffic model {traffic}")
        self.wenv = wenv.PlanningAgentEnvironment(
            dyn, cfg, sim_agent_actors=actors, sim_agent_params=[None] * len(actors))
        self.obs_dim = 7 + 2 * len(LOOKAHEAD) + 5 * self.k

    # ------------------------------------------------------------------ helpers
    def _scenario(self, idx):
        return jax.tree_util.tree_map(lambda a: a[idx], self.states)

    def _route(self, idx):
        return {k: v[idx] for k, v in self.routes.items()}

    @staticmethod
    def _sdc(sim):
        return jnp.argmax(sim.object_metadata.is_sdc)

    def _ego(self, sim):
        traj = sim.current_sim_trajectory
        i = self._sdc(sim)
        return (traj.x[i, 0], traj.y[i, 0], traj.yaw[i, 0],
                traj.vel_x[i, 0], traj.vel_y[i, 0])

    # -------------------------------------------------------------------- reset
    def reset_idx(self, idx):
        idx = jnp.asarray(idx, jnp.int32)
        sim = self.wenv.reset(self._scenario(idx))
        r = self._route(idx)
        x, y, _, vx, vy = self._ego(sim)
        s, _, _, _ = geo.project_to_route(r["route"], r["route_valid"], x, y)
        zero = jnp.zeros((), jnp.float32)
        return EnvState(sim=sim, idx=idx, s=s, speed=jnp.hypot(vx, vy),
                        accel=zero, kappa=zero, ep_return=zero,
                        ep_steps=jnp.zeros((), jnp.int32))

    def reset(self, key):
        return self.reset_idx(jax.random.randint(key, (), 0, self.num_scenarios))

    # -------------------------------------------------------------- observation
    def observe(self, es):
        sim = es.sim
        r = self._route(es.idx)
        x, y, yaw, vx, vy = self._ego(sim)
        s, lat, path_yaw, _ = geo.project_to_route(r["route"], r["route_valid"], x, y)
        heading_err = geo.wrap_angle(yaw - path_yaw)
        length = jnp.maximum(r["s_end"] - r["s_init"], 1.0)
        progress = (s - r["s_init"]) / length
        t_left = (LAST_STEP - sim.timestep) / EPISODE_STEPS

        look = []
        for d in LOOKAHEAD:
            p = geo.point_at(r["route"], r["route_valid"], s + d)
            lx, ly = geo.to_ego_frame(p[0] - x, p[1] - y, yaw)
            look += [lx / 25.0, ly / 25.0]

        nb = self._neighbors(sim, x, y, yaw, vx, vy)
        feats = jnp.stack([es.speed / 15.0, heading_err, jnp.clip(lat / 4.0, -3, 3),
                           progress, t_left, es.accel / MAX_ACCEL,
                           es.kappa / MAX_KAPPA] + look)
        return jnp.concatenate([feats, nb.reshape(-1)]).astype(jnp.float32)

    def _neighbors(self, sim, x, y, yaw, vx, vy, raw=False):
        traj = sim.current_sim_trajectory
        n = traj.x.shape[0]
        others = traj.valid[:, 0] & (jnp.arange(n) != self._sdc(sim))
        dx, dy = traj.x[:, 0] - x, traj.y[:, 0] - y
        d2 = jnp.where(others, dx ** 2 + dy ** 2, jnp.inf)
        order = jnp.argsort(d2)[: self.k]
        px, py = geo.to_ego_frame(dx[order], dy[order], yaw)
        rvx, rvy = geo.to_ego_frame(traj.vel_x[order, 0] - vx,
                                    traj.vel_y[order, 0] - vy, yaw)
        present = (d2[order] < NEIGHBOR_RANGE ** 2).astype(jnp.float32)
        if raw:
            return px, py, rvx, rvy, present, traj.length[order, 0]
        nb = jnp.stack([px / NEIGHBOR_RANGE, py / NEIGHBOR_RANGE,
                        rvx / 15.0, rvy / 15.0, present], -1)
        return nb * present[:, None]

    def _log_arclength(self, sim, r, t):
        i = self._sdc(sim)
        lx = jax.lax.dynamic_index_in_dim(sim.log_trajectory.x[i], t, keepdims=False)
        ly = jax.lax.dynamic_index_in_dim(sim.log_trajectory.y[i], t, keepdims=False)
        s, _, _, _ = geo.project_to_route(r["route"], r["route_valid"], lx, ly)
        return s

    def ego_offroad(self, sim):
        """Waymax's off-road check run for the ego only (the stock metric
        checks every object and dominates the step time)."""
        traj = datatypes.select_by_onehot(
            sim.current_sim_trajectory, sim.object_metadata.is_sdc, keepdims=True)
        return rg_metrics.is_offroad(traj, sim.roadgraph_points).reshape(())

    # --------------------------------------------------------------------- step
    def bicycle_action(self, a_norm):
        a = jnp.clip(a_norm, -1.0, 1.0) * jnp.array([MAX_ACCEL, MAX_KAPPA])
        return datatypes.Action(data=a, valid=jnp.ones((1,), bool))

    def step(self, es, a_norm):
        """RL step: normalised [accel, curvature] in [-1, 1]."""
        return self.step_waymax(es, self.bicycle_action(a_norm))

    def step_waymax(self, es, action):
        cfg = self.reward_cfg
        sim = self.wenv.step(es.sim, action)
        m = self.wenv.metrics(sim)
        collision = m["overlap"].value.reshape(()) > 0
        offroad = self.ego_offroad(sim)

        r = self._route(es.idx)
        x, y, yaw, vx, vy = self._ego(sim)
        _, _, prev_yaw, _, _ = self._ego(es.sim)
        s, lat, _, _ = geo.project_to_route(r["route"], r["route_valid"], x, y)
        speed = jnp.hypot(vx, vy)
        accel = (speed - es.speed) / DT
        dyaw = geo.wrap_angle(yaw - prev_yaw)
        kappa = jnp.where(speed > 0.5, dyaw / (jnp.maximum(speed, 0.5) * DT), 0.0)
        jerk = (accel - es.accel) / DT
        dkappa = kappa - es.kappa

        length = jnp.maximum(r["s_end"] - r["s_init"], 1.0)
        # progress only counts up to where the logged car is (plus a margin),
        # so the agent gains nothing by driving faster than the human did
        cap_prev = self._log_arclength(es.sim, r, es.sim.timestep) + PROGRESS_LEAD
        cap_now = self._log_arclength(sim, r, sim.timestep) + PROGRESS_LEAD
        r_progress = 100.0 * (jnp.minimum(s, cap_now) - jnp.minimum(es.s, cap_prev)) / length
        harsh_brake = jnp.maximum(0.0, -accel - 3.0) / 3.0
        control = jnp.clip((jerk / 10.0) ** 2 + (dkappa / 0.02) ** 2 + harsh_brake, 0.0, 10.0)
        reward = (cfg.w_progress * r_progress
                  - cfg.w_collision * collision
                  - cfg.w_offroad * offroad
                  - cfg.w_control * control)

        timeout = sim.timestep >= LAST_STEP
        done = collision | offroad | timeout
        es2 = EnvState(sim=sim, idx=es.idx, s=s, speed=speed, accel=accel,
                       kappa=kappa, ep_return=es.ep_return + reward,
                       ep_steps=es.ep_steps + 1)

        px, py, rvx, _, present, olen = self._neighbors(sim, x, y, yaw, vx, vy, raw=True)
        info = dict(
            done=done, collision=collision, offroad=offroad,
            progress=(s - r["s_init"]) / length,
            ep_return=es2.ep_return, ep_steps=es2.ep_steps,
            x=x, y=y, yaw=yaw, speed=speed, accel=accel, kappa=kappa,
            jerk=jerk, dkappa=dkappa, lat=lat,
            route_kappa=geo.curvature_at(r["route_kappa"], s),
            near_dx=px[0], near_dy=py[0], near_rvx=rvx[0],
            near_present=present[0], near_len=olen[0],
            reward=reward,
        )
        return es2, self.observe(es2), reward, done, info

    # ----------------------------------------------------- reference controllers
    def pure_pursuit(self, es, lookahead_gain=0.8, max_speed=15.0, headway=1.5,
                     min_gap=2.0, a_max=2.0, b_comf=3.0):
        """Hand-designed baseline: pure pursuit on the logged route plus an
        IDM-style speed rule. Returns a normalised bicycle action."""
        r = self._route(es.idx)
        x, y, yaw, vx, vy = self._ego(es.sim)
        v = jnp.hypot(vx, vy)
        ld = jnp.clip(4.0 + lookahead_gain * v, 5.0, 20.0)
        p = geo.point_at(r["route"], r["route_valid"], es.s + ld)
        lx, ly = geo.to_ego_frame(p[0] - x, p[1] - y, yaw)
        kappa = 2.0 * ly / jnp.maximum(lx ** 2 + ly ** 2, 1.0)

        # target speed: the logged average speed, capped
        v0 = jnp.clip(jnp.mean(r["log_speed"][INIT_STEP:]) + 1.0, 3.0, max_speed)

        # lead vehicle: nearest object ahead within half a lane width
        px, py, rvx, _, present, olen = self._neighbors(es.sim, x, y, yaw, vx, vy, raw=True)
        ego_len = es.sim.current_sim_trajectory.length[self._sdc(es.sim), 0]
        ahead = (present > 0) & (px > 0) & (jnp.abs(py) < 2.0)
        gap = jnp.where(ahead, px - 0.5 * (olen + ego_len), jnp.inf)
        j = jnp.argmin(gap)
        gap_j = jnp.maximum(gap[j], 0.1)
        dv = -rvx[j]                           # positive when closing in
        s_star = min_gap + v * headway + v * dv / (2.0 * jnp.sqrt(a_max * b_comf))
        interaction = jnp.where(jnp.isfinite(gap[j]), (jnp.maximum(s_star, 0.0) / gap_j) ** 2, 0.0)
        accel = a_max * (1.0 - (v / v0) ** 4 - interaction)
        accel = jnp.clip(accel, -MAX_ACCEL, a_max)
        return jnp.array([accel / MAX_ACCEL, kappa / MAX_KAPPA])

    def expert_action(self, es):
        """Logged human trajectory (needs ego_dynamics='state')."""
        sim = es.sim
        i = self._sdc(sim)
        t = sim.timestep + 1
        lt = sim.log_trajectory
        get = lambda a: jax.lax.dynamic_index_in_dim(a[i], t, keepdims=False)
        data = jnp.stack([get(lt.x), get(lt.y), get(lt.yaw), get(lt.vel_x), get(lt.vel_y)])
        return datatypes.Action(data=data, valid=jnp.ones((1,), bool))

    def idm_ego_action(self, es):
        """Waymax's IDM route-following policy driving the ego (state dynamics)."""
        if not hasattr(self, "_idm_ego"):
            self._idm_ego = agents.IDMRoutePolicy(
                is_controlled_func=lambda s: s.object_metadata.is_sdc,
                desired_vel=self.idm_speed)
        out = self._idm_ego.select_action(None, es.sim, None, None)
        i = self._sdc(es.sim)
        return datatypes.Action(data=out.action.data[i], valid=out.action.valid[i])
