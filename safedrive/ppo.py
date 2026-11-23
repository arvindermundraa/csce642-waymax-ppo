"""PPO with a Gaussian actor and an MLP critic, written in JAX/Flax.

The structure follows CleanRL's ppo_continuous_action.py: GAE, clipped
surrogate, value loss, advantage normalisation, learning-rate annealing and
gradient clipping. Rollouts are a jax.lax.scan over vmapped environments so a
whole update runs as one compiled function.
"""
import dataclasses
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState


@dataclasses.dataclass
class PPOConfig:
    num_envs: int = 64
    num_steps: int = 64
    total_timesteps: int = 2_000_000
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 8
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    log_std_init: float = -1.6
    reward_scale: float = 0.1

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def num_updates(self):
        return self.total_timesteps // self.batch_size


class ActorCritic(nn.Module):
    action_dim: int = 2
    hidden: int = 256
    log_std_init: float = -1.6

    @nn.compact
    def __call__(self, x):
        init = nn.initializers.orthogonal
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=init(np.sqrt(2)))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=init(np.sqrt(2)))(a))
        mean = nn.Dense(self.action_dim, kernel_init=init(0.01))(a)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init),
                             (self.action_dim,))
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=init(np.sqrt(2)))(x))
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=init(np.sqrt(2)))(v))
        value = nn.Dense(1, kernel_init=init(1.0))(v)
        return mean, log_std, value[..., 0]


def gaussian_logp(x, mean, log_std):
    return jnp.sum(-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std
                   - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


@struct.dataclass
class ObsNorm:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @staticmethod
    def create(dim):
        return ObsNorm(jnp.zeros(dim), jnp.ones(dim), jnp.array(1e-4))

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot
        return ObsNorm(mean, m2 / tot, tot)

    def __call__(self, x):
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -10, 10)


@struct.dataclass
class Runner:
    train_state: TrainState
    obs_norm: ObsNorm
    env_state: object
    obs: jax.Array
    key: jax.Array


def make_trainer(env, cfg: PPOConfig):
    net = ActorCritic(hidden=cfg.hidden, log_std_init=cfg.log_std_init)

    def lr_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    def init(key):
        k_net, k_env, key = jax.random.split(key, 3)
        params = net.init(k_net, jnp.zeros((env.obs_dim,)))
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                         optax.adam(lr_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        es = jax.vmap(env.reset)(jax.random.split(k_env, cfg.num_envs))
        obs = jax.vmap(env.observe)(es)
        return Runner(ts, ObsNorm.create(env.obs_dim), es, obs, key)

    def env_step(carry, _):
        runner = carry
        key, k_act, k_reset = jax.random.split(runner.key, 3)
        obs_n = runner.obs_norm(runner.obs)
        mean, log_std, value = net.apply(runner.train_state.params, obs_n)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        logp = gaussian_logp(action, mean, log_std)

        es2, obs2, reward, done, info = jax.vmap(env.step)(runner.env_state, action)
        # auto-reset finished episodes with a new random scenario
        fresh = jax.vmap(env.reset)(jax.random.split(k_reset, cfg.num_envs))
        fresh_obs = jax.vmap(env.observe)(fresh)
        pick = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        es_next = jax.tree_util.tree_map(pick, fresh, es2)
        obs_next = pick(fresh_obs, obs2)

        transition = dict(obs=runner.obs, action=action, logp=logp, value=value,
                          reward=reward * cfg.reward_scale, done=done,
                          ep_done=done, ep_return=info["ep_return"],
                          collision=info["collision"], offroad=info["offroad"],
                          progress=info["progress"])
        return runner.replace(env_state=es_next, obs=obs_next, key=key), transition

    def gae(traj, last_value):
        def body(carry, t):
            adv, next_value = carry
            nonterm = 1.0 - t["done"]
            delta = t["reward"] + cfg.gamma * next_value * nonterm - t["value"]
            adv = delta + cfg.gamma * cfg.gae_lambda * nonterm * adv
            return (adv, t["value"]), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value),
                              traj, reverse=True)
        return adv, adv + traj["value"]

    def loss_fn(params, obs_norm, mb):
        mean, log_std, value = net.apply(params, obs_norm(mb["obs"]))
        logp = gaussian_logp(mb["action"], mean, log_std)
        ratio = jnp.exp(logp - mb["logp"])
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_loss = 0.5 * ((value - mb["ret"]) ** 2).mean()
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        total = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
        return total, dict(pg_loss=pg, v_loss=v_loss, entropy=entropy, approx_kl=approx_kl)

    @jax.jit
    def update(runner):
        runner, traj = jax.lax.scan(env_step, runner, None, cfg.num_steps)
        obs_norm = runner.obs_norm.update(traj["obs"])
        _, _, last_value = net.apply(runner.train_state.params, runner.obs_norm(runner.obs))
        adv, ret = gae(traj, last_value)
        batch = dict(obs=traj["obs"], action=traj["action"], logp=traj["logp"],
                     adv=adv, ret=ret)
        batch = jax.tree_util.tree_map(lambda a: a.reshape((cfg.batch_size,) + a.shape[2:]), batch)

        def epoch(carry, _):
            ts, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, cfg.batch_size)
            mbs = jax.tree_util.tree_map(
                lambda a: a[perm].reshape((cfg.num_minibatches, -1) + a.shape[1:]), batch)

            def minibatch(ts, mb):
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    ts.params, runner.obs_norm, mb)
                return ts.apply_gradients(grads=grads), aux

            ts, aux = jax.lax.scan(minibatch, ts, mbs)
            return (ts, key), aux

        (ts, key), aux = jax.lax.scan(epoch, (runner.train_state, runner.key), None,
                                      cfg.update_epochs)
        runner = runner.replace(train_state=ts, obs_norm=obs_norm, key=key)

        d = traj["ep_done"]
        n_done = jnp.maximum(d.sum(), 1)
        stats = dict(
            episodes=d.sum(),
            ep_return=(traj["ep_return"] * d).sum() / n_done,
            collision=(traj["collision"] * d).sum() / n_done,
            offroad=(traj["offroad"] * d).sum() / n_done,
            progress=(traj["progress"] * d).sum() / n_done,
            **jax.tree_util.tree_map(jnp.mean, aux),
        )
        return runner, stats

    return net, init, update


def save_checkpoint(path, runner, update_idx, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = dict(params=jax.device_get(runner.train_state.params),
                opt_state=jax.device_get(runner.train_state.opt_state),
                obs_norm=jax.device_get(runner.obs_norm),
                update=update_idx, config=dataclasses.asdict(cfg))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(env, cfg, seed, run_dir, log_every=5, ckpt_every=20):
    """Runs PPO, writing progress.csv and checkpoint.pkl to run_dir.
    Resumes from an existing checkpoint in run_dir."""
    os.makedirs(run_dir, exist_ok=True)
    net, init, update = make_trainer(env, cfg)
    runner = init(jax.random.PRNGKey(seed))
    ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
    start = 0
    if os.path.exists(ckpt_path):
        ck = load_checkpoint(ckpt_path)
        ts = runner.train_state.replace(params=ck["params"], opt_state=ck["opt_state"],
                                        step=ck["update"] * cfg.num_minibatches * cfg.update_epochs)
        runner = runner.replace(train_state=ts, obs_norm=ck["obs_norm"])
        start = ck["update"]
        print(f"resuming from update {start}")

    log_path = os.path.join(run_dir, "progress.csv")
    if start == 0 or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("update,env_steps,wall_s,episodes,ep_return,collision,offroad,"
                    "progress,pg_loss,v_loss,entropy,approx_kl\n")
    t0 = time.time()
    for u in range(start, cfg.num_updates):
        runner, stats = update(runner)
        stats = {k: float(v) for k, v in jax.device_get(stats).items()}
        steps = (u + 1) * cfg.batch_size
        with open(log_path, "a") as f:
            f.write(f"{u + 1},{steps},{time.time() - t0:.1f},{stats['episodes']:.0f},"
                    f"{stats['ep_return']:.3f},{stats['collision']:.4f},{stats['offroad']:.4f},"
                    f"{stats['progress']:.4f},{stats['pg_loss']:.4f},{stats['v_loss']:.4f},"
                    f"{stats['entropy']:.4f},{stats['approx_kl']:.5f}\n")
        if (u + 1) % log_every == 0:
            print(f"[{u + 1}/{cfg.num_updates}] steps={steps} ret={stats['ep_return']:.1f} "
                  f"coll={stats['collision']:.3f} off={stats['offroad']:.3f} "
                  f"prog={stats['progress']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (u + 1) % ckpt_every == 0 or u + 1 == cfg.num_updates:
            save_checkpoint(ckpt_path, runner, u + 1, cfg)
    return runner


def make_policy(ckpt, deterministic=True):
    """Returns a function obs -> normalised action from a saved checkpoint."""
    cfg = ckpt["config"]
    net = ActorCritic(hidden=cfg["hidden"], log_std_init=cfg["log_std_init"])
    params, obs_norm = ckpt["params"], ckpt["obs_norm"]
    obs_norm = ObsNorm(jnp.asarray(obs_norm.mean), jnp.asarray(obs_norm.var),
                       jnp.asarray(obs_norm.count))

    def policy(obs):
        mean, _, _ = net.apply(params, obs_norm(obs))
        return mean

    return policy
