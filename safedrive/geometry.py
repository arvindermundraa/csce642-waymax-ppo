"""Route geometry helpers (all jit-friendly)."""
import jax.numpy as jnp

from safedrive.data import ROUTE_SPACING


def wrap_angle(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def to_ego_frame(dx, dy, yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def project_to_route(route, route_valid, x, y):
    """Projects a point onto the resampled route.

    Returns arc length s, signed lateral offset (left positive), the route
    heading at the projection and the index of the nearest route point.
    """
    n = route.shape[0]
    d2 = (route[:, 0] - x) ** 2 + (route[:, 1] - y) ** 2
    d2 = jnp.where(route_valid, d2, jnp.inf)
    i = jnp.argmin(d2)
    i0 = jnp.clip(i, 0, n - 2)
    i0 = jnp.where(route_valid[jnp.clip(i0 + 1, 0, n - 1)], i0, jnp.maximum(i0 - 1, 0))
    p0, p1 = route[i0], route[i0 + 1]
    t = p1 - p0
    t = t / (jnp.linalg.norm(t) + 1e-6)
    rel = jnp.array([x, y]) - p0
    along = jnp.clip(rel @ t, -ROUTE_SPACING, 2 * ROUTE_SPACING)
    lat = t[0] * rel[1] - t[1] * rel[0]
    s = i0 * ROUTE_SPACING + along
    heading = jnp.arctan2(t[1], t[0])
    return s, lat, heading, i


def point_at(route, route_valid, s):
    """Route point at arc length s (clamped to the valid part)."""
    last = jnp.sum(route_valid) - 1
    f = jnp.clip(s / ROUTE_SPACING, 0.0, last.astype(jnp.float32))
    i0 = jnp.floor(f).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, last)
    w = f - i0
    return route[i0] * (1 - w) + route[i1] * w


def curvature_at(route_kappa, s):
    i = jnp.clip(jnp.round(s / ROUTE_SPACING).astype(jnp.int32), 0, route_kappa.shape[0] - 1)
    return route_kappa[i]
