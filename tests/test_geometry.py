import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safedrive import compat  # noqa: E402,F401
from safedrive import geometry as geo  # noqa: E402
from safedrive.data import ROUTE_POINTS  # noqa: E402


def straight_route():
    x = np.arange(ROUTE_POINTS, dtype=np.float32)
    route = np.stack([x, np.zeros_like(x)], -1)
    return jnp.asarray(route), jnp.ones(ROUTE_POINTS, bool)


def test_projection_on_straight_route():
    route, valid = straight_route()
    s, lat, heading, _ = geo.project_to_route(route, valid, 12.3, 1.5)
    assert abs(float(s) - 12.3) < 1e-3
    assert abs(float(lat) - 1.5) < 1e-3       # left of the route is positive
    assert abs(float(heading)) < 1e-6


def test_point_at_and_ego_frame():
    route, valid = straight_route()
    p = geo.point_at(route, valid, 7.5)
    assert np.allclose(np.asarray(p), [7.5, 0.0])
    # a point straight ahead of a car facing +y should land on the ego x axis
    fx, fy = geo.to_ego_frame(0.0, 5.0, np.pi / 2)
    assert abs(float(fx) - 5.0) < 1e-5 and abs(float(fy)) < 1e-5


def test_cross_patch_handles_2d():
    a, b = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    assert float(jnp.cross(a, b)) == 1.0
