"""Small patches so Waymax (last released for JAX 0.4.x) runs on current JAX.

Import this module before anything from waymax.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# JAX 0.6 removed the tree_* aliases from the top-level namespace.
for _name in ("tree_map", "tree_leaves", "tree_flatten", "tree_unflatten",
              "tree_structure", "tree_transpose"):
    if not hasattr(jax, _name):
        setattr(jax, _name, getattr(jtu, _name))

# Newer jnp.cross no longer accepts 2D vectors; Waymax's off-road check uses them.
_orig_cross = jnp.cross


def _cross(a, b, *args, **kwargs):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape[-1] == 2 and b.shape[-1] == 2:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return _orig_cross(a, b, *args, **kwargs)


if getattr(jnp.cross, "__name__", "") != "_cross":
    jnp.cross = _cross
