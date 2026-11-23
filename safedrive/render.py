"""Top-down rendering of an episode for the videos in the report.

We draw our own frames instead of using waymax.visualization because its
figure-to-array helper relies on a matplotlib API that newer versions removed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

# Waymax roadgraph type ids for road edges (boundary=15, median=16)
ROAD_EDGE_TYPES = (15, 16)


def _box(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                    [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([x, y])


def draw_frame(sim, route, title="", status="", view=45.0, dpi=80):
    """Returns an RGB array for one simulator state (numpy pytree)."""
    t = int(sim.timestep)
    traj = sim.sim_trajectory
    sdc = int(np.argmax(sim.object_metadata.is_sdc))
    ex, ey = float(traj.x[sdc, t]), float(traj.y[sdc, t])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    rg = sim.roadgraph_points
    v = np.asarray(rg.valid)
    edge = v & np.isin(np.asarray(rg.types), ROAD_EDGE_TYPES)
    lane = v & ~edge
    ax.scatter(rg.x[lane], rg.y[lane], s=0.5, c="#c8c8c8")
    ax.scatter(rg.x[edge], rg.y[edge], s=1.0, c="#555555")
    ax.plot(route[:, 0], route[:, 1], "--", lw=1, c="#3b7dd8", alpha=0.6)

    for i in range(traj.x.shape[0]):
        if not traj.valid[i, t]:
            continue
        box = _box(traj.x[i, t], traj.y[i, t], traj.yaw[i, t],
                   traj.length[i, t], traj.width[i, t])
        color = "#1f5fbf" if i == sdc else "#9a9a9a"
        ax.add_patch(Polygon(box, closed=True, fc=color, ec="k", lw=0.5,
                             alpha=0.95 if i == sdc else 0.8))

    ax.set_xlim(ex - view, ex + view)
    ax.set_ylim(ey - view, ey + view)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if status:
        ax.text(0.02, 0.97, status, transform=ax.transAxes, va="top", fontsize=9,
                color="#b00020" if "COLLISION" in status or "OFF-ROAD" in status else "k")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
