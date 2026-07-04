"""Render per-object trajectory plots from a 4DSG JSON file.

For each object with >= 2 observations, generates a compact XZ (bird's-eye)
trajectory image with time-encoded colour (blue→red) and start/end markers.

Usage:
    python tests/test_trajectory_plot.py [path/to/4dsg.json] [output_dir]

Defaults:
    4dsg.json  = assets/diag_youtube000/4dsg.json
    output_dir = assets/diag_youtube000/trajectory_plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── Coordinate system ──
# X = right, Y = down, Z = forward (away from camera)
# Bird's-eye view = XZ plane (X → right, Z → up in plot = forward in world)


def _offset_label(ax, x, y, xs, zs, text, fontsize=7, color="black"):
    """Place a text label near (x, y), nudged away from the trajectory centre."""
    cx = float(np.mean(xs))
    cz = float(np.mean(zs))
    rng = max(xs.max() - xs.min(), zs.max() - zs.min(), 1e-6)
    # Nudge direction: away from centroid of trajectory
    dx = x - cx
    dz = y - cz
    d = np.sqrt(dx**2 + dz**2)
    if d < 1e-8:
        dx, dz = 0.0, 1.0
        d = 1.0
    offset = 0.08 * rng
    ox = x + dx / d * offset
    oz = y + dz / d * offset
    ax.annotate(
        text, xy=(x, y), xytext=(ox, oz),
        fontsize=fontsize, fontweight="bold", color=color, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
        zorder=10,
    )


def render_trajectory_xz(
    pts: np.ndarray,
    ts: np.ndarray,
    obj_id: int,
    out_path: Path,
    img_size: int = 224,
) -> Path:
    """Render a single object's XZ trajectory plot.

    Args:
        pts: (N, 3) world-frame centroids
        ts: (N,) timestamps in seconds
        obj_id: object ID for title
        out_path: save path (.png)
        img_size: pixel size of output image

    Returns:
        out_path
    """
    xs = pts[:, 0]   # X = right/left
    zs = pts[:, 2]   # Z = forward/backward

    # Normalise timestamps to [0, 1] for colourmap
    t_norm = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-6)

    fig, ax = plt.subplots(figsize=(img_size / 100, img_size / 100), dpi=100)

    # Colour map: blue (early) → red (late)
    cmap = plt.cm.coolwarm

    # Draw connecting lines with time-encoded colour
    for i in range(len(xs) - 1):
        colour = cmap(t_norm[i])
        ax.plot(
            [xs[i], xs[i + 1]], [zs[i], zs[i + 1]],
            color=colour, linewidth=1.5, zorder=1,
        )

    # Scatter points with time colour
    ax.scatter(
        xs, zs, c=t_norm, cmap="coolwarm", s=30, edgecolors="black",
        linewidths=0.5, zorder=2,
    )

    # Start marker (green triangle)
    ax.scatter(xs[0], zs[0], marker="^", s=90, color="green",
               edgecolors="black", linewidths=0.8, zorder=5)

    # End marker (red square)
    ax.scatter(xs[-1], zs[-1], marker="s", s=90, color="red",
               edgecolors="black", linewidths=0.8, zorder=5)

    # Arrow from second-to-last to last point (direction indicator)
    if len(xs) >= 2:
        dx = xs[-1] - xs[-2]
        dz = zs[-1] - zs[-2]
        norm = np.sqrt(dx**2 + dz**2)
        if norm > 1e-6:
            rng = max(xs.max() - xs.min(), zs.max() - zs.min(), 0.01)
            scale = 0.12 * rng / norm
            ax.annotate(
                "", xy=(xs[-1] + dx * scale, zs[-1] + dz * scale),
                xytext=(xs[-1], zs[-1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                zorder=4,
            )

    # Axis labels and formatting
    ax.set_xlabel("X (right →)", fontsize=7)
    ax.set_ylabel("Z (forward →)", fontsize=7)
    # Title line 1: Object ID.  Line 2: green=start time, red=end time
    ax.set_title(
        f"Object {obj_id}\n"
        f"\u25B2 start {ts[0]:.1f}s   \u25A0 end {ts[-1]:.1f}s   blue→red = time",
        fontsize=6,
    )
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return out_path


def render_all_trajectories(dsg_path: Path, out_dir: Path, img_size: int = 224):
    """Render trajectory plots for all objects in a 4DSG file."""
    with open(dsg_path) as f:
        dsg = json.load(f)

    tracks = dsg.get("tracks", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    skipped = []

    for track in tracks:
        oid = track["object_id"]
        fk = track["F_k"]

        if len(fk) < 2:
            skipped.append((oid, len(fk), "too few points"))
            continue

        pts = np.array([[o["c"][0], o["c"][1], o["c"][2]] for o in fk])
        ts = np.array([o["t"] for o in fk])

        # Check if effectively stationary (path too short to visualise)
        path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if path_length < 1e-4:
            skipped.append((oid, len(fk), "zero path length"))
            continue

        out_path = out_dir / f"traj_obj_{oid:04d}.png"
        render_trajectory_xz(pts, ts, oid, out_path, img_size)
        rendered.append((oid, len(fk), out_path))

    return rendered, skipped


def main():
    root = Path(__file__).resolve().parent.parent

    if len(sys.argv) >= 2:
        dsg_path = Path(sys.argv[1])
    else:
        dsg_path = root / "assets" / "diag_youtube000" / "4dsg.json"

    if len(sys.argv) >= 3:
        out_dir = Path(sys.argv[2])
    else:
        out_dir = dsg_path.parent / "trajectory_plots"

    if not dsg_path.exists():
        print(f"ERROR: {dsg_path} does not exist. Run the pipeline first.")
        sys.exit(1)

    print(f"Input:  {dsg_path}")
    print(f"Output: {out_dir}")
    print()

    rendered, skipped = render_all_trajectories(dsg_path, out_dir)

    for oid, n_obs, path in rendered:
        print(f"  Object {oid:3d}: {n_obs:3d} obs → {path.name}")
    for oid, n_obs, reason in skipped:
        print(f"  Object {oid:3d}: {n_obs:3d} obs — SKIPPED ({reason})")

    print(f"\nRendered: {len(rendered)},  Skipped: {len(skipped)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
