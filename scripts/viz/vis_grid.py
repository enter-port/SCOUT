"""Merge each experiment's three vis_final runs into ONE 2x2 grid figure.

No rollouts: reads the existing all.hdf5s (core 20 + 20 rollouts each),
re-renders the shared initial scene ONCE (agentview + top-down), and overlays
each run's 20 EE trajectories on the SAME top-down camera (fitted over the
union of all three runs -> directly comparable pixel scale across panels).

Layout per experiment:
  (0,0) initial scene (agentview)     (0,1) g1_guide_on  (guided)
  (1,0) g1_guide_off (SCOUT DP plain) (1,1) g2_DP       (DP-arm DP)

Writes <exp>/trajviz_grid.png next to the existing per-run trajviz.png
(which are left untouched).

Run: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python soe_scripts/vis_grid.py [exp_dir ...]
"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scout.eval.visualize_trajectories import (
    EEF_KEY,
    add_gradient_colorbar,
    draw_gradient_path,
    make_env,
    mark_endpoints,
    project_to_pixels,
    render_initial_views,
    trajectory_cmap,
)

REPO = "/root/workspace/baojiachun/scout"
ROOT = os.path.join(REPO, "data", "vis_final")
RUNS = [("g1_guide_on", "guide ON (guided)"),
        ("g1_guide_off", "guide OFF (SCOUT DP plain)"),
        ("g2_DP", "DP-arm DP (plain)")]


def load_run(exp_dir, run_dir):
    """(eef list, per-traj success, avg_jerk, first appended demo's state0)."""
    d = os.path.join(ROOT, exp_dir, run_dir)
    h5path = os.path.join(d, "all.hdf5")
    trajs, succ = [], []
    with h5py.File(h5path, "r") as f:
        for k in sorted((x for x in f["data"].keys()
                         if str(x).startswith("demo")
                         and int(str(x).split("_")[1]) >= 20),
                        key=lambda s: int(str(s).split("_")[1])):
            trajs.append(np.asarray(f["data"][k][EEF_KEY][()], dtype=np.float64))
            succ.append(bool(f["data"][k]["success"][0]))
        state0 = np.asarray(f["data/demo_20"]["states"][0])
    jerk = None
    for jf in glob.glob(os.path.join(d, "log", "*.json")):
        j = json.load(open(jf))
        jerk = j.get("avg_jerk")
    return trajs, succ, jerk, state0, h5path


def main():
    exps = sys.argv[1:] or sorted(
        d for d in os.listdir(ROOT)
        if d.startswith("exp") and os.path.isdir(os.path.join(ROOT, d))
        and "-round5" not in d)
    cmap = trajectory_cmap()
    for exp in exps:
        runs, fit = [], []
        state0 = h5path = None
        for run_dir, label in RUNS:
            trajs, succ, jerk, s0, hp = load_run(exp, run_dir)
            runs.append([label, trajs, succ, jerk])
            fit.extend(trajs)
            if state0 is None:
                state0, h5path = s0, hp
        fit = np.concatenate(fit, axis=0)

        env = make_env(h5path)
        img_agent, img_top, cam_pos, fov = render_initial_views(
            env, state0, fit, res=768)
        if hasattr(env, "close"):
            env.close()
        h_px, w_px = img_top.shape[:2]
        for r in runs:
            r.append([project_to_pixels(t, cam_pos, fov, w_px, h_px)
                      for t in r[1]])

        fig, axes = plt.subplots(2, 2, figsize=(13.6, 13.6))
        axes[0][0].imshow(img_agent)
        axes[0][0].set_title("initial scene (agentview, demo_20)", fontsize=12)
        slots = [(0, 1), (1, 0), (1, 1)]
        last_ax = None
        for (i, j), (label, trajs, succ, jerk, px) in zip(slots, runs):
            ax = axes[i][j]
            ax.imshow(img_top)
            ax.set_xlim(0, w_px)
            ax.set_ylim(h_px, 0)
            for p in px:
                draw_gradient_path(ax, p, cmap, lw=1.7, alpha=0.9)
                mark_endpoints(ax, p[0], p[-1])
            sub = f"{sum(succ)}/{len(succ)} succ"
            if jerk is not None:
                sub += f", jerk {jerk:.2f}"
            ax.set_title(f"{label} -- {sub}", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            last_ax = ax
        add_gradient_colorbar(fig, last_ax, cmap)
        fig.suptitle(f"{exp} -- seed-42 scenes 42..61, 20 rollouts each "
                     "(yellow=start, magenta=end)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(ROOT, exp, "trajviz_grid.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
