"""Multi-view version of the first-action-chunk visualization (user
2026-08-27 task 1).

Reads the paths saved by ``visualize_first_chunk`` (``paths.npz``:
``<arm>_exec_<i>`` executed EE paths + ``<arm>_prop_<i>`` commanded proposal
waypoints) and re-renders the SAME data from every implemented camera view
(:func:`visualize_trajectories.build_views`: stock agentview, fitted top-down,
front, side_left, side_right, corner45, gripper eye-in-hand). One 2x3 figure
per view, same layout as ``first_chunk_viz.png``:

    (0,0) scene in this view   (0,1) unguided executed  (0,2) guided executed
    (1,0) bare view (no lines) (1,1) unguided proposals (1,2) guided proposals

No policy / no GPU model work -- pure re-rendering of saved paths.

Usage (server, from the repo root):

    MUJOCO_GL=egl python -m scout.eval.visualize_multiview \
        --config configs/eval_can_entropy.yaml \
        --core-hdf5 <...>/rollout/can_core.hdf5 --seed 42 \
        --paths-dir data/vis_first_chunk/can_s233_round4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--core-hdf5", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paths-dir", required=True,
                   help="dir containing paths.npz from visualize_first_chunk")
    p.add_argument("--res", type=int, default=768)
    p.add_argument("--views", default=None,
                   help="comma-separated subset of views (default: all)")
    args = p.parse_args()

    from scout.eval.factories import load_cfg, make_default_env_factory
    from scout.eval.visualize_trajectories import (
        add_gradient_colorbar, build_views, draw_gradient_path, make_env,
        mark_endpoints, project_to_pixels_frame, render_view,
        trajectory_cmap,
    )

    d = np.load(os.path.join(args.paths_dir, "paths.npz"))
    arms = {}
    for lbl in ("unguided", "guided"):
        for kind in ("exec", "prop"):
            ks = sorted((k for k in d.files
                         if k.startswith(f"{lbl}_{kind}_")),
                        key=lambda s: int(s.rsplit("_", 1)[1]))
            if not ks:
                raise SystemExit(f"paths.npz has no {lbl}_{kind}_* entries")
            arms.setdefault(lbl, {})[kind] = [d[k] for k in ks]
    print(f"[multiview] loaded {len(arms['unguided']['exec'])} paths/arm "
          f"from {args.paths_dir}")

    cfg = load_cfg(args.config)
    cfg.dataset.path = args.core_hdf5
    ef = make_default_env_factory(cfg)
    env0 = ef()
    env0.reset(seed=args.seed)
    state0 = env0.get_state()
    if hasattr(env0, "close"):
        env0.close()

    venv = make_env(args.core_hdf5)
    venv.reset_to({"states": np.asarray(state0, dtype=np.float64)})
    sim = venv.env.sim

    fit = np.concatenate([q for a in arms.values() for q in
                          a["exec"] + a["prop"]], axis=0)
    views = build_views(sim, fit_points=fit)
    if args.views:
        keep = set(args.views.split(","))
        views = {k: v for k, v in views.items() if k in keep}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = trajectory_cmap()
    for vname, view in views.items():
        img, cam_xpos, cam_xmat, fov = render_view(venv, view, args.res)
        h_px, w_px = img.shape[:2]

        def proj(pts):
            return project_to_pixels_frame(pts, cam_xpos, cam_xmat, fov,
                                           w_px, h_px)

        fig, axes = plt.subplots(2, 3, figsize=(19.2, 13.5),
                                 gridspec_kw={"width_ratios":
                                              [1.0, 1.25, 1.25]})
        # left column: the bare scene in this view (top) + bare view w/o
        # lines (bottom, reference); the four data panels to the right
        for r in (0, 1):
            axes[r][0].imshow(img)
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[0][0].set_title(f"scene: {vname}")
        axes[1][0].set_title(f"bare view: {vname} (no overlay)", fontsize=11)
        for r, kind, what in ((0, "exec", "executed (sim, 8 env steps)"),
                              (1, "prop", "action proposals (commanded)")):
            for c, lbl in enumerate(("unguided", "guided"), start=1):
                ax = axes[r][c]
                ax.imshow(img)
                ax.set_xlim(0, w_px); ax.set_ylim(h_px, 0)
                ax.set_xticks([]); ax.set_yticks([])
                for q in arms[lbl][kind]:
                    px = proj(q)
                    draw_gradient_path(ax, px, cmap, lw=1.7, alpha=0.9)
                    mark_endpoints(ax, px[0], px[-1])
                ax.set_title(f"{lbl}, {what}", fontsize=11)
        add_gradient_colorbar(fig, axes[1][2], cmap)
        fig.suptitle(f"view = {vname} -- first-action-chunk, seed "
                     f"{args.seed}, 20 draws/arm", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(args.paths_dir, f"first_chunk_viz_{vname}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[multiview] saved {out}")
    if hasattr(venv, "close"):
        venv.close()


if __name__ == "__main__":
    main()
