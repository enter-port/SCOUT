"""First-chunk proposal-sampling VIDEO, v2 (user 2026-08-27 feedback).

Spec (verbatim from the user's 5 requirements):

* views: agentview / eye_in_hand / topdown ONLY (--views to subset);
* the rendered episode MUST be a SUCCESS: phase 1 rolls out episodes
  (re-seeding torch per attempt) until one succeeds, sampling at every
  replan N=10 chunks (the executed sample #0 + 9 extra proposals); phase 2
  replays that successful episode action-by-action and renders;
* smooth playback: no frozen holds -- every env step emits ``step_frames``
  consecutive video frames (default 3 @ 24 fps), the overlay drawn on every
  frame;
* the current replan's 10 chunk trajectories (executed one in WHITE, the 9
  proposals in the time gradient) stay on screen for the whole chunk,
  including while the arm executes it.

Phase-2 replay re-executes the recorded actions from the same init state
(MuJoCo is deterministic, so the trajectory is identical) -- no policy calls
while rendering.

Usage (server, from the repo root; GPU must be idle):

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python -m scout.eval.visualize_chunk_video \
        --config configs/eval_can_entropy.yaml --task can \
        --base-dp-ckpt <DP-SCOUT-exp4>/checkpoints/299.ckpt \
        --vib-ckpt    <dyn-SCOUT-exp4>/<ts>/scout_vib.ckpt \
        --core-hdf5   <...>/rollout/can_core.hdf5 \
        --guide atypical --atypical-cap 2.5 --seed 42 \
        --out-dir data/vis_first_chunk/can_s233_round4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image

_CMAP = None


def _cmap():
    global _CMAP
    if _CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap
        _CMAP = LinearSegmentedColormap.from_list(
            "y_o_r_m", ["#FFDF3F", "#FF8A00", "#F51B07", "#C000A0"])
    return _CMAP


def _burn(img, polylines, text):
    """One video frame: scene render + overlaid chunk paths + caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cmap = _cmap()
    fig = plt.figure(figsize=(img.shape[1] / 100.0, img.shape[0] / 100.0),
                     dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    for i, px in enumerate(polylines):
        ok = np.isfinite(px).all(axis=1)
        px = px[ok]
        if len(px) < 2:
            continue
        if i == 0:      # the sample that actually executes
            ax.plot(px[:, 0], px[:, 1], color="white", lw=3.2,
                    alpha=0.95, zorder=4)
            ax.scatter(*px[0], s=42, c="white", edgecolors="black",
                       linewidths=0.6, zorder=5)
            ax.scatter(*px[-1], s=70, c="white", marker="X",
                       edgecolors="black", linewidths=0.6, zorder=5)
        else:
            segs = np.stack([px[:-1], px[1:]], axis=1)
            colors = cmap(np.linspace(0.0, 1.0, len(segs)))
            ax.add_collection(LineCollection(segs, colors=colors,
                                             linewidths=1.9, alpha=0.85,
                                             capstyle="round", zorder=3))
    ax.text(0.012, 0.024, text, transform=ax.transAxes, color="white",
            fontsize=9, va="bottom",
            bbox=dict(facecolor="black", alpha=0.55, pad=2.5,
                      edgecolor="none"))
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--base-dp-ckpt", required=True)
    p.add_argument("--vib-ckpt", required=True)
    p.add_argument("--core-hdf5", required=True)
    p.add_argument("--guide", choices=["off", "atypical"], default="atypical")
    p.add_argument("--atypical-cap", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--max-attempts", type=int, default=30,
                   help="phase-1 rejection cap: episodes re-rolled (fresh "
                        "torch seed) until one succeeds")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--step-frames", type=int, default=3,
                   help="video frames per executed env step (smoothness)")
    p.add_argument("--views", default="agentview,topdown,eye_in_hand")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cuda-visible-devices", default=None)
    args = p.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scout.eval.factories import (
        load_cfg, make_default_env_factory, make_lpb_dp_factory,
        make_scout_vib_factory,
    )
    from scout.eval.rollout_vec import _batch_obs
    from scout.eval.visualize_trajectories import (
        build_views, project_to_pixels_frame, render_view,
    )

    cfg = load_cfg(args.config)
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
    cfg.vib.ckpt_path = args.vib_ckpt
    cfg.vib.base_dp_ckpt = args.base_dp_ckpt
    cfg.dataset.path = args.core_hdf5

    np.random.seed(args.seed)

    dp = make_lpb_dp_factory(device)(args.base_dp_ckpt)
    chunk_len = int(getattr(dp, "n_action_steps", 1))
    env_factory = make_default_env_factory(cfg)

    guided = args.guide != "off"
    if guided:
        from scout.eval.rollout_pipeline import RolloutPipeline
        vib = make_scout_vib_factory(cfg, device)(args.vib_ckpt)
        pipeline = RolloutPipeline(
            cfg=cfg, dp_factory=None, scout_vib_factory=None,
            env_factory=None, device=device, guided=True,
            guide_mode=args.guide,
            entropy_kwargs={"atypical_cap": args.atypical_cap})
        pipeline._attach_planner(dp, vib)
        planner = dp.scout_planner
        style_dim = int(getattr(planner.scout_vib, "style_dim", 16))
    arm = "guided" if guided else "unguided"
    print(f"[chunk_video] arm={arm} n={args.n_samples} chunk={chunk_len} "
          f"max_steps={args.max_steps} device={device}", flush=True)

    # ------------------------------------------------------------------ #
    # phase 1: rejection-rollout until SUCCESS, recording chunks per replan
    # ------------------------------------------------------------------ #
    env = env_factory()
    obs = env.reset(seed=args.seed)
    state0 = env.get_state()
    if hasattr(env, "close"):
        env.close()

    def sample_chunks(obs, n):
        obs_batch = _batch_obs([obs] * n, device)
        if guided:
            planner.set_z(torch.randn(n, style_dim, device=device,
                                      dtype=torch.float32))
            if hasattr(planner, "set_row_context"):
                planner.set_row_context([0] * n)
            # no no_grad: the guided denoise loop needs autograd
            return dp.predict_action_dyn_guided(obs_batch)["action"] \
                .detach().cpu().numpy()
        with torch.no_grad():
            return dp.predict_action(obs_batch)["action"].detach().cpu().numpy()

    record = None
    for attempt in range(args.max_attempts):
        torch.manual_seed(args.seed + 1000 * attempt)
        env = env_factory()
        obs = env.reset_to(state0)
        replans, actions, steps, success = [], [], 0, False
        while steps < args.max_steps and not success:
            chunks = sample_chunks(obs, args.n_samples)   # (n, chunk, 10)
            replans.append({"chunks": chunks.copy(),
                            "start_ee": np.asarray(
                                obs["robot0_eef_pos"], dtype=np.float64)[0]})
            for t in range(chunk_len):
                if steps >= args.max_steps:
                    break
                obs, r, done, info = env.step(chunks[0][t])
                actions.append(np.asarray(chunks[0][t]))
                steps += 1
                if bool(env.is_success().get("task", False)):
                    success = True
                    break
        if hasattr(env, "close"):
            env.close()
        print(f"[chunk_video] attempt {attempt}: steps={steps} "
              f"success={success}", flush=True)
        if success:
            record = {"replans": replans, "actions": np.stack(actions),
                      "steps": steps}
            break
    if record is None:
        raise SystemExit(f"[chunk_video] no successful episode in "
                         f"{args.max_attempts} attempts -- aborting")
    print(f"[chunk_video] SUCCESS episode: {record['steps']} steps, "
          f"{len(record['replans'])} replans", flush=True)

    # ------------------------------------------------------------------ #
    # phase 2: deterministic replay of the recorded actions + rendering
    # ------------------------------------------------------------------ #
    env = env_factory()
    env.reset_to(state0)
    renv = env.env                 # robomimic EnvRobosuite render surface
    views = build_views(renv.env.sim, fit_points=None)  # stable box fit
    views = {k: views[k] for k in args.views.split(",")}

    out_dir = os.path.join(args.out_dir, f"video_{arm}")
    fr_root = os.path.join(out_dir, "frames")
    shutil.rmtree(fr_root, ignore_errors=True)
    fdirs = {v: os.path.join(fr_root, v) for v in views}
    for d in fdirs.values():
        os.makedirs(d, exist_ok=True)
    counters = {v: 0 for v in views}

    def save(v, frame):
        Image.fromarray(frame).save(
            os.path.join(fdirs[v], f"{counters[v]:05d}.png"))
        counters[v] += 1

    def render_frame(views, props, text):
        """Render every view once at the CURRENT sim state; return dict."""
        frames = {}
        for vname, view in views.items():
            img, cx, cm, fov = render_view(renv, view, args.res)
            h_px, w_px = img.shape[:2]
            plines = [project_to_pixels_frame(q, cx, cm, fov, w_px, h_px)
                      for q in props]
            frames[vname] = _burn(img, plines, text)
        return frames

    step_i, replan_i = 0, 0
    for rp in record["replans"]:
        props = [np.concatenate([rp["start_ee"][None, :],
                                 c[:, :3]], axis=0)
                 for c in rp["chunks"]]
        for t in range(chunk_len):
            if step_i >= len(record["actions"]):
                break
            env.step(record["actions"][step_i])
            step_i += 1
            succ = step_i >= record["steps"]
            text = (f"{arm} | replan {replan_i} step {step_i}/"
                    f"{record['steps']}"
                    + (" | SUCCESS" if succ else ""))
            frames = render_frame(views, props, text)
            for _ in range(args.step_frames):
                for v, fr in frames.items():
                    save(v, fr)
        replan_i += 1
        print(f"[chunk_video] rendered replan {replan_i}/{len(record['replans'])}"
              f" step {step_i}", flush=True)

    # success tail: hold the final state briefly
    props = []          # bare frames
    for _ in range(args.fps):
        frames = render_frame(views, props,
                              f"{arm} | SUCCESS @ step {record['steps']}")
        for v, fr in frames.items():
            save(v, fr)

    for v in views:
        mp4 = os.path.join(out_dir, f"chunk_video_{v}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", str(args.fps),
             "-i", os.path.join(fdirs[v], "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
            check=True)
        print(f"[chunk_video] saved {mp4}", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"arm": arm, "guide": args.guide, "seed": args.seed,
                   "n_samples": args.n_samples, "chunk_len": chunk_len,
                   "steps": record["steps"],
                   "replans": len(record["replans"]),
                   "success": True, "frames": counters}, f, indent=2)
    print(f"[chunk_video] done: success episode {record['steps']} steps, "
          f"{replan_i} replans", flush=True)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
