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

v3 (2026-09-08, user order -- TOOLHANG 3-arm grid video): ``--guide orbit``
arm added (same injection path as atypical via RolloutPipeline
``guide_mode="orbit"``; TH9-5 v3 parameter set = raw dose + lam/delta/
sigma/decay/anneal/fb-soft flags), ``--guidance-scale`` / ``--gst``
overrides, ``--stock-camera`` for tasks without an agentview (tool_hang =
sideview), and the phase-1/phase-2 machinery factored into importable
functions consumed by :mod:`scout.eval.visualize_arms_video` (the 3-arm
3x3 grid driver). Single-arm CLI behavior is unchanged.

Usage (server, from the repo root; GPU must be idle):

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python -m scout.eval.visualize_chunk_video \
        --config configs/eval_can_entropy.yaml --task can \
        --base-dp-ckpt <DP-SCOUT-exp4>/checkpoints/299.ckpt \
        --vib-ckpt    <dyn-SCOUT-exp4>/<ts>/scout_vib.ckpt \
        --core-hdf5   <...>/rollout/can_core.hdf5 \
        --guide atypical --atypical-cap 2.5 --seed 42 \
        --out-dir data/vis_first_chunk/can_s233_round4

    # tool_hang orbit arm (TH9-5-s233 round5-final ckpts, r6-eval params):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python -m scout.eval.visualize_chunk_video \
        --config configs/eval_tool_hang_entropy.yaml --task tool_hang \
        --base-dp-ckpt <DP-ORBIT-exp5>/checkpoints/299.ckpt \
        --vib-ckpt    <dyn-ORBIT-exp5>/<ts>/scout_vib.ckpt \
        --core-hdf5   <...>/rollout/tool_hang_core.hdf5 \
        --guide orbit --guidance-scale 0.5 --atypical-cap 2.5 \
        --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05 \
        --orbit-sigma-decay 0.5 --orbit-noise-anneal 2 --orbit-round 6 \
        --orbit-fb-clamp soft --stock-camera sideview --max-steps 700 \
        --seed 42 --out-dir data/vis_th95_r5/orbit
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


# --------------------------------------------------------------------------- #
# reusable machinery (consumed by visualize_arms_video; single-arm main()
# below runs the exact same code path)
# --------------------------------------------------------------------------- #
def load_run_config(config_path, base_dp_ckpt, vib_ckpt, core_hdf5,
                    guidance_scale=None, gst=None):
    """eval config + the four ckpt/path overrides (same assignments the
    original inline main() did; factories read them lazily)."""
    from scout.eval.factories import load_cfg
    cfg = load_cfg(config_path)
    cfg.base_dp.initial_ckpt_path = base_dp_ckpt
    cfg.vib.ckpt_path = vib_ckpt
    cfg.vib.base_dp_ckpt = base_dp_ckpt
    cfg.dataset.path = core_hdf5
    if guidance_scale is not None:
        cfg.exploration.guidance_scale = float(guidance_scale)
    if gst is not None:
        cfg.exploration.guidance_start_timestep = int(gst)
    return cfg


def make_policy(cfg, base_dp_ckpt, vib_ckpt, guide, entropy_kwargs, device):
    """(dp, planner, chunk_len) with the SCOUT planner attached for guided
    modes. ``guide`` in {"off", "atypical", "orbit"}; orbit kwargs keys are
    exactly what RolloutPipeline._attach_planner's orbit branch reads."""
    from scout.eval.factories import (make_lpb_dp_factory,
                                      make_scout_vib_factory)
    from scout.eval.rollout_pipeline import RolloutPipeline

    dp = make_lpb_dp_factory(device)(base_dp_ckpt)
    chunk_len = int(getattr(dp, "n_action_steps", 1))
    planner = None
    if guide != "off":
        pipeline = RolloutPipeline(
            cfg=cfg, dp_factory=None, scout_vib_factory=None,
            env_factory=None, device=device, guided=True,
            guide_mode=guide,
            entropy_kwargs=entropy_kwargs)
        vib = make_scout_vib_factory(cfg, device)(vib_ckpt)
        pipeline._attach_planner(dp, vib)
        planner = dp.scout_planner
    return dp, planner, chunk_len


def make_chunk_sampler(dp, planner, device):
    """sample_chunks(obs, n) -> (n, chunk, A) numpy -- n proposals from the
    SAME obs: guided modes draw a fresh z per row (visualizes the guidance's
    per-sample spread); unguided uses the plain stochastic DDPM sampler."""
    from scout.eval.rollout_vec import _batch_obs

    guided = planner is not None
    style_dim = None
    if guided:
        style_dim = int(getattr(planner.scout_vib, "style_dim", 16))

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

    return sample_chunks


def roll_until_success(env_factory, sample_chunks, chunk_len, state0,
                       max_steps, max_attempts, seed, n_samples=10,
                       tag="[chunk_video]"):
    """Phase 1: rejection-rollout (fresh torch seed per attempt, SAME init
    state) until one episode succeeds; records the n_samples-proposal chunk
    set at every replan. Returns {replans, actions, steps}."""
    record = None
    for attempt in range(max_attempts):
        torch.manual_seed(seed + 1000 * attempt)
        env = env_factory()
        obs = env.reset_to(state0)
        replans, actions, steps, success = [], [], 0, False
        while steps < max_steps and not success:
            chunks = sample_chunks(obs, n_samples)   # (n, chunk, A)
            replans.append({"chunks": chunks.copy(),
                            "start_ee": np.asarray(
                                obs["robot0_eef_pos"], dtype=np.float64)[0]})
            for t in range(chunk_len):
                if steps >= max_steps:
                    break
                obs, r, done, info = env.step(chunks[0][t])
                actions.append(np.asarray(chunks[0][t]))
                steps += 1
                if bool(env.is_success().get("task", False)):
                    success = True
                    break
        if hasattr(env, "close"):
            env.close()
        print(f"{tag} attempt {attempt}: steps={steps} "
              f"success={success}", flush=True)
        if success:
            record = {"replans": replans, "actions": np.stack(actions),
                      "steps": steps}
            break
    if record is None:
        raise SystemExit(f"{tag} no successful episode in "
                         f"{max_attempts} attempts -- aborting")
    print(f"{tag} SUCCESS episode: {record['steps']} steps, "
          f"{len(record['replans'])} replans", flush=True)
    return record


def record_fit_points(records):
    """Union of every EE waypoint (replan anchors + all proposal paths) over
    phase-1 records -- the shared top-down fit / look-at center for
    multi-arm renders (stable, comparable scale across panels)."""
    pts = []
    for rec in records:
        for rp in rec["replans"]:
            pts.append(rp["start_ee"][None, :])
            pts.append(rp["chunks"][:, :, :3].reshape(-1, 3))
    return np.concatenate(pts, axis=0)


def render_replay(env_factory, state0, record, chunk_len, views, out_dir,
                  res, fps, step_frames, label, tag="[chunk_video]"):
    """Phase 2: deterministic replay of the recorded actions + rendering.
    ``views`` = build_views() output (SHARED camera poses across arms when
    the caller passes the same dict). Writes PNG frames under
    ``out_dir/frames/<view>/NNNNN.png`` (plus the 1 s SUCCESS hold) and
    returns the per-view frame counts. Encoding is left to the caller
    (:func:`encode_mp4`)."""
    from scout.eval.visualize_trajectories import (
        build_views, project_to_pixels_frame, render_view,
    )

    env = env_factory()
    env.reset_to(state0)
    renv = env.env                 # robomimic EnvRobosuite render surface
    if views is None:
        views = build_views(renv.env.sim, fit_points=None)

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

    def render_frame(props, text):
        """Render every view once at the CURRENT sim state; return dict."""
        frames = {}
        for vname, view in views.items():
            img, cx, cm, fov = render_view(renv, view, res)
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
            text = (f"{label} | replan {replan_i} step {step_i}/"
                    f"{record['steps']}"
                    + (" | SUCCESS" if succ else ""))
            frames = render_frame(props, text)
            for _ in range(step_frames):
                for v, fr in frames.items():
                    save(v, fr)
        replan_i += 1
        print(f"{tag} rendered replan {replan_i}/{len(record['replans'])}"
              f" step {step_i}", flush=True)

    # success tail: hold the final state briefly
    props = []          # bare frames
    for _ in range(fps):
        frames = render_frame(props,
                              f"{label} | SUCCESS @ step {record['steps']}")
        for v, fr in frames.items():
            save(v, fr)

    if hasattr(env, "close"):
        env.close()
    return counters


def encode_mp4(frames_dir, mp4, fps, quiet=False):
    """ffmpeg PNG sequence -> mp4 (libx264; even-sized frames required)."""
    os.makedirs(os.path.dirname(mp4) or ".", exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-framerate", str(fps),
         "-i", os.path.join(frames_dir, "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
        check=True)
    if not quiet:
        print(f"[chunk_video] saved {mp4}", flush=True)


# --------------------------------------------------------------------------- #
# single-arm CLI (original behavior; see module docstring for examples)
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--base-dp-ckpt", required=True)
    p.add_argument("--vib-ckpt", required=True)
    p.add_argument("--core-hdf5", required=True)
    p.add_argument("--guide", choices=["off", "atypical", "orbit"],
                   default="atypical")
    p.add_argument("--atypical-cap", type=float, default=2.5)
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="override cfg.exploration.guidance_scale (raw dose)")
    p.add_argument("--gst", type=int, default=None,
                   help="override cfg.exploration.guidance_start_timestep")
    p.add_argument("--orbit-lam", type=float, default=0.5)
    p.add_argument("--orbit-delta", type=float, default=0.25)
    p.add_argument("--orbit-sigma", type=float, default=0.25)
    p.add_argument("--orbit-sigma-decay", type=float, default=1.0)
    p.add_argument("--orbit-noise-anneal", type=float, default=1.0)
    p.add_argument("--orbit-round", type=int, default=1)
    p.add_argument("--orbit-fb-clamp", choices=["none", "soft"],
                   default="none")
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
    p.add_argument("--stock-camera", default="agentview",
                   help="stock camera name (tool_hang: sideview)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cuda-visible-devices", default=None)
    args = p.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scout.eval.factories import make_default_env_factory
    from scout.eval.visualize_trajectories import build_views

    cfg = load_run_config(args.config, args.base_dp_ckpt, args.vib_ckpt,
                          args.core_hdf5, args.guidance_scale, args.gst)

    np.random.seed(args.seed)

    ek = {"atypical_cap": args.atypical_cap}
    if args.guide == "orbit":
        ek.update({"orbit_lam": args.orbit_lam,
                   "orbit_delta": args.orbit_delta,
                   "orbit_sigma": args.orbit_sigma,
                   "orbit_sigma_decay": args.orbit_sigma_decay,
                   "orbit_noise_anneal": args.orbit_noise_anneal,
                   "orbit_round": args.orbit_round,
                   "orbit_fb_clamp": args.orbit_fb_clamp})
    dp, planner, chunk_len = make_policy(
        cfg, args.base_dp_ckpt, args.vib_ckpt, args.guide, ek, device)
    sample_chunks = make_chunk_sampler(dp, planner, device)
    env_factory = make_default_env_factory(cfg)

    arm = "guided" if args.guide != "off" else "unguided"
    print(f"[chunk_video] arm={arm} guide={args.guide} n={args.n_samples} "
          f"chunk={chunk_len} max_steps={args.max_steps} device={device}",
          flush=True)

    # phase 0: shared init state from the seed-fixed scene
    env = env_factory()
    env.reset(seed=args.seed)
    state0 = env.get_state()
    if hasattr(env, "close"):
        env.close()

    record = roll_until_success(env_factory, sample_chunks, chunk_len,
                                state0, args.max_steps, args.max_attempts,
                                args.seed, n_samples=args.n_samples)

    env = env_factory()
    env.reset_to(state0)
    renv = env.env
    views = build_views(renv.env.sim, fit_points=None,
                        stock_camera=args.stock_camera)
    if hasattr(env, "close"):
        env.close()
    views = {k: views[k] for k in args.views.split(",")}

    out_dir = os.path.join(args.out_dir, f"video_{arm}")
    counters = render_replay(env_factory, state0, record, chunk_len, views,
                             out_dir, args.res, args.fps, args.step_frames,
                             label=arm)
    for v in views:
        encode_mp4(os.path.join(out_dir, "frames", v),
                   os.path.join(out_dir, f"chunk_video_{v}.mp4"), args.fps)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"arm": arm, "guide": args.guide, "seed": args.seed,
                   "n_samples": args.n_samples, "chunk_len": chunk_len,
                   "steps": record["steps"],
                   "replans": len(record["replans"]),
                   "success": True, "frames": counters}, f, indent=2)
    print(f"[chunk_video] done: success episode {record['steps']} steps, "
          f"{len(record['replans'])} replans", flush=True)


if __name__ == "__main__":
    main()
