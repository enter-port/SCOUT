"""3-arm 3x3 grid trajectory VIDEO (user order 2026-09-08).

Renders ONE seed-fixed scene through three policies -- rows, in order:

    DP          : the DP-baseline arm's ckpt, guidance OFF
    SCOUT       : the atypical arm's ckpt + its dyn, guide_mode="atypical"
    SCOUT-orbit : the orbit arm's ckpt + its dyn, guide_mode="orbit"

-- with the SAME columns (views; tool_hang default = sideview / topdown /
eye_in_hand). Every arm runs the visualize_chunk_video machinery verbatim:
phase-1 rejection-rollout until SUCCESS (fresh torch seed per attempt, the
SAME init state for all arms = the seed-42 scene), sampling 10 chunks at
every replan; phase-2 deterministic replay + overlay render (executed chunk
WHITE, 9 proposals in the time gradient, caption = arm | replan | step).

The three arms SHARE one build_views() camera set: the top-down camera and
the re-posed views are fitted over the UNION of all arms' phase-1 EE
waypoints, so pixel scale is directly comparable across panels. Episodes
have different lengths: shorter rows hold their final SUCCESS frame.

Output: ``<out-dir>/arms_grid.mp4`` (3x3), per-arm frame trees under
``<out-dir>/<arm>/frames/<view>/`` and ``<out-dir>/summary.json``.

Usage (server, repo root; GPU must be idle; TH9-5-s233 round5 ckpts,
r6-eval parameter era):

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python -m scout.eval.visualize_arms_video \
        --config configs/eval_tool_hang_entropy.yaml --task tool_hang \
        --core-hdf5  <TOOLHANG-s233>/tool_hang/rollout/tool_hang_core.hdf5 \
        --dp-ckpt    <...>/train/DP/DP-DP-exp5/checkpoints/299.ckpt \
        --aty-ckpt   <...>/train/DP/DP-ATY-exp5/checkpoints/299.ckpt \
        --aty-vib    <...>/train/dyn/dyn-ATY-exp5/<ts>/scout_vib.ckpt \
        --orb-ckpt   <...>/train/DP/DP-ORBIT-exp5/checkpoints/299.ckpt \
        --orb-vib    <...>/train/dyn/dyn-ORBIT-exp5/<ts>/scout_vib.ckpt \
        --guidance-scale 0.5 --atypical-cap 2.5 \
        --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05 \
        --orbit-sigma-decay 0.5 --orbit-noise-anneal 2 --orbit-round 6 \
        --orbit-fb-clamp soft --stock-camera sideview --max-steps 700 \
        --seed 42 --out-dir <TOOLHANG-s233>/vis_r5_grid
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from scout.eval.visualize_chunk_video import (
    encode_mp4,
    load_run_config,
    make_chunk_sampler,
    make_policy,
    record_fit_points,
    render_replay,
    roll_until_success,
)


def compose_grid(arm_dirs, view_names, out_dir, fps):
    """Stream-compose the 3x3 grid mp4 from the per-arm frame trees.

    ``arm_dirs``: list of (arm_dir) in ROW order; each contains
    ``frames/<view>/NNNNN.png``. Frame index t is clamped per arm (shorter
    episodes hold their last frame). Grid = rows (arms) x columns (views).
    Returns ``(total_frames, grid_dir, per_arm_frame_counts)``.
    """
    def frame_path(arm_dir, view, idx):
        return os.path.join(arm_dir, "frames", view, f"{idx:05d}.png")

    counts = []
    for ad in arm_dirs:
        n = len([f for f in os.listdir(os.path.join(ad, "frames",
                                                    view_names[0]))
                 if f.endswith(".png")])
        counts.append(n)
    total = max(counts)
    grid_dir = os.path.join(out_dir, "_grid_frames")
    os.makedirs(grid_dir, exist_ok=True)
    for t in range(total):
        row_imgs = []
        for ad, n in zip(arm_dirs, counts):
            idx = min(t, n - 1)
            cells = [np.asarray(Image.open(frame_path(ad, v, idx)))
                     for v in view_names]
            row_imgs.append(np.concatenate(cells, axis=1))   # hstack views
        grid = np.concatenate(row_imgs, axis=0)              # vstack arms
        Image.fromarray(grid).save(os.path.join(grid_dir, f"{t:05d}.png"))
    return total, grid_dir, counts


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--core-hdf5", required=True)
    # row 1: DP baseline (guide off; vib unused but kept for config shape)
    p.add_argument("--dp-ckpt", required=True)
    p.add_argument("--dp-vib", default=None,
                   help="optional placeholder (DP row is unguided)")
    # row 2: SCOUT atypical
    p.add_argument("--aty-ckpt", required=True)
    p.add_argument("--aty-vib", required=True)
    # row 3: SCOUT-orbit
    p.add_argument("--orb-ckpt", required=True)
    p.add_argument("--orb-vib", required=True)
    # shared guidance dose (TH9-5 v3: raw s0.5 / cap 2.5 / gst from config=50)
    p.add_argument("--atypical-cap", type=float, default=2.5)
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="override cfg.exploration.guidance_scale (raw dose)")
    p.add_argument("--gst", type=int, default=None)
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
    p.add_argument("--max-attempts", type=int, default=40)
    p.add_argument("--res", type=int, default=480)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--step-frames", type=int, default=3)
    p.add_argument("--views", default="sideview,topdown,eye_in_hand")
    p.add_argument("--stock-camera", default="sideview")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keep-grid-frames", action="store_true",
                   help="keep the _grid_frames/ intermediate after encode")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    from scout.eval.factories import make_default_env_factory
    from scout.eval.visualize_trajectories import build_views

    # one config skeleton; per-arm ckpt overrides happen in make_policy's
    # factories (the vib entries are placeholders for the unguided row)
    cfg = load_run_config(args.config, args.dp_ckpt, args.aty_vib,
                          args.core_hdf5, args.guidance_scale, args.gst)
    env_factory = make_default_env_factory(cfg)

    np.random.seed(args.seed)

    # shared init state: every arm replays the SAME seed-42 scene
    env = env_factory()
    env.reset(seed=args.seed)
    state0 = env.get_state()
    if hasattr(env, "close"):
        env.close()

    aty_ek = {"atypical_cap": args.atypical_cap}
    orb_ek = dict(aty_ek)
    orb_ek.update({"orbit_lam": args.orbit_lam,
                   "orbit_delta": args.orbit_delta,
                   "orbit_sigma": args.orbit_sigma,
                   "orbit_sigma_decay": args.orbit_sigma_decay,
                   "orbit_noise_anneal": args.orbit_noise_anneal,
                   "orbit_round": args.orbit_round,
                   "orbit_fb_clamp": args.orbit_fb_clamp})
    arms = [
        ("dp", "DP (guide off)", args.dp_ckpt, args.aty_vib, "off", None),
        ("aty", "SCOUT (atypical)", args.aty_ckpt, args.aty_vib,
         "atypical", aty_ek),
        ("orb", "SCOUT-orbit (orbit)", args.orb_ckpt, args.orb_vib,
         "orbit", orb_ek),
    ]

    records, chunk_lens = {}, {}
    for key, label, ckpt, vib, guide, ek in arms:
        print(f"[arms_video] === arm {key} ({label}) phase 1 ===", flush=True)
        # per-arm vib base: the sd fully overwrites E_s anyway (strict load),
        # but keep the config self-consistent with the arm's own DP
        cfg.vib.base_dp_ckpt = ckpt
        dp, planner, chunk_len = make_policy(cfg, ckpt, vib, guide,
                                             ek or {"atypical_cap":
                                                    args.atypical_cap},
                                             device)
        sampler = make_chunk_sampler(dp, planner, device)
        records[key] = roll_until_success(
            env_factory, sampler, chunk_len, state0, args.max_steps,
            args.max_attempts, args.seed, n_samples=args.n_samples,
            tag=f"[arms_video:{key}]")
        chunk_lens[key] = chunk_len
        del dp, planner, sampler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # shared camera set: fitted over the UNION of all arms' waypoints
    est = (args.max_steps * args.step_frames + args.fps) * len(arms) \
        * len(args.views.split(","))
    print(f"[arms_video] frame estimate <= {est} PNGs "
          f"(res {args.res}^2, ~{est * (args.res * args.res * 3) / 2**30:.1f}"
          f" GiB raw worst case) -- check disk before phase 2", flush=True)
    fit = record_fit_points(list(records.values()))
    env = env_factory()
    env.reset_to(state0)
    renv = env.env
    views = build_views(renv.env.sim, fit_points=fit,
                        stock_camera=args.stock_camera,
                        center=fit.mean(axis=0))
    if hasattr(env, "close"):
        env.close()
    view_names = args.views.split(",")
    views = {v: views[v] for v in view_names}

    arm_dirs = []
    for (key, label, ckpt, vib, guide, ek) in arms:
        print(f"[arms_video] === arm {key} ({label}) phase 2 render ===",
              flush=True)
        adir = os.path.join(args.out_dir, key)
        render_replay(env_factory, state0, records[key], chunk_lens[key],
                      views, adir, args.res, args.fps, args.step_frames,
                      label=label, tag=f"[arms_video:{key}]")
        arm_dirs.append(adir)

    total, grid_dir, counts = compose_grid(arm_dirs, view_names, args.out_dir,
                                           args.fps)
    encode_mp4(grid_dir, os.path.join(args.out_dir, "arms_grid.mp4"),
               args.fps)
    if not args.keep_grid_frames:
        import shutil
        shutil.rmtree(grid_dir, ignore_errors=True)

    summary = {
        "seed": args.seed, "views": view_names,
        "arms": [{"key": k, "label": lbl, "guide": g,
                  "ckpt": ck, "steps": records[k]["steps"],
                  "replans": len(records[k]["replans"]),
                  "chunk_len": chunk_lens[k], "frames": counts[i]}
                 for i, (k, lbl, ck, _v, g, _ek) in enumerate(arms)],
        "grid_frames": total, "res": args.res, "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "orbit_round": args.orbit_round,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[arms_video] done: {total} grid frames -> "
          f"{os.path.join(args.out_dir, 'arms_grid.mp4')}", flush=True)


if __name__ == "__main__":
    main()
