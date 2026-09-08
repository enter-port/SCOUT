"""Trajectory visualization for SCOUT rollout / core robomimic hdf5 files.

Given an hdf5 in robomimic layout (``data/demo_N/obs/robot0_eef_pos`` ...),
this tool

1. re-creates the robomimic environment, resets it to one demo's initial
   state (``states[0]``) and renders two views of the INITIAL scene: the
   standard ``agentview`` camera, and a temporary TOP-DOWN camera (the
   ``agentview`` camera re-posed straight above the workspace, auto-fitted
   to cover every trajectory point);
2. overlays EVERY selected demo's end-effector trajectory on the top-down
   render, colored by a per-trajectory time gradient
   yellow -> orange -> red -> magenta (start -> end);
3. falls back to a plain 2D top-down matplotlib plot when the hdf5 has no
   ``states`` group (the sim then cannot be reset / re-rendered).

The world->pixel mapping is an exact pinhole projection: the top-down camera
pose (position / identity quat / vertical fov) is chosen by this script, so
any 3-D world point maps to pixels in closed form and the overlay is
geometrically aligned with the rendered scene, not eyeballed.

Usage (server, from the scout repo root; MUJOCO_GL=egl for offscreen):

    MUJOCO_GL=egl python -m scout.eval.visualize_trajectories \
        data/robomimic/can/ph/image_v141_abs_core20.hdf5

    # only a round's exploration rollouts (core demos occupy ids 0..19):
    MUJOCO_GL=egl python -m scout.eval.visualize_trajectories \
        data/can/experiment1/can_SCOUT_exp2/all.hdf5 --min-demo-id 20
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

EEF_KEY = "obs/robot0_eef_pos"
OBJ_KEY = "obs/object"
GRADIENT_ANCHORS = ["#FFDF3F", "#FF8A00", "#F51B07", "#C000A0"]  # yellow..magenta


def trajectory_cmap() -> LinearSegmentedColormap:
    """yellow -> orange -> red -> magenta (purple-red) time gradient."""
    return LinearSegmentedColormap.from_list("y_o_r_m", GRADIENT_ANCHORS)


# --------------------------------------------------------------------------- #
# hdf5 reading
# --------------------------------------------------------------------------- #
def _demo_id(name: str) -> Optional[int]:
    tail = name.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def select_demos(h5, mask: Optional[str], min_demo_id: Optional[int]) -> List[str]:
    """All ``data/demo*`` names, optionally filtered by a mask key and/or a
    minimum numeric demo id (SCOUT round files append exploration demos with
    ids >= n_core after the core demos)."""
    from scout.eval.hdf5_writer import _demo_list
    demos = _demo_list(h5, mask)
    if min_demo_id is not None:
        demos = [d for d in demos if _demo_id(d) is not None
                 and _demo_id(d) >= min_demo_id]
    if not demos:
        raise SystemExit("no demos selected (check --mask / --min-demo-id)")
    return sorted(demos, key=lambda d: (_demo_id(d) is None, _demo_id(d)))


def load_trajectories(h5, demos: List[str], with_object: bool) -> List[dict]:
    trajs = []
    for name in demos:
        grp = h5["data"][name]
        if EEF_KEY not in grp:
            raise SystemExit(f"{name}: missing '{EEF_KEY}' -- not a robomimic "
                             "hdf5 with EE observations")
        eef = np.asarray(grp[EEF_KEY][()], dtype=np.float64)
        obj = None
        if with_object and OBJ_KEY in grp["obs"]:
            obj = np.asarray(grp[OBJ_KEY][()], dtype=np.float64)[:, :3]
        success = None
        if "success" in grp:
            success = bool(np.any(np.asarray(grp["success"][()])))
        trajs.append({"name": name, "eef": eef, "obj": obj, "success": success})
    return trajs


# --------------------------------------------------------------------------- #
# environment replay + rendering (server-proven creation path)
# --------------------------------------------------------------------------- #
def make_env(dataset_path: str):
    """robomimic env for state replay + offscreen rendering.

    Mirrors ``scout.eval.rollout.make_robomimic_env_factory`` (env_meta from
    the dataset, ``create_env_from_metadata``, ``hard_reset=False`` so
    robosuite does not re-serialize the model XML on every reset).
    """
    import collections

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    # robomimic's env.reset() -> get_observation() needs the obs-key ->
    # modality mapping initialized (same call as in scout.eval.rollout).
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {"low_dim": ["robot0_eef_pos", "robot0_eef_quat",
                     "robot0_gripper_qpos"]})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["use_object_obs"] = False
    env_meta["env_kwargs"]["use_camera_obs"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True,
        use_image_obs=False,
    )
    env.env.hard_reset = False
    env.reset()
    return env


def topdown_camera_pose(points: np.ndarray, fov_deg: float,
                        margin: float = 0.12) -> np.ndarray:
    """Straight-down camera position covering every point (square render).

    Camera looks along -z with identity quat (world axes = camera axes), so
    pixel u grows with world +x and pixel v (downwards) grows with world -y.
    A point p is inside the frame iff h >= p.z + max(|dx|, |dy|)/tan(fov/2);
    take the max over all points plus a margin.
    """
    tan_half = np.tan(np.radians(fov_deg) / 2.0)
    cx = 0.5 * (points[:, 0].min() + points[:, 0].max())
    cy = 0.5 * (points[:, 1].min() + points[:, 1].max())
    r = np.maximum(np.abs(points[:, 0] - cx), np.abs(points[:, 1] - cy))
    h = float(np.max(points[:, 2] + r / tan_half)) + margin
    return np.array([cx, cy, h])


def project_to_pixels(points: np.ndarray, cam_pos: np.ndarray, fov_deg: float,
                      width: int, height: int) -> np.ndarray:
    """Exact pinhole projection for the straight-down camera (identity quat).

    Deprecated convenience wrapper: use :func:`project_to_pixels_frame` (works
    for ANY camera pose via the sim-reported frame). This closed form is only
    valid for the world-aligned identity-quat top-down camera; convention
    verified to sub-pixel accuracy against segmentation masks (mujoco 2.3.7 +
    robosuite EGL + robomimic's ``im[::-1]`` row flip).
    """
    f = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    d = points - cam_pos[None, :]
    depth = -d[:, 2]                       # camera looks along -z
    u = 0.5 * width + f * d[:, 0] / depth
    v = 0.5 * height - f * d[:, 1] / depth
    return np.stack([u, v], axis=1)


# --------------------------------------------------------------------------- #
# multi-view cameras + general exact projection (any camera pose)
# --------------------------------------------------------------------------- #
def project_to_pixels_frame(points, cam_xpos, cam_xmat, fov_deg: float,
                            width: int, height: int,
                            min_depth: float = 0.01) -> np.ndarray:
    """Exact pinhole projection for ANY mujoco camera, via its DATA-level
    frame (``sim.data.cam_xpos`` / ``cam_xmat`` AFTER ``sim.forward()``).

    ``cam_xmat`` rows are the camera axes in world coords; the camera looks
    along its -z axis (so visible points have negative camera-z). With
    identity frame this reduces exactly to the top-down closed form above
    (verified sub-pixel vs segmentation). Vertical ``camera_fov`` in degrees.

    Points closer than ``min_depth`` to the camera (e.g. the EE waypoint at
    the eye-in-hand camera's own origin -- depth 0) project to garbage; they
    are returned as NaN so downstream plotting silently skips them.
    """
    f = 0.5 * height / np.tan(np.radians(fov_deg) / 2.0)
    d = np.asarray(points, dtype=np.float64) \
        - np.asarray(cam_xpos, dtype=np.float64)[None, :]
    # Camera axes are the COLUMNS of cam_xmat (verified 2026-08-28 against
    # red-pixel-mask ground truth on the stock agentview: cols-rule err 18px
    # = can-radius level, rows-rule err >=375px; rows only coincide with
    # columns for symmetric frames, e.g. the identity-quat top-down camera).
    R = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    x_c, y_c, z_c = d @ R[:, 0], d @ R[:, 1], d @ R[:, 2]
    depth = -z_c
    bad = depth < min_depth
    depth_s = np.where(bad, 1.0, depth)      # avoid div-by-0, masked below
    u = 0.5 * width + f * x_c / depth_s
    v = 0.5 * height - f * y_c / depth_s
    out = np.stack([u, v], axis=1)
    out[bad] = np.nan
    return out


def lookat_camera_frame(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera (pos, quat) looking from ``pos`` at ``target``.

    Returns the mujoco quat (w,x,y,z) orienting the camera frame with x
    right, y up, looking along the pos->target direction (mujoco cameras
    look along their -z, so the frame's z axis points back at the viewer).

    NOTE the quat conversion MUST be mujoco's own ``mju_mat2Quat``:
    robosuite's ``transform_utils.mat2quat`` returns (x,y,z,w) order while
    mujoco ``cam_quat`` expects (w,x,y,z) -- writing robosuite's output
    directly shuffles the components into a garbage rotation (camera looks
    at the sky; verified 2026-08-28).
    """
    import mujoco

    pos = np.asarray(pos, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-8:        # looking straight along `up`
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up_v = np.cross(right, fwd)
    R = np.stack([right, up_v, -fwd], axis=0)   # rows = camera axes (world)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R).ravel())
    return pos, quat


WORKSPACE_CENTER = np.array([0.0, -0.15, 0.9])   # can task table workspace


def build_views(sim, fit_points: Optional[np.ndarray] = None) -> dict:
    """Every implemented camera view for the can workspace.

    Re-posed views share the model's ``agentview`` camera (pos + quat set per
    render, see :func:`render_view`); ``agentview`` restores the STOCK pose.
    ``eye_in_hand`` renders the gripper-mounted camera natively (no re-pose;
    its frame moves with the arm -- handled by projecting through the
    data-level frame at render time).

    ``fit_points`` (optional): trajectory points the top-down camera should
    cover; None -> a generous fixed workspace box (stable scale across video
    frames).
    """
    cid = sim.model.camera_name2id("agentview")
    fov = float(sim.model.cam_fovy[cid])
    # "agentview" = the STOCK robomimic camera pose (XML values, read from
    # the fresh model before any re-pose). The projection now handles its
    # frame exactly (column-axes rule; red-mask verified err 18px @512).
    views = {
        "agentview": {"camera": "agentview",
                      "pos": np.array(sim.model.cam_pos[cid], dtype=np.float64),
                      "quat": np.array(sim.model.cam_quat[cid],
                                       dtype=np.float64)},
    }
    if fit_points is None:
        fit_points = np.array(
            [[dx, dy, dz]
             for dx in (-0.35, 0.35) for dy in (-0.6, 0.3)
             for dz in (0.85, 1.3)], dtype=np.float64)
    views["topdown"] = {"camera": "agentview",
                        "pos": topdown_camera_pose(fit_points, fov),
                        "quat": np.array([1.0, 0.0, 0.0, 0.0])}
    for name, pos in (("front", np.array([0.0, 1.35, 1.45])),
                      ("side_left", np.array([-1.35, -0.15, 1.35])),
                      ("side_right", np.array([1.35, -0.15, 1.35])),
                      ("corner45", np.array([0.95, 1.0, 1.75]))):
        p, q = lookat_camera_frame(pos, WORKSPACE_CENTER)
        views[name] = {"camera": "agentview", "pos": p, "quat": q}
    views["eye_in_hand"] = {"camera": "robot0_eye_in_hand"}
    return views


def render_view(renv, view: dict, res: int):
    """Render one view on a robomimic ``EnvRobosuite`` (``renv``).

    Re-poses the shared camera (model + ``sim.forward()`` -- see
    ``render_initial_views`` for why forward is mandatory) unless the view is
    native (e.g. eye_in_hand). Returns ``(img, cam_xpos, cam_xmat, fovy)`` --
    feed the frame straight into :func:`project_to_pixels_frame`.
    """
    sim = renv.env.sim
    if "pos" in view:
        cid = sim.model.camera_name2id(view["camera"])
        sim.model.cam_pos[cid] = view["pos"]
        sim.model.cam_quat[cid] = view["quat"]
        sim.forward()
    img = renv.render(mode="rgb_array", height=res, width=res,
                      camera_name=view["camera"])
    cid = sim.model.camera_name2id(view["camera"])
    return (img, np.array(sim.data.cam_xpos[cid], dtype=np.float64),
            np.array(sim.data.cam_xmat[cid], dtype=np.float64),
            float(sim.model.cam_fovy[cid]))


def render_initial_views(env, state0: np.ndarray, fit_points: np.ndarray,
                         res: int):
    """Two renders of the initial state: agentview (original pose) + top-down.

    The top-down view re-poses the ``agentview`` camera in the mujoco model
    (position + identity quat) -- a model-level render setting, no env reset
    involved, so the restored simulator state is untouched.

    NOTE the mandatory ``sim.forward()`` after the re-pose: mujoco renders
    fixed cameras from the DATA-level frame (``data.cam_xpos`` /
    ``data.cam_xmat``), which ``mj_forward`` recomputes from
    ``model.cam_pos`` / ``model.cam_quat``; the render path
    (``mjv_updateScene``) never reads the model arrays directly.  Without
    the forward call the renderer keeps the PREVIOUS camera frame and the
    re-pose silently does nothing (the "top-down" render stays the stock
    angled agentview, while ``project_to_pixels`` keeps projecting with the
    new pose -> misaligned overlay).
    """
    env.reset_to({"states": np.asarray(state0, dtype=np.float64)})
    img_agent = env.render(mode="rgb_array", height=res, width=res,
                           camera_name="agentview")
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")
    fov_deg = float(sim.model.cam_fovy[cam_id])
    cam_pos = topdown_camera_pose(fit_points, fov_deg)
    sim.model.cam_pos[cam_id] = cam_pos
    sim.model.cam_quat[cam_id] = np.array([1.0, 0.0, 0.0, 0.0])
    sim.forward()
    # post-forward sanity: project_to_pixels' closed form is exact only for
    # a world-aligned camera frame (identity quat on a fixed worldbody cam).
    xmat = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    if not np.allclose(xmat, np.eye(3), atol=1e-6):
        raise RuntimeError(
            f"topdown camera frame is not world-aligned (cam_xmat="
            f"{xmat.tolist()}); project_to_pixels' identity-quat closed form "
            f"would misalign")
    img_top = env.render(mode="rgb_array", height=res, width=res,
                         camera_name="agentview")
    return img_agent, img_top, cam_pos, fov_deg


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def draw_gradient_path(ax, pts2d: np.ndarray, cmap, lw: float, alpha: float,
                       zorder: int = 3):
    """Polyline colored along itself from cmap(0)=yellow to cmap(1)=magenta."""
    if len(pts2d) < 2:
        return
    segs = np.stack([pts2d[:-1], pts2d[1:]], axis=1)
    colors = cmap(np.linspace(0.0, 1.0, len(segs)))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw,
                                     alpha=alpha, capstyle="round",
                                     zorder=zorder))


def mark_endpoints(ax, start_xy, end_xy):
    ax.scatter(*start_xy, s=16, c=GRADIENT_ANCHORS[0], edgecolors="black",
               linewidths=0.4, zorder=5)
    ax.scatter(*end_xy, s=30, c=GRADIENT_ANCHORS[-1], marker="X",
               edgecolors="white", linewidths=0.4, zorder=5)


def add_gradient_colorbar(fig, ax, cmap):
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["start (yellow)", "end (magenta)"])
    cbar.outline.set_linewidth(0.5)
    return cbar


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Render the initial scene of a robomimic hdf5 and overlay "
                    "every demo's EE trajectory with a yellow->orange->red->"
                    "magenta time gradient.")
    p.add_argument("hdf5", help="robomimic hdf5 (core or SCOUT rollout file)")
    p.add_argument("--mask", default=None,
                   help="demo filter key under mask/ (e.g. scout_aug, core_20)")
    p.add_argument("--min-demo-id", type=int, default=None,
                   help="only demos with numeric id >= N (round files: the "
                        "exploration rollouts appended after the core)")
    p.add_argument("--bg-demo", default=None,
                   help="demo name whose states[0] is rendered as the "
                        "background initial scene (default: first selected)")
    p.add_argument("--with-object", action="store_true",
                   help="also draw the object path (obs/object[:, :3], thin)")
    p.add_argument("--res", type=int, default=768,
                   help="render resolution in px (default 768)")
    p.add_argument("--lw", type=float, default=1.7,
                   help="trajectory line width (default 1.7)")
    p.add_argument("--out", default=None,
                   help="output png (default: <hdf5 stem>_trajviz.png)")
    p.add_argument("--save-raws", action="store_true",
                   help="also save the two raw renders as separate pngs")
    p.add_argument("--no-render", action="store_true",
                   help="skip the sim entirely; plain 2D top-down plot instead")
    args = p.parse_args()

    import h5py

    hdf5_path = Path(args.hdf5)
    with h5py.File(hdf5_path, "r") as h5:
        demos = select_demos(h5, args.mask, args.min_demo_id)
        trajs = load_trajectories(h5, demos, args.with_object)
        bg_demo = args.bg_demo or demos[0]
        bg_state = (h5["data"][bg_demo]["states"][0]
                    if "states" in h5["data"][bg_demo] else None)
        env_name = "env"
        try:
            import json
            env_name = json.loads(
                h5["data"].attrs["env_args"])["env_name"]
        except Exception:
            pass

    print(f"{hdf5_path.name}: {len(trajs)} demos "
          f"({demos[0]}..{demos[-1]}), env={env_name}")
    for t in trajs:
        succ = "-" if t["success"] is None else str(t["success"])
        e0, e1 = t["eef"][0], t["eef"][-1]
        print(f"  {t['name']:<12s} T={len(t['eef']):<4d} success={succ:<5s} "
              f"eef {np.round(e0, 3)} -> {np.round(e1, 3)}")

    cmap = trajectory_cmap()
    fit_points = np.concatenate(
        [t["eef"] for t in trajs]
        + [t["obj"] for t in trajs if t["obj"] is not None], axis=0)
    out_path = Path(args.out) if args.out else hdf5_path.parent / (
        hdf5_path.stem + "_trajviz.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_sim = (not args.no_render) and bg_state is not None
    if (not args.no_render) and bg_state is None:
        print("WARNING: no 'states' in the hdf5 -> cannot reset/render the "
              "sim; falling back to a schematic 2D top-down plot")

    if use_sim:
        env = make_env(str(hdf5_path))
        img_agent, img_top, cam_pos, fov_deg = render_initial_views(
            env, bg_state, fit_points, args.res)
        if hasattr(env, "close"):
            env.close()
        h_px, w_px = img_top.shape[:2]
        for t in trajs:
            t["px"] = project_to_pixels(t["eef"], cam_pos, fov_deg, w_px, h_px)
            if t["obj"] is not None:
                t["obj_px"] = project_to_pixels(t["obj"], cam_pos, fov_deg,
                                                w_px, h_px)

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})
        axes[0].imshow(img_agent)
        axes[0].set_title(f"initial scene -- {bg_demo} (agentview)")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        ax = axes[1]
        ax.imshow(img_top)
        ax.set_xlim(0, w_px); ax.set_ylim(h_px, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for t in trajs:
            if t.get("obj_px") is not None:
                draw_gradient_path(ax, t["obj_px"], cmap, lw=args.lw * 0.6,
                                   alpha=0.55, zorder=2)
            draw_gradient_path(ax, t["px"], cmap, lw=args.lw, alpha=0.9)
            mark_endpoints(ax, t["px"][0], t["px"][-1])
        ax.set_title(f"top-down + EE trajectories ({len(trajs)} demos)")
        add_gradient_colorbar(fig, ax, cmap)

        fig.suptitle(f"{hdf5_path.name} -- {env_name} -- initial state of "
                     f"{bg_demo}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=150)
        if args.save_raws:
            plt.imsave(out_path.parent / (out_path.stem + "_agentview.png"),
                       img_agent)
            plt.imsave(out_path.parent / (out_path.stem + "_topdown.png"),
                       img_top)
        # projection sanity anchor: where the manipulated object / EE of the
        # background demo should appear in the overlay
        for t in trajs:
            print(f"  [proj] {t['name']}: start px "
                  f"{np.round(t['px'][0], 0)}, end px {np.round(t['px'][-1], 0)}")
            break
    else:
        # schematic fallback: world x-y plane, equal aspect
        fig, ax = plt.subplots(figsize=(7.4, 7.4))
        for t in trajs:
            if t["obj"] is not None:
                draw_gradient_path(ax, t["obj"][:, :2], cmap,
                                   lw=args.lw * 0.6, alpha=0.55, zorder=2)
            draw_gradient_path(ax, t["eef"][:, :2], cmap, lw=args.lw, alpha=0.9)
            mark_endpoints(ax, t["eef"][0, :2], t["eef"][-1, :2])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"EE trajectories ({len(trajs)} demos, schematic "
                     "top-down)")
        add_gradient_colorbar(fig, ax, cmap)
        fig.suptitle(hdf5_path.name, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150)

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
