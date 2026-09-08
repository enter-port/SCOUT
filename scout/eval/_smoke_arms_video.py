"""Hermetic smoke for the 3-arm grid video machinery (no env / torch /
mujoco): compose_grid padding+layout and record_fit_points union.

Run:  python -m scout.eval._smoke_arms_video
"""
import os
import shutil
import sys
import tempfile
import types

# torch is only *imported* at module level (used inside mains) -- stub it so
# the smoke runs on machines without torch.
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except Exception:
        sys.modules["torch"] = types.ModuleType("torch")

import numpy as np
from PIL import Image

from scout.eval.visualize_arms_video import compose_grid
from scout.eval.visualize_chunk_video import record_fit_points


def _mk_arm(root, name, views, n_frames, size=32):
    d = os.path.join(root, name)
    for v in views:
        os.makedirs(os.path.join(d, "frames", v), exist_ok=True)
        for t in range(n_frames):
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, :, 0] = 10 * (hash(name) % 20)      # per-arm channel
            arr[:, :, 1] = 10 * (views.index(v) + 1)   # per-view channel
            arr[:, :, 2] = 5 * t                        # per-frame marker
            Image.fromarray(arr).save(
                os.path.join(d, "frames", v, f"{t:05d}.png"))
    return d


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="smoke_arms_")
    try:
        views = ["v1", "v2"]
        a1 = _mk_arm(tmp, "a1", views, 5)   # longer episode
        a2 = _mk_arm(tmp, "a2", views, 3)   # shorter episode (padding)

        total, grid_dir, counts = compose_grid([a1, a2], views, tmp, 24)
        assert counts == [5, 3], counts
        assert total == 5, total
        g4 = np.asarray(Image.open(os.path.join(grid_dir, "00004.png")))
        assert g4.shape == (64, 64, 3), g4.shape           # 2 rows x 2 cols

        def cell(img, r, c, size=32):
            return img[r * size:(r + 1) * size, c * size:(c + 1) * size, :]

        # t=0: all cells carry their own arm/view/frame markers
        g0 = np.asarray(Image.open(os.path.join(grid_dir, "00000.png")))
        ref = np.asarray(Image.open(os.path.join(a1, "frames", "v1",
                                                 "00000.png")))
        assert (cell(g0, 0, 0) == ref).all()
        # t=4 (= max): row 1 (a2, 3 frames) must HOLD its last frame (idx 2)
        ref2 = np.asarray(Image.open(os.path.join(a2, "frames", "v2",
                                                  "00002.png")))
        assert (cell(g4, 1, 1) == ref2).all()
        # and row 0 (a1) shows its real frame 4
        ref1 = np.asarray(Image.open(os.path.join(a1, "frames", "v1",
                                                  "00004.png")))
        assert (cell(g4, 0, 0) == ref1).all()

        # record_fit_points: union of anchors (1) + proposal waypoints (2*4)
        recs = [{"replans": [
            {"start_ee": np.zeros(3),
             "chunks": np.arange(2 * 4 * 10).reshape(2, 4, 10)}]}]
        pts = record_fit_points(recs)
        assert pts.shape == (1 + 8, 3), pts.shape
        print("smoke_arms_video: GREEN "
              f"(grid layout/padding + fit-union, {total} frames)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
