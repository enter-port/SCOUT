"""Compatibility entry point; implementation: scout.calib.joint_pr."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scout.calib.joint_pr import calibrate, main  # noqa: E402,F401


if __name__ == "__main__":
    main()
