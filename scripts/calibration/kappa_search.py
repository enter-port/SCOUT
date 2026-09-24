"""Compatibility entry point; implementation: scout.calib.search."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scout.calib.search import solve_kappa  # noqa: E402,F401
