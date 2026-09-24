"""Core-only eta/kappa calibration. See README.md for CLI entry points.

Modules keep model dependencies lazy so importing this package or requesting
CLI help does not load Torch, CUDA, or robomimic.
"""
