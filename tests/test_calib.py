"""Calibration protocol and CLI compatibility checks, without CUDA models."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from scout.calib import core, dp_kl, eta_r, kappa_c, joint_pr

ROOT = Path(__file__).resolve().parents[1]
CFG = SimpleNamespace(exploration={"guidance_start_timestep": 50})


class Tensor:
    """Tiny tensor double for testing observation hooks and reduction order."""
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def abs(self):
        return Tensor(np.abs(self.value))

    def mean(self):
        return self.value.mean()


class MeasurementTests(unittest.TestCase):
    def test_frozen_batch_and_whole_core_action_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "core.hdf5"
            with h5py.File(path, "w") as f:
                data = f.create_group("data", track_order=True)
                for demo, offset in [("demo_2", 20), ("demo_10", 100)]:
                    group = data.create_group(demo)
                    group["abs_actions"] = np.full((8, 2), offset, dtype=np.float32)
                    group["obs/image"] = np.arange(8, dtype=np.uint8)[:, None, None, None] * np.ones((8, 2, 3, 3), dtype=np.uint8)
                    group["obs/state"] = np.arange(8, dtype=np.float32)[:, None] + offset
            obs, scale = core.read_core_batch(path, ["image"], ["state"], 3)
            np.testing.assert_array_equal(obs["state"][:, :, 0], [[100, 101], [101, 102], [102, 103]])
            self.assertEqual(obs["image"].shape, (3, 2, 3, 2, 3))
            self.assertEqual(obs["image"].dtype, np.float32)
            self.assertEqual(float(obs["image"][0, 1, 0, 0, 0]), float(np.float32(1 / 255)))
            self.assertEqual(scale, 60)  # Includes unsampled demo_2.

    def test_measurement_uses_uncapped_cost_capped_gradient_and_restores_hooks(self):
        planner = SimpleNamespace()
        planner._kl_backward = lambda *args: (Tensor([1, 9]), None)

        def guided(*args):
            planner._kl_backward(*args[:3])
            return Tensor([2, 0]), None, None, None

        planner.guided_step = guided
        original_cost = planner._kl_backward
        dp = SimpleNamespace(predict_action_dyn_guided=lambda obs: planner.guided_step(Tensor([10, 20]), None, obs, .5))
        torch = SimpleNamespace(manual_seed=Mock())
        with patch.dict(sys.modules, {"torch": torch}):
            result = core.measure_guidance(dp, planner, {}, 3, 2, record_noisy=True)
            self.assertEqual(result["R_mean"], .75)
            self.assertEqual(result["C_mean"], 5)
            self.assertEqual(result["noisy_scale"].tolist(), [15])
            self.assertIs(planner.guided_step, guided)
            self.assertIs(planner._kl_backward, original_cost)
            torch.manual_seed.assert_called_once_with(0)
            dp.predict_action_dyn_guided = Mock(side_effect=RuntimeError("sampling failed"))
            with self.assertRaises(RuntimeError):
                core.measure_guidance(dp, planner, {}, 3, 2)
            self.assertIs(planner.guided_step, guided)
            self.assertIs(planner._kl_backward, original_cost)

    def test_legacy_c_keeps_torch_per_step_reduction(self):
        planner = SimpleNamespace(_kl_backward=lambda *args: (Tensor([1e8, 1, 1]), None), guided_step=None)

        def sample(obs):
            for _ in range(3):
                planner._kl_backward(None, None)

        with patch.dict(sys.modules, {"torch": SimpleNamespace(manual_seed=Mock())}):
            result = core.measure_guidance(SimpleNamespace(predict_action_dyn_guided=sample), planner,
                                           {}, 1, record_injection=False, step_mean_c=True)
        expected = float(np.mean([float(np.array([1e8, 1, 1], dtype=np.float32).mean())] * 3))
        self.assertEqual(result["C_mean"], expected)
        self.assertEqual(result["n_steps"], 3)


class MedianTests(unittest.TestCase):
    def test_direction_pairs_and_pooled_median(self):
        fm = np.array([[[0.], [10.]], [[2.], [14.]]])
        am = np.array([[[1.], [11.]], [[3.], [15.]]])
        fl, al = np.zeros_like(fm), np.full_like(am, np.log(4))
        _, independent, same = dp_kl.posterior_kl_arrays(fm, fl, am, al)
        expected = np.array([
            .5 * (np.log(4) + (1 + (fm[0] - am[1]) ** 2) / 4 - 1),
            .5 * (np.log(4) + (1 + (fm[1] - am[0]) ** 2) / 4 - 1),
        ]).squeeze(-1)
        np.testing.assert_allclose(independent, expected, rtol=1e-14)
        self.assertEqual(independent.shape, (2, 2))
        self.assertFalse(np.allclose(independent, same))
        self.assertNotEqual(float(np.median(independent)), float(np.median(independent.mean(0))))
        self.assertNotEqual(float(dp_kl.diagonal_kl([0], [0], [2], [np.log(4)])),
                            float(dp_kl.diagonal_kl([2], [np.log(4)], [0], [0])))

    def test_r_loop_accepts_initial_point(self):
        measure = Mock(return_value={"R_mean": .0105})
        eta, _, history = dp_kl.match_r(measure, 2, .01)
        self.assertEqual((eta, len(history)), (2, 1))

    def test_r_loop_returns_last_measured_eta_at_nine_probe_limit(self):
        eta, result, history = dp_kl.match_r(lambda eta: {"R_mean": .02}, 2, .01)
        self.assertEqual(len(history), 9)
        self.assertEqual(eta, 2 / 256)
        self.assertEqual(eta, history[-1]["eta"])
        self.assertEqual(result["R_mean"], .02)

    def test_invalid_or_mismatched_dose_is_rejected(self):
        for value in (0, float("nan")):
            with self.assertRaises(ValueError):
                dp_kl.match_r(lambda eta: {"R_mean": value}, 1, .01)
        metadata = dict(task="can", dp_ckpt="dp", vib_ckpt="dyn", core_hdf5="core")
        calibration = dict(metadata, kappa=3, eta=2, R_mean=.03, R_target=.01, R_converged=True)
        with self.assertRaises(ValueError):
            dp_kl.validate_calibration(calibration, metadata, 3)
        calibration.update(R_mean=.01, dp_ckpt="other")
        with self.assertRaises(ValueError):
            dp_kl.validate_calibration(calibration, metadata, 3)


class LegacyTests(unittest.TestCase):
    def run_eta(self, values):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "eta.json"
            argv = ["eta_r", "--eval-config", "cfg", "--dp-ckpt", "dp", "--vib-ckpt", "dyn",
                    "--core-hdf5", "core", "--eta-prev", "2", "--kappa-prev", "3", "--out", str(output)]
            with patch.dict(sys.modules, {"scout.guidance.entropy_costs": SimpleNamespace(KLCostPlanner=Mock())}), \
                    patch.object(sys, "argv", argv), \
                    patch.object(eta_r, "prepare_core", return_value=(CFG, {"state": np.zeros((128, 2))}, 1)), \
                    patch.object(eta_r, "load_model_pair", return_value=(Mock(), None, None, None)), \
                    patch.object(eta_r, "measure_guidance", side_effect=[{"R_mean": v} for v in values]), \
                    redirect_stdout(io.StringIO()):
                eta_r.main()
            return json.loads(output.read_text())

    def test_eta_always_updates_once_even_when_initial_r_in_band(self):
        result = self.run_eta([.0105, .01])
        self.assertEqual(result["n_updates"], 1)
        self.assertEqual(result["eta"], 2 * .01 / .0105)
        self.assertEqual(result["kappa"], 3)

    def test_eta_legacy_nonconvergence_saves_last_value(self):
        result = self.run_eta([.02] * 4)
        self.assertFalse(result["converged"])
        self.assertEqual((result["n_updates"], result["eta"]), (3, .25))

    def test_c_base_reference_does_not_follow_round_initial_kappa(self):
        calls = []

        def measure(dp, planner, obs, eta, **kwargs):
            calls.append((planner.cap, eta, kwargs))
            return dict(C_mean=4 if len(calls) == 1 else planner.cap, n_steps=50)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "c.json"
            argv = ["kappa_c", "--eval-config", "cfg", "--core-hdf5", "core", "--base-dp-ckpt", "base",
                    "--base-vib-ckpt", "base_dyn", "--base-eta", "7", "--round-dp-ckpt", "round",
                    "--round-vib-ckpt", "round_dyn", "--round-eta", "2", "--kappa0", "8", "--out", str(output)]
            with patch.dict(sys.modules, {
                    "scout.eval.factories": SimpleNamespace(load_cfg=lambda _: CFG),
                    "scout.guidance.entropy_costs": SimpleNamespace(KLCostPlanner=lambda *a, **kw: SimpleNamespace(**kw))}), \
                    patch.object(sys, "argv", argv), \
                    patch.object(kappa_c, "prepare_core", return_value=(CFG, {}, None)), \
                    patch.object(kappa_c, "load_model_pair", return_value=(Mock(), None, None, None)), \
                    patch.object(kappa_c, "measure_guidance", side_effect=measure), redirect_stdout(io.StringIO()):
                kappa_c.main()
            result = json.loads(output.read_text())
        self.assertEqual([(cap, eta) for cap, eta, _ in calls], [(2.5, 7), (8, 2), (4, 2)])
        self.assertTrue(all(kw == dict(record_injection=False, step_mean_c=True) for _, _, kw in calls))
        self.assertTrue(result["converged"])


class EntryPointTests(unittest.TestCase):
    def test_p6_direct_entry_saves_measured_pair_and_fails_outside_band(self):
        for r, converged in ((.01, True), (.03, False)):
            with self.subTest(r=r), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                for name in ("dp", "dyn", "core"):
                    (path / name).touch()
                output = path / "p6.json"
                argv = ["joint_pr", "--task", "can", "--dp-ckpt", str(path / "dp"),
                        "--vib-ckpt", str(path / "dyn"), "--core-hdf5", str(path / "core"),
                        "--potential-cap", "6", "--max-probes", "1", "--out", str(output)]
                with patch.dict(sys.modules, {"scout.guidance.entropy_costs": SimpleNamespace(KLCostPlanner=Mock())}), \
                        patch.object(sys, "argv", argv), patch.object(joint_pr, "check_gpu"), \
                        patch.object(joint_pr, "prepare_core", return_value=(CFG, {"state": np.zeros((128, 2))}, 1)), \
                        patch.object(joint_pr, "load_model_pair", return_value=(Mock(), None, None, None)), \
                        patch.object(joint_pr, "measure_guidance", return_value=dict(R_mean=r, C_mean=3, kl=np.array([1, 5]))), \
                        redirect_stdout(io.StringIO()):
                    if converged:
                        joint_pr.main()
                    else:
                        with self.assertRaises(SystemExit):
                            joint_pr.main()
                result = json.loads(output.read_text())
                self.assertEqual(result["R_converged"], converged)
                self.assertEqual(result["eta"] * result["kappa"], 6)
                self.assertEqual(result["history"][0]["R_mean"], r)

    def test_median_direct_entry_writes_artifacts_and_rejects_failed_dose(self):
        for r, converged in ((.01, True), (.03, False)):
            with self.subTest(r=r), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                for name in ("dp", "dyn", "core"):
                    (path / name).touch()
                output = path / "median.json"

                def diversity(directory, metadata, *args):
                    dest = directory / "dp_diversity"
                    dest.mkdir()
                    summary = dict(metadata, samples=8, B=128,
                                   independent_final_to_anchor=dict(mean=100, quantiles=dict(p50=3)))
                    (dest / "summary.json").write_text(json.dumps(summary))

                def dose(metadata, dp, vib, bridge, adapter, obs, gst, scale, cap, eta, target):
                    self.assertEqual(cap, 3)  # Uses the median, never the mean=100.
                    return dict(metadata, kappa=cap, eta=eta, R_mean=r, R_target=target,
                                R_converged=converged), dict(kl=np.ones((1, 128)))

                argv = ["dp_kl", "--task", "can", "--dp-ckpt", str(path / "dp"),
                        "--vib-ckpt", str(path / "dyn"), "--core-hdf5", str(path / "core"),
                        "--eta-initial", "2", "--out", str(output)]
                with patch.object(sys, "argv", argv), patch.object(dp_kl, "check_gpu"), \
                        patch.object(dp_kl, "prepare_core", return_value=(CFG, {"state": np.zeros((128, 2))}, 1)), \
                        patch.object(dp_kl, "load_model_pair", return_value=(None, None, None, None)), \
                        patch.object(dp_kl, "posterior_diversity", side_effect=diversity), \
                        patch.object(dp_kl, "calibrate_dose", side_effect=dose), redirect_stdout(io.StringIO()):
                    if converged:
                        dp_kl.main()
                    else:
                        with self.assertRaises(ValueError):
                            dp_kl.main()
                result = json.loads(output.read_text())
                self.assertEqual((result["kappa"], result["R_converged"]), (3, converged))
                self.assertTrue(Path(result["median_reference"]).is_file())
                self.assertTrue(Path(result["calibration_path"]).with_suffix(".npz").is_file())

    def test_calibration_cli_help_without_model_dependencies(self):
        entries = [["-m", f"scout.calib.{name}"] for name in ("eta_r", "kappa_c", "joint_pr", "dp_kl")]
        for entry in entries:
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, *entry, "--help"], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_calibration_modules_are_current_implementations(self):
        self.assertTrue(callable(joint_pr.calibrate))
        self.assertTrue(callable(dp_kl.calibrate))
        self.assertTrue(callable(eta_r.main))
        self.assertTrue(callable(kappa_c.main))


if __name__ == "__main__":
    unittest.main()
