"""Orchestration invariants; no CUDA/model dependency."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.atom import campaign
from scripts.atom import calib, dp_train, dyn_train, eval_explore, grid_search
from scripts.atom.common import Context, ROOT, dataset, write_json


class AtomTests(unittest.TestCase):
    def config(self, root):
        core = root / "core.hdf5"
        core.touch()
        eval_config = root / "eval.yaml"
        eval_config.touch()
        return {"task": "can", "gpu": 0, "core_hdf5": str(core), "output_dir": str(root / "out"),
                "eval_config": str(eval_config), "grid": {"enabled": False},
                "calib": {"mode": "none"}, "arms": ["ATY", "DP"]}

    def test_repository_resolution(self):
        self.assertTrue((ROOT / "train.py").is_file())

    def test_every_task_uses_the_four_file_standard_layout(self):
        import yaml
        expected = {"can", "coffee", "coffee_prep", "lift", "square", "threading",
                    "tool_hang", "transport"}
        task_dirs = {p.name for p in (ROOT / "configs").iterdir() if p.is_dir()}
        self.assertEqual(task_dirs, expected)
        for task in sorted(expected):
            directory = ROOT / "configs" / task
            self.assertEqual({p.name for p in directory.iterdir() if p.is_file()},
                             {"campaign.json", "base_dp.yaml", "dyn.yaml", "eval.yaml"})
            config = json.loads((directory / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(config["task"], task)
            self.assertEqual(config["dp_config"], f"{task}/base_dp")
            self.assertEqual(config["dyn_config"], f"configs/{task}/dyn.yaml")
            self.assertEqual(config["eval_config"], f"configs/{task}/eval.yaml")
            self.assertEqual(config["round_plan"][-1],
                             {"rounds": [6], "calib": {"mode": "none"}, "train": False})
            eval_config = yaml.safe_load((directory / "eval.yaml").read_text(encoding="utf-8"))
            base_dp = eval_config["base_dp"]
            policy_config = ROOT / base_dp["config_dir"] / (base_dp["config_name"] + ".yaml")
            self.assertEqual(policy_config, directory / "base_dp.yaml")
            self.assertTrue(policy_config.is_file())

    def test_complete_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            c = self.config(root)
            c.update(grid={"enabled": True, "etas": [1., 2.], "kappas": [2.5]}, calib={"mode": "rc"})
            before = set(root.rglob("*"))
            result = campaign.run(Context(c, True))
            self.assertEqual(len(result), 12)
            self.assertEqual(set(root.rglob("*")), before)

    def test_receipts_skip_success_reject_config_change_and_missing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = Context(self.config(root))
            calls = []
            def action(work):
                calls.append(work)
                path = work / "x"
                path.touch()
                return {"artifacts": [str(path)]}
            stage = root / "stage"
            result = ctx.stage(stage, {"round": 1}, action)
            self.assertEqual(ctx.stage(stage, {"round": 1}, action), result)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                ctx.stage(stage, {"round": 2}, action)
            Path(result["artifacts"][0]).unlink()
            with self.assertRaises(FileNotFoundError):
                ctx.stage(stage, {"round": 1}, action)

    def test_failed_stage_gets_fresh_attempt_no_false_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = Context(self.config(root))
            def fail(work):
                (work / "partial.json").touch()
                raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError):
                ctx.stage(root / "stage", {}, fail)
            self.assertFalse((root / "stage/done.json").exists())
            def retry(work):
                self.assertFalse((work / "partial.json").exists())
                return {}
            ctx.stage(root / "stage", {}, retry)
            self.assertEqual(len(list((root / "stage").glob("attempt-*"))), 2)

    def test_six_round_dataflow_and_last_round_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = Context(self.config(Path(d)), True)
            def rollout(ctx, out, dp, dyn, arm, n, dose, options=None):
                return {"success": f"{arm}/s{n}", "all": f"{arm}/a{n}"}
            dp_inputs, dyn_inputs = [], []
            def dp(ctx, out, paths, name, options=None, resume_run_id=None):
                dp_inputs.append(list(paths))
                return {"dp": name}
            def dyn(ctx, out, dp, paths, name, options=None, resume_run_id=None):
                dyn_inputs.append((dp, list(paths)))
                return {"dyn": name}
            with patch.object(campaign.base_train, "prepare", return_value={"dp": "base", "dyn": "vib"}), \
                 patch.object(campaign.eval_explore, "run", side_effect=rollout), \
                 patch.object(campaign.dp_train, "train", side_effect=dp), \
                 patch.object(campaign.dyn_train, "train", side_effect=dyn):
                results = campaign.run(ctx)
            self.assertEqual(len(results), 12)
            self.assertEqual(len(dp_inputs), 10)
            self.assertEqual(len(dyn_inputs), 5)
            self.assertEqual(dp_inputs[4], [f"ATY/s{i}" for i in range(1, 6)])
            self.assertEqual(dp_inputs[5], ["DP/s1"])
            self.assertEqual(dyn_inputs[-1], ("ATY-round5-DP", [f"ATY/a{i}" for i in range(1, 6)]))

    def test_round_plan_uses_p6_then_kl_median_and_final_eval_only(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.config(Path(d))
            c.update(arms=["ATY"], rounds=6, round_plan=[
                {"rounds": [1], "calib": {"mode": "pr"}, "train": True},
                {"rounds": [2, 3, 4, 5], "calib": {"mode": "dp_kl"}, "train": True},
                {"rounds": [6], "calib": {"mode": "none"}, "train": False},
            ])
            ctx = Context(c, True)
            modes, trained, rounds = [], [], []

            def fake_calib(ctx, out, dp, dyn, previous, base=None, mode=None, options=None):
                modes.append(mode)
                return {"eta": previous["eta"] + 1, "kappa": previous["kappa"] + 1}

            def fake_rollout(ctx, out, dp, dyn, arm, number, dose, options=None):
                rounds.append(number)
                return {"success": f"s{number}", "all": f"a{number}"}

            def fake_dp(ctx, out, paths, name, options=None, resume_run_id=None):
                trained.append(("dp", name))
                return {"dp": name}

            def fake_dyn(ctx, out, dp, paths, name, options=None, resume_run_id=None):
                trained.append(("dyn", name))
                return {"dyn": name}

            with patch.object(campaign.base_train, "prepare", return_value={"dp": "base", "dyn": "dyn"}), \
                 patch.object(campaign.grid_search, "search", return_value={"ATY": {"eta": 1, "kappa": 2}}), \
                 patch.object(campaign.calib, "calibrate", side_effect=fake_calib), \
                 patch.object(campaign.eval_explore, "run", side_effect=fake_rollout), \
                 patch.object(campaign.dp_train, "train", side_effect=fake_dp), \
                 patch.object(campaign.dyn_train, "train", side_effect=fake_dyn):
                result = campaign.run(ctx)

            self.assertEqual(len(result), 6)
            self.assertEqual(modes, ["pr", "dp_kl", "dp_kl", "dp_kl", "dp_kl"])
            self.assertEqual(rounds, [1, 2, 3, 4, 5, 6])
            self.assertEqual(len(trained), 10)

    def test_empty_successes_still_retrain_core(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            ctx = Context(self.config(Path(d)), True)
            with patch.object(ctx, "run") as command:
                dp_train.train(ctx, ctx.root / "dp", [])
            args = list(map(str, command.call_args.args[0]))
            self.assertEqual(args[args.index("--config-path") + 1], "configs/can")
            self.assertEqual(args[args.index("--config-name") + 1], "base_dp")
            self.assertIn("task.train_filter_key=scout_aug", args)
            self.assertIn("task.dataset.seed=233", args)
            self.assertIn("training.resume=False", args)

    def test_dyn_real_run_launches_training_and_records_checkpoint(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = Context(self.config(root))
            dp = str(root / "dp.ckpt")
            Path(dp).touch()

            def train_process(cmd, log, extra_env=None):
                self.assertEqual(cmd[:3], [ctx.py, "-m", "scout.train_vib"])
                config_path = Path(cmd[cmd.index("--config") + 1])
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["dataset"]["zarr_path"], str(ctx.core))
                self.assertEqual(config["model"]["E_s"]["base_dp_ckpt"], dp)
                self.assertEqual(config["num_epochs"], 2)
                checkpoint = Path(config["save_dir"]) / "run" / "scout_vib.ckpt"
                checkpoint.parent.mkdir()
                checkpoint.touch()

            with patch.object(ctx, "run", side_effect=train_process) as command:
                result = dyn_train.train(ctx, root / "dyn", dp, base=True, options={"epochs": 2})
            command.assert_called_once()
            self.assertTrue(Path(result["dyn"]).is_file())
            self.assertTrue((root / "dyn" / "done.json").is_file())

    def test_rollout_guided_eval_frozen_rescue_and_sharding(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = Context(self.config(Path(d)), True)
            with patch.object(ctx, "run") as command:
                eval_explore.run(ctx, ctx.root, "dp", "dyn", "ATY", 6, {"eta": 9, "kappa": 2})
            first = list(map(str, command.call_args_list[0].args[0]))
            second = list(map(str, command.call_args_list[1].args[0]))
            self.assertEqual(first[first.index("--guide") + 1], "atypical")
            self.assertIn("--vib-ckpt", first)
            self.assertIn("--save-failed-set", first)
            self.assertIn("--failed-set-json", second)
            self.assertIn("--stop-on-first-success", second)
            self.assertEqual(second[second.index("--guidance-scale") + 1], "9")

    def test_calibration_modes_dispatch_without_c_after_pr(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.config(Path(d))
            for mode, modules in [("r", ["scout.calib.eta_r"]), ("rc", ["scout.calib.eta_r", "scout.calib.kappa_c"]),
                                  ("pr", ["scout.calib.joint_pr"]), ("dp_kl", ["scout.calib.dp_kl"])]:
                ctx = Context(c, True)
                with patch.object(ctx, "module") as call:
                    calib.calibrate(ctx, ctx.root, "dp", "dyn", {"eta": 3, "kappa": 2.5},
                                    {"dp": "bdp", "dyn": "bdyn", "eta": 2, "kappa": 1}, mode)
                self.assertEqual([x.args[0] for x in call.call_args_list], modules)

    def test_grid_reuses_same_failure_set_and_selects_measured_peak(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.config(Path(d))
            c["grid"] = {"enabled": True, "etas": [1, 2, 3], "kappas": [2.5]}
            ctx = Context(c)
            frozen_inputs = []
            def evaluate(ctx, out, dp, dyn, arm, n, dose, frozen):
                frozen_inputs.append(frozen)
                return {"failed": "same-failures.json", "eval": "same-eval.json", "metrics": str(out),
                        "pass_at_k": {1: .4, 2: .7, 3: .7}[dose["eta"]]}
            with patch.object(grid_search.eval_explore, "run", side_effect=evaluate):
                winners = grid_search.search(ctx, {"dp": "base", "dyn": "dyn"}, ["ATY"])
            self.assertIsNone(frozen_inputs[0])
            self.assertEqual(frozen_inputs[1]["failed"], "same-failures.json")
            self.assertEqual(frozen_inputs[1], frozen_inputs[2])
            self.assertEqual(winners["ATY"], {"eta": 2, "kappa": 2.5})

    def test_accumulation_keeps_core_once_and_excludes_future_rounds(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = Context(self.config(root))
            for name, values in [("core.hdf5", [10]), ("round1.hdf5", [10, 20]),
                                 ("round2.hdf5", [10, 30]), ("future.hdf5", [10, 999])]:
                with h5py.File(root / name, "w") as f:
                    group = f.create_group("data")
                    for i, value in enumerate(values):
                        group.create_group(f"demo_{i}").create_dataset("actions", data=[[value]])
            output = root / "accum.hdf5"
            dataset(ctx, output, [root / "round1.hdf5", root / "round2.hdf5"])
            with h5py.File(output) as f:
                self.assertEqual([f[f"data/demo_{i}/actions"][0, 0] for i in range(3)], [10, 20, 30])
                self.assertEqual(int(f["mask/scout_aug/mask"][:].sum()), 3)

    def test_pr_rejects_wrong_measurement_even_if_solver_claims_convergence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = Context(self.config(root))
            def solver(module, args, log):
                write_json(args[args.index("--out") + 1],
                           {"eta": 2, "kappa": 3, "R_mean": .02, "R_converged": True})
            with patch.object(ctx, "module", side_effect=solver), \
                 patch("scripts.atom.calib.subprocess.check_output", return_value="GPU-test"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    calib.calibrate(ctx, root / "calib", "dp", "dyn", {"eta": 2, "kappa": 3}, mode="pr")
            self.assertFalse((root / "calib/done.json").exists())


if __name__ == "__main__":
    unittest.main()
