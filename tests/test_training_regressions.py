"""Failure-path regressions without GPUs, real training or W&B uploads."""
import ast
from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.atom import calib, campaign, eval_explore, log_explore, shard_rollout
from scripts.atom.common import Context, ROOT, read_json, write_json


class TrainingRegressions(unittest.TestCase):
    def context(self, root, **options):
        for name in ("core.hdf5", "eval.yaml"):
            (root / name).touch()
        config = dict(task="tool_hang", gpu=0, output_dir=str(root / "out"),
                      core_hdf5=str(root / "core.hdf5"), eval_config=str(root / "eval.yaml"),
                      grid={"enabled": False}, arms=["ATY", "DP"], rounds=2,
                      calib={"mode": "none"})
        config.update(options)
        return Context(config)

    def test_eval_only_final_logging_with_minimal_and_full_metrics(self):
        # Execute the actual final logging block, isolating MuJoCo/model imports.
        tree = ast.parse((ROOT / "scout/eval/run_rollout.py").read_text(encoding="utf-8"))
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "wandb_run is not None"
                     and "final/eval_success_rate" in ast.unparse(n))
        code = compile(ast.Module(body=[block], type_ignores=[]), "run_rollout", "exec")
        for minimal in (False, True):
            for eval_only in (False, True):
                with self.subTest(minimal=minimal, eval_only=eval_only):
                    run = Mock()
                    metrics = dict(success_rate=.2, n_failed=80)
                    if not eval_only:
                        metrics.update(pass_at_5=.4, avg_jerk=.1)
                    exec(code, dict(wandb_run=run, split_mode=False,
                                    args=SimpleNamespace(wandb_minimal=minimal, eval_only=eval_only),
                                    cfg=SimpleNamespace(eval=SimpleNamespace(n_init_states=100)),
                                    metrics=metrics))
                    self.assertEqual(run.log.call_count, 1 if eval_only else 2)
                    self.assertEqual(run.log.call_args_list[0].args[0]["eval/success_rate"], .2)

    def test_r_and_c_reject_failed_or_false_positive_convergence(self):
        for mode, failing_phase in (("r", "R"), ("rc", "R"), ("rc", "C")):
            for flag, value in ((False, 1.), (True, 2.), (True, float("nan"))):
                with self.subTest(mode=mode, phase=failing_phase, flag=flag, value=value), \
                        tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    ctx = self.context(root)
                    called = []

                    def solver(module, args, log):
                        called.append(module)
                        is_r = module.endswith("eta_r")
                        result = dict(eta=1., kappa=2.5, converged=True)
                        if is_r:
                            result.update(R_mean=.01)
                        else:
                            result.update(C_mean=1., C_target=1.)
                        if (is_r and failing_phase == "R") or (not is_r and failing_phase == "C"):
                            result.update(converged=flag)
                            result["R_mean" if is_r else "C_mean"] = value * (.01 if is_r else 1.)
                        write_json(args[args.index("--out") + 1], result)

                    with patch.object(ctx, "module", side_effect=solver):
                        with self.assertRaisesRegex(ValueError, "converge|target band"):
                            calib.calibrate(ctx, root / "calib", "dp", "dyn", {"eta": 1, "kappa": 2.5},
                                            base={"dp": "b", "dyn": "v", "eta": 1, "kappa": 2.5}, mode=mode)
                    self.assertFalse((root / "calib/done.json").exists())
                    self.assertTrue(list((root / "calib").glob("attempt-*/eta.json")))
                    self.assertEqual(len(called), 1 if failing_phase == "R" else 2)

    def test_successful_rc_keeps_both_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ctx = self.context(root)

            def solver(module, args, log):
                write_json(args[args.index("--out") + 1],
                           dict(eta=1., kappa=2.5, converged=True, R_mean=.01, C_mean=1., C_target=1.))

            with patch.object(ctx, "module", side_effect=solver):
                result = calib.calibrate(ctx, root / "calib", "dp", "dyn", {"eta": 1, "kappa": 2.5},
                                         base={"dp": "b", "dyn": "v", "eta": 1, "kappa": 2.5}, mode="rc")
            self.assertEqual({Path(p).name for p in result["artifacts"]}, {"eta.json", "kappa.json"})
            self.assertTrue((root / "calib/done.json").exists())

    def test_rc_forwards_explicit_bracket_settings(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(Path(d), calib={"c_solver": "bracket", "c_max_probes": 12})
            ctx.dry = True
            with patch.object(ctx, "module") as call:
                calib.calibrate(ctx, ctx.root, "dp", "dyn", {"eta": 1, "kappa": 2.5},
                                base={"dp": "b", "dyn": "v", "eta": 1, "kappa": 2.5}, mode="rc")
            args = call.call_args.args[1]
            self.assertEqual(args[args.index("--solver") + 1], "bracket")
            self.assertEqual(args[args.index("--max-probes") + 1], 12)
            self.assertIn("--require-converged", args)

    def test_dp_never_calibrates_and_stages_share_round_identity(self):
        for arms in (["DP"], ["DP", "ATY"]):
            with self.subTest(arms=arms), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                ctx = self.context(root, arms=arms, calib={"mode": "rc"})
                training = []

                def rollout(ctx, out, dp, dyn, arm, number, dose, options=None):
                    return {"wandb_run_id": f"{arm}-{number}"}

                def dp_train(ctx, out, data, name, options=None, resume_run_id=None):
                    training.append(("dp", name, resume_run_id))
                    return {"dp": "trained_dp"}

                def dyn_train(ctx, out, dp, data, name, options=None, resume_run_id=None):
                    training.append(("dyn", name, resume_run_id))
                    return {"dyn": "trained_dyn"}

                with patch.object(campaign.base_train, "prepare", return_value={"dp": "dp", "dyn": "dyn"}), \
                        patch.object(campaign.calib, "calibrate", return_value={"eta": 1, "kappa": 2.5}) as calibration, \
                        patch.object(campaign.eval_explore, "run", side_effect=rollout), \
                        patch.object(campaign.dp_train, "train", side_effect=dp_train), \
                        patch.object(campaign.dyn_train, "train", side_effect=dyn_train):
                    results = campaign.run(ctx)
                self.assertTrue(all(r["calib_mode"] == "none" for r in results if r["arm"] == "DP"))
                self.assertEqual(calibration.call_count, 0 if arms == ["DP"] else 3)
                self.assertIn(("dp", "DP-round1", "DP-1"), training)
                if "ATY" in arms:
                    self.assertIn(("dp", "SCOUT-aty-round1", "ATY-1"), training)
                    self.assertIn(("dyn", "SCOUT-aty-round1", "ATY-1"), training)

    def test_explore_upload_retry_reuses_rollout_and_original_id(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            ctx = self.context(root, wandb_mode="online", wandb_project="test-project")
            calls = []
            fail_upload = True

            def module(name, args, log, **kwargs):
                nonlocal fail_upload
                calls.append(name)
                if name == "scout.eval.run_rollout":
                    self.assertEqual(args[args.index("--wandb-name") + 1], "SCOUT-aty-round1")
                    write_json(args[args.index("--save-failed-set") + 1], {})
                    write_json(args[args.index("--output-json") + 1], {"wandb_run_id": "original-id"})
                elif name == "scripts.atom.shard_rollout":
                    self.assertIn("--no-wandb", args)  # Workers never compete for one run.
                    write_json(args[1], dict(n_init_states=100, explore_try_times=5,
                                            success_rate=.2, pass_at_5=.4))
                else:
                    self.assertEqual(name, "scripts.atom.log_explore")
                    self.assertEqual(args[args.index("--run-id") + 1], "original-id")
                    self.assertEqual(args[args.index("--name") + 1], "SCOUT-aty-round1")
                    if fail_upload:
                        fail_upload = False
                        raise RuntimeError("simulated upload failure")

            with patch.object(ctx, "module", side_effect=module):
                with self.assertRaisesRegex(RuntimeError, "upload failure"):
                    eval_explore.run(ctx, root / "rollout", "dp", "dyn")
                self.assertTrue((root / "rollout/explore/done.json").exists())
                self.assertFalse((root / "rollout/explore_wandb/done.json").exists())
                result = eval_explore.run(ctx, root / "rollout", "dp", "dyn")
                eval_explore.run(ctx, root / "rollout", "dp", "dyn")
            self.assertEqual(result["wandb_run_id"], "original-id")
            self.assertEqual(calls.count("scout.eval.run_rollout"), 1)
            self.assertEqual(calls.count("scripts.atom.shard_rollout"), 1)
            self.assertEqual(calls.count("scripts.atom.log_explore"), 2)

    def test_explore_publisher_uses_existing_run_and_history(self):
        for tries in (5, 10):
            with self.subTest(tries=tries):
                run = Mock()
                wandb = Mock()
                wandb.init.return_value = run
                with patch.dict(sys.modules, wandb=wandb):
                    log_explore.publish(dict(explore_try_times=tries, n_failed=80,
                                             pass_at_5=.4, exploration_rescued=20, avg_jerk=None),
                                        project="p", run_id="existing", name="DP-round1",
                                        mode="online", directory="unused")
                self.assertEqual(wandb.init.call_args.kwargs["id"], "existing")
                self.assertEqual(wandb.init.call_args.kwargs["resume"], "must")
                payload = run.log.call_args.args[0]
                self.assertEqual(payload[f"explore/pass@{tries}"], .4)
                self.assertEqual(payload["explore/success_count"], 20)
                self.assertNotIn("step", run.log.call_args.kwargs)
                self.assertNotIn("explore/avg_jerk", payload)
                run.finish.assert_called_once_with()

    def test_cleanup_removes_only_explicit_shards_after_successful_merge(self):
        for fail_merge in (False, True):
            with self.subTest(fail_merge=fail_merge), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                outputs = root / "outputs"
                outputs.mkdir()
                directory = root / "rollouts" / "attempt"
                directory.mkdir(parents=True)
                shard = directory.with_name("attempt-shard0of1")
                shard.mkdir()
                (shard / "log.json").touch()
                unrelated = directory.with_name("attempt-shard1of2")
                unrelated.mkdir()
                paths = [outputs / p for p in ("explore.json", "success.hdf5", "all.hdf5")]
                for p in paths:
                    shard_rollout._suffix(p, 0, 1).touch()
                process = Mock()
                process.wait.return_value = 0
                process.poll.return_value = 0
                argv = ["1", *map(str, paths), str(root / "core.hdf5"), "--",
                        "--output-dir", str(directory)]
                with patch.object(shard_rollout.subprocess, "Popen", return_value=process), \
                        patch.object(shard_rollout.subprocess, "run",
                                     side_effect=subprocess.CalledProcessError(1, "merge") if fail_merge else None), \
                        patch.dict(shard_rollout.os.environ, CLEANUP_SHARDS="1"), redirect_stdout(io.StringIO()):
                    if fail_merge:
                        with self.assertRaises(subprocess.CalledProcessError):
                            shard_rollout.main(argv)
                    else:
                        shard_rollout.main(argv)
                self.assertEqual(shard.exists(), fail_merge)
                self.assertEqual(shard_rollout._suffix(paths[0], 0, 1).exists(), fail_merge)
                self.assertTrue(unrelated.exists())
                self.assertTrue(directory.exists())


if __name__ == "__main__":
    unittest.main()
