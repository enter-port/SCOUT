"""Threshold, full-result requirements and destructive scope of the beta gate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import subprocess
import sys

from scripts.atom import beta_gate
from scripts.atom.common import write_json
from scripts.launch_campaign import config_template, process_matches, spawn


class BetaGateTests(unittest.TestCase):
    def test_threshold_is_four_and_waits_for_all_seeds(self):
        pairs = {'233': {'DP': {'rescued': 30}, 'ATY': {'rescued': 27}},
                 '2333': {'DP': {'rescued': 30}, 'ATY': {'rescued': 31}}}
        self.assertIs(beta_gate.choose_trigger(pairs, 4), False)
        pairs['233']['ATY']['rescued'] = 26
        self.assertEqual(beta_gate.choose_trigger(pairs, 4), 233)
        pairs['2333']['ATY'] = None
        self.assertIsNone(beta_gate.choose_trigger(pairs, 4))

    def test_rejects_partial_round_and_uses_rescue_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = config_template('tool_hang', 1e-6, 6)
            cfg['output_dir'] = d
            self.assertIsNone(beta_gate.round_one(cfg))
            result = root / 'metrics.json'
            receipt = root / 'rounds/ATY/round-1/rollout/explore/done.json'
            write_json(receipt, {'result': {'metrics': str(result)}})
            m = dict(n_init_states=100, explore_try_times=5, shards_merged=7, exploration_rescued=33)
            write_json(result, m)
            with self.assertRaises(ValueError): beta_gate.round_one(cfg)
            m['shards_merged'] = 8; write_json(result, m)
            self.assertEqual(beta_gate.round_one(cfg)['rescued'], 33)

    def test_six_round_budget_and_counts_are_preserved(self):
        c = config_template('tool_hang', 1e-6, 6)
        self.assertEqual(c['dyn']['beta'], 1e-6)
        self.assertEqual([r['train'] for r in c['round_plan']], [True]*5+[False])
        self.assertEqual([r['calib']['mode'] for r in c['round_plan']], ['r']+['rc']*4+['none'])
        self.assertEqual((c['rollout']['workers'], c['rollout']['envs_per_worker']), (8,25))
        self.assertEqual((c['dp']['epochs'], c['dyn']['epochs']), (600,300))

    def test_spawn_creates_independent_session(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.launch_campaign.subprocess.Popen') as p:
            spawn('module', [], Path(d)/'log')
            self.assertTrue(p.call_args.kwargs['start_new_session'])

    def test_unrelated_pid_is_not_live_job(self):
        import os
        self.assertFalse(process_matches(os.getpid(), 'unrelated.module'))

    @unittest.skipUnless(os.name == 'posix', 'OS sessions are used on Linux training hosts')
    def test_stopping_aty_tree_preserves_independent_dp(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d)/'manifest.json'; name = 's233-aty'
            protected = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
            target = subprocess.Popen([sys.executable, '-c',
                'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True); time.sleep(60)',
                'scripts.atom.parallel_campaign', str(manifest), name],
                stdout=subprocess.PIPE, text=True, start_new_session=True)
            import psutil
            child = psutil.Process(int(target.stdout.readline()))
            try:
                beta_gate.stop_job(manifest, name, {'pid': target.pid}, Path(d)/'audit.json')
                self.assertIsNotNone(target.poll())
                self.assertTrue(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE)
                self.assertIsNone(protected.poll())
            finally:
                for p in (target, protected):
                    if p.poll() is None: p.kill()
                    p.wait()
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE: child.kill()
                target.stdout.close()


if __name__ == '__main__': unittest.main()
