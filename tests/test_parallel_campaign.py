"""Shared-base scheduling and stale process safeguards; no training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.atom import parallel_campaign as parallel
from scripts.atom.common import read_json, write_json


class ParallelCampaignTests(unittest.TestCase):
    def test_base_barrier_assigns_shared_pairs_to_all_six_arms(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'core').touch()
            (root / 'eval').touch()
            manifest = dict(root=str(root), jobs={})
            pairs = {}
            for seed in (233, 2333, 23333):
                base = f's{seed}-base'
                config = dict(task='tool_hang', gpu=0, seed=seed, output_dir=str(root / base),
                              core_hdf5=str(root / 'core'), eval_config=str(root / 'eval'),
                              rollout={'workers': 8, 'envs_per_worker': 25})
                path = root / (base + '.json')
                write_json(path, config)
                manifest['jobs'][base] = dict(kind='base', config=str(path))
                pairs[base] = {}
                for key in ('dp', 'dyn'):
                    ckpt = root / f'{base}-{key}.ckpt'
                    ckpt.touch()
                    pairs[base][key] = str(ckpt)
                for arm in ('dp', 'aty'):
                    name = f's{seed}-{arm}'
                    path = root / (name + '.json')
                    write_json(path, dict(config, output_dir=str(root / name)))
                    manifest['jobs'][name] = dict(kind='chain', config=str(path), base_job=base)
            manifest_path = root / 'manifest.json'
            write_json(manifest_path, manifest)
            batches = []

            def launch(path, m, names):
                batches.append(names)
                if len(batches) == 1:
                    self.assertEqual(len(names), 3)
                    for name in names:
                        write_json(parallel.status_path(m, name), dict(status='completed', result=pairs[name]))
                else:
                    self.assertEqual(len(names), 6)
                    for name in names:
                        spec = m['jobs'][name]
                        config = read_json(spec['config'])
                        pair = pairs[spec['base_job']]
                        self.assertEqual((config['base_dp'], config['base_dyn']), (pair['dp'], pair['dyn']))
                        self.assertEqual(config['rollout'], {'workers': 8, 'envs_per_worker': 25})

            with patch.object(parallel, 'launch_and_wait', side_effect=launch):
                parallel.supervise(manifest_path)
            self.assertEqual(len(batches), 2)
            self.assertTrue((root / 'completed.json').exists())

    def test_pid_reuse_is_not_treated_as_active_training(self):
        with patch.object(Path, 'read_bytes', return_value=b'python\0unrelated_job.py\0'):
            self.assertFalse(parallel.active(dict(status='running', pid=123), 's233-aty', Path('/manifest')))

    def test_completed_jobs_are_not_launched_again(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = dict(root=str(root), jobs={'job': {}})
            write_json(parallel.status_path(manifest, 'job'), dict(status='completed'))
            with patch.object(parallel.subprocess, 'Popen') as spawn:
                parallel.launch_and_wait(root / 'manifest.json', manifest, ['job'])
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
