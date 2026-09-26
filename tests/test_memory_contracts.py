"""Storage equivalence, metric-only protocol, and the two guidance contracts."""
import copy
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch

from scout.eval.hdf5_writer import _stack_obs, write_rollouts_to_hdf5
from scout.eval.observation_storage import storage_snapshot
from scout.eval.rollout import make_obs_adapter
from scout.eval.rollout_vec import _VecRunner, evaluate_exploration_vec
from scout.eval.traj_spool import TrajSpool
from scout.guidance.cost import encode_action_chunk


class Env:
    rollout_exceptions = ()
    def reset_to(self, state):
        self.i = 0; self.goal = state['goal']; return self.obs()
    def obs(self):
        return {'sideview_image': np.full((2, 3, 8, 8), self.i / 255., np.float32),
                'robot0_eef_pos': np.full((2, 3), self.i, np.float32)}
    def get_state(self):
        return np.array([self.i], np.float32)
    def step(self, action):
        self.i += 1
        return self.obs(), float(action[0]), self.i == 7, {}
    def is_success(self):
        return {'task': self.i >= self.goal}
    def close(self):
        pass


class Policy:
    def predict_action(self, obs):
        b = next(iter(obs.values())).shape[0]
        return {'action': torch.ones(b, 2, 7)}


class MemoryContracts(unittest.TestCase):
    def test_observation_storage_is_exact_and_eight_times_smaller(self):
        rng = np.random.default_rng(42)
        raw = [{'sideview_image': rng.uniform(-.1, 1.1, (2, 3, 84, 84)).astype(np.float32),
                'robot0_eef_pos': rng.normal(size=(2, 3)).astype(np.float32)} for _ in range(8)]
        packed = [storage_snapshot(o) for o in raw]
        for key in raw[0]:
            np.testing.assert_array_equal(_stack_obs(raw, key, 8), _stack_obs(packed, key, 8))
        self.assertEqual(sum(o['sideview_image'].nbytes for o in raw),
                         8 * sum(o['sideview_image'].nbytes for o in packed))
        expected = packed[0]['sideview_image'].copy()
        raw[0]['sideview_image'].fill(0)
        np.testing.assert_array_equal(packed[0]['sideview_image'], expected)

    def episodes(self, compact):
        rows = []
        def done(slot):
            rows.append(copy.deepcopy(slot.traj))
        runner = _VecRunner(Policy(), Env, 3, 2, 7, 'cpu', guided=False,
                            on_done=done, compact_obs=compact)
        # Numeric state allows HDF5 state comparison too.
        runner.run(deque([({'goal': g}, i, 0) for i, g in enumerate((3, 20, 5, 4))]), record_obs=True)
        if compact:
            self.assertTrue(all(s.traj is None and not s.obs_list for s in runner.slots))
        runner.close()
        for row in rows:
            # Initial state dictionaries aren't serialized by the real writer;
            # avoid the fake env's dict first state when constructing its test file.
            row['states'] = [np.array([i], np.float32) for i in range(row['horizon'])]
        return rows

    def test_rollout_and_full_hdf5_values_match(self):
        raw, packed = self.episodes(False), self.episodes(True)
        for a, b in zip(raw, packed):
            for key in ('actions', 'dones', 'rewards', 'success', 'horizon'):
                np.testing.assert_array_equal(a[key], b[key])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); core = root / 'core.hdf5'
            with h5py.File(core, 'w') as f:
                g = f.create_group('data/demo_0'); g.attrs['num_samples'] = 1
                g.create_dataset('actions', data=np.zeros((1, 7), np.float32))
                g.create_dataset('obs/sideview_image', data=np.zeros((1, 8, 8, 3), np.uint8))
                g.create_dataset('obs/robot0_eef_pos', data=np.zeros((1, 3), np.float32))
                f['data'].attrs['env_args'] = '{}'; f['data'].attrs['total'] = 1
            write_rollouts_to_hdf5(str(core), str(root / 'a.hdf5'), raw)
            write_rollouts_to_hdf5(str(core), str(root / 'b.hdf5'), packed)
            # Exercise the production spool, including delayed first failures.
            sp = TrajSpool(str(core), str(root / 'success.hdf5'), str(root / 'spool.hdf5'),
                           rule='split', flush_every=2, verbose=False)
            for i, row in enumerate(packed): sp.on_traj(row, i, 0)
            sp.finalize(); sp.close()
            with h5py.File(root / 'a.hdf5') as a:
                for path in ('b.hdf5', 'spool.hdf5'):
                    with h5py.File(root / path) as b:
                        names = []; a.visit(names.append)
                        other = []; b.visit(other.append)
                        self.assertEqual(names, other)
                        for name in names:
                            if isinstance(a[name], h5py.Dataset):
                                np.testing.assert_array_equal(a[name][()], b[name][()], err_msg=name)

    def test_metrics_only_does_not_change_actions_retries_or_success(self):
        def run(metrics_only):
            rows = []
            def sink(traj, i, j):
                rows.append((i, j, traj['success'], traj['actions'].copy(), len(traj['obs'])))
            sink.metrics_only = metrics_only
            result = evaluate_exploration_vec(Policy(), Env, [{'goal': 3}, {'goal': 20}],
                horizon=7, try_times=5, n_envs=3, n_action_steps=2, device='cpu', guided=False,
                traj_sink=sink, stop_on_first_success=True)
            return result, rows
        _, raw = run(False); _, slim = run(True)
        self.assertEqual(len(raw), 6)
        for a, b in zip(raw, slim):
            self.assertEqual(a[:3], b[:3]); np.testing.assert_array_equal(a[3], b[3])
            self.assertGreater(a[4], 0); self.assertEqual(b[4], 0)

    def test_guidance_image_scale_matches_training_range(self):
        obs = {'view': torch.full((2, 1, 3, 84, 84), .5), 'pos': torch.ones(2, 1, 3)}
        actual = make_obs_adapter(['view'], ['pos'])(obs)
        self.assertEqual(actual['visual']['view'].shape, (2, 1, 3, 76, 76))
        torch.testing.assert_close(actual['visual']['view'], torch.full((2, 1, 3, 76, 76), .5))

    def test_guidance_only_backpropagates_through_executed_actions(self):
        x = torch.arange(160., requires_grad=True).reshape(1, 16, 10); x.retain_grad()
        result = encode_action_chunk(x, SimpleNamespace(action_dim=80), lambda t: t * 2, 1, 8)
        torch.testing.assert_close(result, (x[:, 1:9] * 2).reshape(1, 80))
        result.sum().backward()
        self.assertEqual(x.grad[:, :1].count_nonzero(), 0)
        self.assertEqual(x.grad[:, 9:].count_nonzero(), 0)
        self.assertEqual(x.grad[:, 1:9].count_nonzero(), 80)
        with self.assertRaises(ValueError):
            encode_action_chunk(x, SimpleNamespace(action_dim=80), lambda t: t, 1, 7)


if __name__ == '__main__': unittest.main()
