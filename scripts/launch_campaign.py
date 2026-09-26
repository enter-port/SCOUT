"""One-command registered, shared-base, multi-seed DP / SCOUT-aty campaign.

Example (from repo root): python -m scripts.launch_campaign --experiment-id ID
  --seeds 233 2333 23333 --methods DP SCOUT-aty --core-demos 40
  --gpus 0 1 2 3 4 5 --beta 1e-6 --wandb-prefix TOOL_HANG-2026-09-25

Use --prepare-only to inspect configs before launching. Repeating the command
reuses an identical manifest; changed settings require an explicitly cleaned root.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.atom.common import ROOT, read_json, write_json


def registry():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import experiment_registry
    return experiment_registry


def config_template(task, beta, rounds):
    plan = []
    for n in range(1, rounds + 1):
        plan.append(dict(rounds=[n], train=n < rounds,
                         calib={'mode': 'r' if n == 1 else ('rc' if n < rounds else 'none')}))
    return dict(version=3, task=task, arms=['ATY'], rounds=rounds,
        wandb_mode='online', dp_config=f'{task}/base_dp', dyn_config=f'configs/{task}/dyn.yaml',
        eval_config=f'configs/{task}/eval.yaml',
        dp=dict(base_epochs=600, base_checkpoint_every=100, epochs=600,
                checkpoint_every=300, batch_size=64, workers=8),
        dyn=dict(base_epochs=300, epochs=300, batch_size=64, steps_per_epoch=100, beta=beta),
        dose=dict(eta=1., kappa=2.5), grid=dict(enabled=False),
        calib=dict(target_r=.01, band=.1, batch_size=128, r_max_repeat=8,
                   c_solver='bracket', c_max_probes=12, c_kappa_min=.001, c_kappa_max=100.),
        round_plan=plan, rollout=dict(scenes=100, seed=42, pass_k=5, workers=8,
                    eval_envs=25, envs_per_worker=25, flush_every=100, stop_on_first_success=True))


def process_matches(pid, module, path=None, name=None):
    """Check identity and zombie status, resolving mount aliases."""
    try:
        import psutil
        p = psutil.Process(int(pid))
        cmd = p.cmdline()
        if p.status() == psutil.STATUS_ZOMBIE or module not in cmd:
            return False
        paths = [Path(v).resolve() for v in cmd if v.startswith('/')]
        return (path is None or Path(path).resolve() in paths) and (name is None or name in cmd)
    except (OSError, ValueError, KeyError, psutil.Error):
        return False


def spawn(module, args, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a') as stream:
        return subprocess.Popen([sys.executable, '-u', '-m', module, *map(str, args)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True).pid


def prepare(args):
    reg = registry()
    record = reg.find(args.experiment_id)
    if record['category'] != 'full' or record['task'] != args.task.upper():
        raise ValueError('Experiment identity/task/category mismatch')
    root = (reg.BASE / record['data_path']).resolve()
    arms = ['ATY' if m == 'SCOUT-aty' else m for m in args.methods]
    if len(set(arms)) != len(arms) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds or methods')
    if len(args.gpus) < len(args.seeds) * len(arms) or len(set(args.gpus)) != len(args.gpus) or 7 in args.gpus:
        raise ValueError('Require one distinct GPU per arm; GPU7 is reserved')
    source = Path(args.source or ROOT / f'data/robomimic/{args.task}/ph/image_v141_abs.hdf5').resolve()
    settings = dict(seeds=args.seeds, arms=arms, core_demos=args.core_demos, beta=args.beta,
                    rounds=args.rounds, gpus=args.gpus, source=str(source), wandb_prefix=args.wandb_prefix)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path).get('settings') != settings:
            raise ValueError('Existing manifest differs; refusing to overwrite active or completed work')
        return manifest_path
    if list(root.iterdir()):
        raise ValueError('Fresh preparation requires an empty registered data root')
    import h5py
    import numpy as np
    with h5py.File(source) as f:
        total = len(f['data'])
    if not 0 < args.core_demos < total:
        raise ValueError('Invalid core demo count')
    template = config_template(args.task, args.beta, args.rounds)
    root.joinpath('cores').mkdir()
    jobs = {}
    for i, seed in enumerate(args.seeds):
        core = root / f'cores/core{args.core_demos}_s{seed}.hdf5'
        subprocess.run([sys.executable, '-m', 'scripts.atom.split_core', str(source),
                        str(core), str(args.core_demos), str(seed)], cwd=ROOT, check=True)
        indices = sorted(map(int, np.random.default_rng(seed).choice(total, args.core_demos, replace=False)))
        write_json(core.with_suffix('.json'), dict(seed=seed, source=str(source),
                   source_demo_indices=indices, source_count=total, n=args.core_demos))
        cfg = dict(copy.deepcopy(template), seed=seed, gpu=args.gpus[i],
                   core_hdf5=str(core), output_dir=str(root / f's{seed}/shared'),
                   wandb_project=f'{args.wandb_prefix}-s{seed}')
        base_name = f's{seed}-base'
        path = root / f'configs/{base_name}.json'
        write_json(path, cfg)
        jobs[base_name] = dict(kind='base', config=str(path))
        for j, arm in enumerate(arms):
            name = f's{seed}-{arm.lower()}'
            chain = dict(copy.deepcopy(cfg), gpu=args.gpus[i * len(arms) + j], arms=[arm],
                         output_dir=str(root / f's{seed}/{arm}'))
            path = root / f'configs/{name}.json'
            write_json(path, chain)
            jobs[name] = dict(kind='chain', config=str(path), base_job=base_name)
    manifest = dict(root=str(root), experiment_uid=record['uid'], settings=settings, jobs=jobs,
                    created_at=time.time(), generation='input-contract-memory-v1')
    write_json(manifest_path, manifest)
    write_json(root / 'monitor_policy.json', dict(enabled=set(arms) == {'DP', 'ATY'},
        threshold=4, comparison='any_seed', betas=[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5],
        state='initial', interval_minutes=30, calibration_wandb=False))
    r = reg.find(record['uid'])
    reg.update(r['uid'], {'manual_status': 'planned', 'configuration': settings}, reg.revision(r))
    return manifest_path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--experiment-id', required=True)
    p.add_argument('--task', default='tool_hang')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--methods', choices=['DP', 'SCOUT-aty'], nargs='+', default=['DP', 'SCOUT-aty'])
    p.add_argument('--core-demos', type=int, required=True)
    p.add_argument('--beta', type=float, default=1e-6)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--wandb-prefix', required=True)
    p.add_argument('--source')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.rounds < 2 or a.beta <= 0:
        p.error('Require rounds >= 2 and positive beta')
    manifest = prepare(a)
    root = manifest.parent
    if not a.prepare_only:
        state_path = root / 'supervisor.json'
        state = read_json(state_path) if state_path.exists() else {}
        if (not (root / 'completed.json').exists() and not process_matches(
                state.get('pid', -1), 'scripts.atom.parallel_campaign', manifest)):
            pid = spawn('scripts.atom.parallel_campaign', ['supervise', '--manifest', manifest],
                        root / 'logs/supervisor.log')
            write_json(state_path, dict(pid=pid, started_at=time.time()))
        reg = registry(); r = reg.find(a.experiment_id)
        reg.update(r['uid'], {'manual_status': 'running'}, reg.revision(r))
    print(manifest)


if __name__ == '__main__':
    main()
