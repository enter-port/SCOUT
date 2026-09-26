"""Durable round-one beta gate for a registered multi-seed campaign.

check is the only monitor command. It returns health and a requested next check
interval, and performs the explicitly configured ATY-only transition. Search
uses the same eight scene slices, env count and retry protocol as formal runs.
Its three GPU slots are the paused ATY GPUs; healthy DP jobs remain independent.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from scripts.launch_campaign import registry, process_matches, spawn
from .common import Context, ROOT, lock, read_json, write_json
from . import calib, dyn_train


def discard_trajectory(*args):
    pass


discard_trajectory.metrics_only = True


def read_optional(path):
    return read_json(path) if Path(path).exists() else {}


def round_one(cfg):
    receipt = Path(cfg['output_dir']) / f"rounds/{cfg['arms'][0]}/round-1/rollout/explore/done.json"
    if not receipt.exists():
        return None
    result = read_json(receipt)['result']
    metrics = read_json(result['metrics'])
    opts = cfg['rollout']
    if (metrics['n_init_states'] != opts['scenes'] or
            metrics['explore_try_times'] != opts['pass_k'] or
            metrics['shards_merged'] != opts['workers']):
        raise ValueError(f'Incomplete result: {receipt}')
    return dict(metrics=metrics, receipt=result, rescued=int(metrics['exploration_rescued']))


def choose_trigger(pairs, threshold):
    """Wait for every pair, then use the largest per-seed integer deficit."""
    if not pairs or any(not row.get('DP') or not row.get('ATY') for row in pairs.values()):
        return None
    gaps = {int(seed): row['DP']['rescued'] - row['ATY']['rescued'] for seed, row in pairs.items()}
    seed = max(sorted(gaps), key=gaps.get)
    return seed if gaps[seed] >= threshold else False


def entered_round2_training(cfg):
    dp = Path(cfg['output_dir']) / 'rounds/ATY/round-2/dp'
    if (dp / 'done.json').exists():
        return True
    for path in dp.glob('attempt-*/train.log'):
        # CLI initialization alone is insufficient: require the actual epoch loop.
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 65536))
            if b'Training epoch' in f.read():
                return True
    return False


def stop_job(manifest_path, name, state, audit):
    """Stop a verified independent job tree, never its tmux/supervisor group."""
    import psutil
    if not process_matches(state.get('pid', -1), 'scripts.atom.parallel_campaign', manifest_path, name):
        # A vanished controller may leave children. Refuse destructive cleanup
        # until any exact-config orphan has been accounted for.
        cfg_path = Path(read_json(manifest_path)['jobs'][name]['config']).resolve()
        cfg = read_json(cfg_path); root = Path(cfg['output_dir']).resolve()
        orphans = []
        for p in psutil.process_iter(['pid', 'cmdline', 'status']):
            cmd = p.info['cmdline'] or []
            if p.info['status'] == psutil.STATUS_ZOMBIE:
                continue
            for arg in cmd:
                if arg.startswith('/'):
                    path = Path(arg).resolve()
                    if path == cfg_path or root == path or root in path.parents:
                        orphans.append(p.pid); break
        if orphans:
            raise RuntimeError(f'Unowned ATY processes must be recovered before cleanup: {orphans}')
        return
    parent = psutil.Process(state['pid'])
    if os.getsid(parent.pid) != parent.pid:
        raise RuntimeError('Refusing stop: job lacks its own process session')
    parent.send_signal(signal.SIGSTOP)
    processes = [parent, *parent.children(recursive=True)]
    write_json(audit, [dict(pid=p.pid, created=p.create_time(), cmd=p.cmdline())
                       for p in processes if p.is_running()])
    for p in processes:
        try:
            p.send_signal(signal.SIGSTOP)
        except psutil.NoSuchProcess:
            pass
    for p in reversed(processes):
        try:
            p.terminate(); p.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass
    # This host does not support pidfd_open; psutil.wait_procs is unsuitable.
    for _ in range(30):
        alive = [p for p in processes if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
        if not alive:
            return
        time.sleep(1)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    for _ in range(10):
        if not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in processes):
            return
        time.sleep(1)
    raise RuntimeError('ATY processes did not exit; cleanup aborted')


def delete_aty_runs(project, audit):
    import wandb
    api = wandb.Api(timeout=60)
    full = project if '/' in project else f'{api.default_entity}/{project}'
    doomed = [r for r in api.runs(full) if r.name.startswith('SCOUT-aty-round')]
    write_json(audit, [dict(id=r.id, name=r.name, project=full) for r in doomed])
    for r in doomed:
        r.delete(delete_artifacts=True)
    for _ in range(15):
        if not [r for r in wandb.Api(timeout=60).runs(full) if r.name.startswith('SCOUT-aty-round')]:
            return
        time.sleep(2)
    raise RuntimeError('Remote ATY deletion not yet verified')


def shard(config, slot):
    ctx = Context(read_json(config)); os.environ.update(ctx.env)
    import torch
    from scout.eval.factories import load_cfg, make_default_env_factory, make_lpb_dp_factory, make_scout_vib_factory
    from scout.eval.rollout_pipeline import RolloutPipeline
    dose = read_json(ctx.root / 'dose.json')
    opts = ctx.c['rollout']; dyn = dose['dyn']; dp = ctx.c['base_dp']
    def action(work):
        cfg = load_cfg(str(ctx.eval_config))
        cfg.base_dp.initial_ckpt_path = dp
        cfg.vib.ckpt_path, cfg.vib.base_dp_ckpt = dyn, dp
        cfg.dataset.path = str(ctx.core)
        cfg.eval.n_init_states, cfg.eval.n_envs, cfg.eval.seed = opts['scenes'], opts['envs_per_worker'], opts['seed']
        cfg.exploration.guidance_scale = dose['eta']
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        device = torch.device('cuda')
        def progress(phase, payload, baseline_solved=0, n_total=0):
            write_json(work / 'progress.json', dict(time=time.time(), phase=phase, payload=payload))
        pipe = RolloutPipeline(cfg, make_lpb_dp_factory(device), make_scout_vib_factory(cfg, device),
            make_default_env_factory(cfg), device=device, guided=True, guide_mode='atypical',
            entropy_kwargs={'atypical_cap': dose['kappa'], 'eta_dimless': False},
            failed_set_json=ctx.c['failed_set_json'])
        result = pipe.run(dp, vib_ckpt=dyn, on_progress=progress, explore_mode='rescue',
            explore_try_times=opts['pass_k'], stop_on_first_success=opts['stop_on_first_success'],
            scene_slice=(slot, opts['workers']), traj_sink=discard_trajectory)
        metrics = dict(result['metrics'], task=ctx.task, seed=ctx.seed, primary_chain=False,
            n_init_states=opts['scenes'], explore_try_times=opts['pass_k'], beta=ctx.c['dyn']['beta'])
        write_json(work / 'result.json', metrics)
        return {'json': str(work / 'result.json'), 'artifacts': [str(work / 'result.json')]}
    ctx.stage(ctx.root / f'pass5/shard{slot}', {'dose': dose, 'slot': slot}, action)


def candidate(config):
    ctx = Context(read_json(config)); ctx.root.mkdir(parents=True, exist_ok=True)
    if ctx.wandb != 'disabled':
        raise ValueError('Calibration must not publish W&B')
    with lock(ctx.root / '.candidate.lock'):
        def status(state, **fields):
            write_json(ctx.root / 'status.json', dict(state=state, pid=os.getpid(), time=time.time(), **fields))
        try:
            status('running', phase='dyn')
            trained = dyn_train.train(ctx, ctx.root / 'dyn', ctx.c['base_dp'], base=True)
            status('running', phase='R')
            try:
                dose = calib.calibrate(ctx, ctx.root / 'calib', ctx.c['base_dp'], trained['dyn'],
                                      ctx.c['dose'], mode='r')
            except ValueError as e:
                if 'calibration' not in str(e):
                    raise
                status('unusable', reason=str(e)); return
            write_json(ctx.root / 'dose.json', dict(dose, dyn=trained['dyn']))
            status('running', phase='pass5')
            def one(slot):
                ctx.module('scripts.atom.beta_gate', ['shard', '--config', config, '--slot', slot],
                           ctx.root / f'logs/shard{slot}.log')
                return read_json(ctx.root / f'pass5/shard{slot}/done.json')['result']['json']
            with ThreadPoolExecutor(max_workers=ctx.c['rollout']['workers']) as pool:
                paths = list(pool.map(one, range(ctx.c['rollout']['workers'])))
            from scout.eval.merge_sharded import merge_metrics
            metrics = merge_metrics([read_json(p) for p in paths])
            failed = read_json(ctx.c['failed_set_json']); opts = ctx.c['rollout']
            if (metrics['shards_merged'] != opts['workers'] or metrics['n_init_states'] != opts['scenes']
                    or metrics['baseline_solved'] != failed['baseline_solved']
                    or metrics['failed_init_indices'] != sorted(failed['failed_init_indices'])):
                raise ValueError('Incomplete or mismatched frozen scene coverage')
            metrics.update(primary_chain=False, seed=ctx.seed, beta=ctx.c['dyn']['beta'])
            write_json(ctx.root / 'result.json', metrics)
            write_json(ctx.root / 'done.json', dict(metrics=metrics, dyn=trained['dyn'], dose=dose, config=str(config)))
            status('completed')
        except BaseException:
            status('failed', error=traceback.format_exc()); raise


def search(root):
    m = read_json(root / 'manifest.json')
    with lock(root / '.supervisor.lock'):
        processes = {}
        while True:
            busy, pending, failures = set(), [], []
            for config in m['configs']:
                cfg = read_json(config); cr = Path(cfg['output_dir']); s = read_optional(cr / 'status.json')
                child = processes.get(config)
                alive = (child is not None and child.poll() is None) or process_matches(
                    s.get('pid', -1), 'scripts.atom.beta_gate', config, 'candidate')
                if alive:
                    busy.add(cfg['gpu'])
                elif s.get('state') in ('completed', 'unusable'):
                    continue
                elif s.get('state') in ('running', 'failed'):
                    failures.append(config)
                else:
                    pending.append((config, cfg))
            for config, cfg in pending[:]:
                if cfg['gpu'] in busy:
                    continue
                log = root / 'logs' / f"{Path(config).stem}.log"; log.parent.mkdir(exist_ok=True)
                with log.open('a') as stream:
                    processes[config] = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.atom.beta_gate',
                        'candidate', '--config', config], cwd=ROOT, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                busy.add(cfg['gpu']); pending.remove((config, cfg))
            write_json(root / 'status.json', dict(state='running', pid=os.getpid(), time=time.time(), failures=failures))
            if not busy and not pending:
                if failures:
                    write_json(root / 'status.json', dict(state='needs_recovery', failures=failures)); return
                results = [read_json(Path(read_json(p)['output_dir']) / 'done.json') for p in m['configs']
                           if (Path(read_json(p)['output_dir']) / 'done.json').exists()]
                if not results:
                    write_json(root / 'status.json', dict(state='no_valid_candidate')); return
                best = max(results, key=lambda r: (r['metrics']['exploration_rescued'],
                           -abs(r['metrics']['beta'] - m['initial_beta']), -r['metrics']['beta']))
                write_json(root / 'selection.json', best)
                write_json(root / 'status.json', dict(state='selection_ready')); return
            time.sleep(10)


def begin_search(manifest, state, pairs):
    root = Path(manifest['root']); reg = registry()
    if 'aux_uid' not in state:
        record = reg.create(dict(task=read_json(next(iter(manifest['jobs'].values()))['config'])['task'].upper(),
            category='auxiliary', title='Round1 rescue beta calibration',
            configuration={'parent_uid': manifest['experiment_uid'], 'betas': state['betas'], 'primary_chain': False},
            expected_rounds=1, tags=['beta-calibration'], notes='No W&B; metrics-only pass@5, fixed base DP/failures.'))
        # registry.create returns the record (some versions wrap it in detail).
        record = record.get('record', record)
        state['aux_uid'] = record['uid']
        write_json(root / 'monitor_policy.json', state)
    record = reg.find(state['aux_uid']); aux = (reg.BASE / record['data_path']).resolve()
    audit = reg.folder(reg.find(manifest['experiment_uid'])) / 'history' / f"beta-gate-{record['uid']}"
    audit.mkdir(parents=True, exist_ok=True)
    # Stop all three before network deletion; W&B latency must not let later
    # seeds keep training a configuration already rejected by this gate.
    for seed in manifest['settings']['seeds']:
        name = f's{seed}-aty'; status = root / f'status/{name}.json'
        if not (audit / f'{name}-cleaned.json').exists():
            stop_job(root / 'manifest.json', name, read_optional(status), audit / f'{name}-processes.json')
            write_json(status, dict(status='paused', reason='round1 beta gate'))
    for seed in manifest['settings']['seeds']:
        name = f's{seed}-aty'; cfg = read_json(manifest['jobs'][name]['config'])
        done = audit / f'{name}-cleaned.json'
        if done.exists():
            continue
        delete_aty_runs(cfg['wandb_project'], audit / f'{name}-wandb.json')
        arm_root = Path(cfg['output_dir']).resolve()
        if arm_root != root.resolve() / f's{seed}/ATY':
            raise ValueError('Unsafe ATY cleanup root')
        write_json(audit / f'{name}-config.json', cfg)
        write_json(audit / f'{name}-result.json', pairs.get(str(seed), pairs.get(seed)))
        if arm_root.exists():
            shutil.rmtree(arm_root)
        write_json(done, {'cleaned_at': time.time()})
    if not (aux / 'manifest.json').exists():
        seed = state['trigger_seed']
        cfg = copy.deepcopy(read_json(manifest['jobs'][f's{seed}-aty']['config']))
        base = read_json(root / f'status/s{seed}-base.json')['result']
        cfg.update(base_dp=base['dp'], wandb_mode='disabled', failed_set_json=pairs[str(seed)]['DP']['receipt']['failed'])
        cfg.pop('base_dyn', None)
        gpus = [read_json(manifest['jobs'][f's{s}-aty']['config'])['gpu'] for s in manifest['settings']['seeds']]
        configs = []
        for i, beta in enumerate(state['betas']):
            c = copy.deepcopy(cfg); name = f'beta-{beta:g}'
            c.update(gpu=gpus[i % len(gpus)], output_dir=str(aux / 'candidates' / name))
            c['dyn']['beta'] = beta
            path = aux / f'configs/{name}.json'; write_json(path, c); configs.append(str(path))
        write_json(aux / 'manifest.json', dict(configs=configs, initial_beta=manifest['settings']['beta']))
    state.update(state='searching', aux_root=str(aux), audit=str(audit))
    write_json(root / 'monitor_policy.json', state)
    state['search_pid'] = spawn('scripts.atom.beta_gate', ['search', '--root', aux], aux / 'logs/supervisor.log')
    write_json(root / 'monitor_policy.json', state)
    r = reg.find(record['uid']); reg.update(r['uid'], {'manual_status': 'running'}, reg.revision(r))


def check(root):
    manifest = read_json(root / 'manifest.json'); policy_path = root / 'monitor_policy.json'
    with lock(root / '.monitor.lock'):
        state = read_json(policy_path)
        health = {}
        for name, spec in manifest['jobs'].items():
            s = read_optional(root / f'status/{name}.json')
            health[name] = ('completed' if s.get('status') == 'completed' else 'running' if
                process_matches(s.get('pid', -1), 'scripts.atom.parallel_campaign', root / 'manifest.json', name)
                else s.get('status', 'pending'))
            if health[name] == 'running' and not process_matches(s.get('pid', -1),
                    'scripts.atom.parallel_campaign', root / 'manifest.json', name):
                health[name] = 'missing'
            elif health[name] == 'running' and s.get('status') != 'running':
                health[name] = 'starting'
        event = None
        if state.get('enabled') and state['state'] == 'initial':
            pairs = {str(seed): {arm: round_one(read_json(manifest['jobs'][f's{seed}-{arm.lower()}']['config']))
                     for arm in ('DP', 'ATY')} for seed in manifest['settings']['seeds']}
            trigger = choose_trigger(pairs, state['threshold'])
            if trigger is not None:
                state['round1_comparison'] = pairs
                if trigger is not False:
                    state.update(state='pausing', trigger_seed=trigger)
                else:
                    state['state'] = 'accepted'
                    event = 'round1 accepted'
                write_json(policy_path, state)
        if state['state'] == 'pausing':
            begin_search(manifest, state, state['round1_comparison'])
            event = 'paused three ATY chains; beta calibration started'
        if state['state'] == 'searching':
            aux = Path(state['aux_root']); selection = read_optional(aux / 'selection.json')
            if selection:
                state.update(state='restarting', selection=selection, restarted=[])
                write_json(policy_path, state)
            else:
                s = read_optional(aux / 'status.json')
                if s.get('state') in ('needs_recovery', 'no_valid_candidate'):
                    event = s
                elif not process_matches(s.get('pid', state.get('search_pid', -1)), 'scripts.atom.beta_gate', aux, 'search'):
                    event = 'search supervisor missing; inspect candidate identities before recovery'
        if state['state'] == 'restarting':
            best = state['selection']; beta = best['metrics']['beta']
            for seed in manifest['settings']['seeds']:
                if seed in state['restarted']:
                    continue
                name = f's{seed}-aty'; path = manifest['jobs'][name]['config']; cfg = read_json(path)
                s = read_optional(root / f'status/{name}.json')
                if process_matches(s.get('pid', -1), 'scripts.atom.parallel_campaign', root / 'manifest.json', name):
                    if cfg['dyn']['beta'] != beta:
                        raise RuntimeError('Unexpected live ATY generation')
                else:
                    cfg['dyn']['beta'] = beta
                    cfg.pop('base_dyn', None)
                    if seed == best['metrics']['seed']:
                        cfg['base_dyn'] = best['dyn']
                    write_json(path, cfg)
                    pid = spawn('scripts.atom.parallel_campaign', ['job', '--manifest', root / 'manifest.json',
                                '--name', name], root / f'logs/{name}-beta-{beta:g}.log')
                    # Allow the job to acquire its lock and write its own status.
                    for _ in range(30):
                        s = read_optional(root / f'status/{name}.json')
                        if s.get('pid') == pid and s.get('status') == 'running':
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError(f'Restarted {name} did not enter running state')
                state['restarted'].append(seed); write_json(policy_path, state)
            state.update(state='restarted', interval_minutes=60)
            event = f'three ATY restarted with beta={beta:g}; monitor hourly'
            reg = registry(); r = reg.find(state['aux_uid']); reg.sync(r['uid'])
            reg.update(r['uid'], {'manual_status': 'completed'}, reg.revision(reg.find(r['uid'])))
            write_json(policy_path, state)
        if state['state'] in ('accepted', 'restarted'):
            progressed = []
            for seed in manifest['settings']['seeds']:
                cfg = read_json(manifest['jobs'][f's{seed}-aty']['config'])
                progressed.append(health[f's{seed}-aty'] in ('running', 'completed') and
                                  entered_round2_training(cfg))
            if all(progressed):
                state.update(state='monitor_complete', enabled=False)
                write_json(policy_path, state); event = 'all three ATY entered round2 retraining; monitoring complete'
        result = dict(experiment_uid=manifest['experiment_uid'], state=state['state'], jobs=health,
                      interval_minutes=state['interval_minutes'], monitor_complete=state['state'] == 'monitor_complete', event=event)
        write_json(root / 'monitor_latest.json', result)
        print(__import__('json').dumps(result))
        return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['check', 'search', 'candidate', 'shard'])
    p.add_argument('--root', type=Path); p.add_argument('--config', type=Path); p.add_argument('--slot', type=int)
    a = p.parse_args()
    if a.action in ('check', 'search'):
        if not a.root: p.error('--root required')
        (check if a.action == 'check' else search)(a.root.resolve())
    else:
        if not a.config: p.error('--config required')
        if a.action == 'candidate': candidate(a.config.resolve())
        else: shard(a.config.resolve(), a.slot)


if __name__ == '__main__':
    main()
