import json, sys, time
import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

def scan(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        b = ds.scan_data()
        if b is None:
            return
        r = wandb_internal_pb2.Record()
        r.ParseFromString(b)
        yield r

_bad = []

def parse_val(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s

def recover(wandb_file, project, name, dyn_axes=False):
    hist = []
    for r in scan(wandb_file):
        if r.WhichOneof('record_type') != 'history':
            continue
        row = {}
        for it in r.history.item:
            if it.nested_key:
                # actual layout: nested_key[0] = metric name, value_json =
                # the value string split into single characters
                row[it.nested_key[0]] = parse_val(''.join(it.value_json))
            else:
                for k, v in zip(it.key, it.value_json):
                    row[k] = parse_val(v)
        if row:
            hist.append(row)
    print(f'{name}: {len(hist)} merged rows; sample {dict(list(hist[0].items())[:4]) if hist else None}')
    run = wandb.init(project=project, name=name,
                     tags=['recovered', 'from-local-wandb'],
                     notes=f'recovered from local .wandb: {wandb_file}')
    if dyn_axes:
        wandb.define_metric('dyn/epoch', hidden=True)
        for m in ('dyn/latent_mse', 'dyn/kl', 'dyn/lr'):
            wandb.define_metric(m, step_metric='dyn/epoch')
    t0 = time.time()
    for i, row in enumerate(hist):
        step = row.pop('_step', None)
        row.pop('_timestamp', None)
        row.pop('_runtime', None)
        if not row:
            continue
        if step is None:
            run.log(row)
        else:
            run.log(row, step=int(step))
        if i and i % 20000 == 0:
            print(f'  {i}/{len(hist)} rows, {time.time()-t0:.0f}s')
            sys.stdout.flush()
    run.finish()
    print(f'{name}: DONE -> {run.url}')

recover('/root/workspace/baojiachun/scout/data/experiment2/transport/train/dyn/dyn-base/wandb/run-20260819_165149-k68zgl6y/run-k68zgl6y.wandb',
        'TRANSPORT-experiment2', 'SCOUT-round0', dyn_axes=True)
recover('/root/workspace/baojiachun/scout/data/experiment2/transport/train/DP/DP-base/wandb/run-20260819_143205-rp3noggs/run-rp3noggs.wandb',
        'TRANSPORT-experiment2', 'DP-base', dyn_axes=False)
