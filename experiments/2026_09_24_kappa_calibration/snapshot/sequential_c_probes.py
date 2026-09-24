import json,subprocess,sys
from pathlib import Path
r=Path('/mnt/workspace/baojiachun/scout');e=r/'experiments/2026_09_24_kappa_calibration';py='/mnt/workspace/baojiachun/.venv_mg/bin/python'
task,gpu,uuid=sys.argv[1:]
base=e/'base'/task/'diagnostics/k2.5.json'
eta=json.loads(base.read_text())['eta'];cap=2.5
for rn in ['round1','round2']:
 source=e/rn/task;out=source/'sequential_c.json'
 cmd=[py,'scripts/calibration/kappa_pcalib.py','--source',str(source),'--c-reference',str(base),'--sequential-c','--initial-kappa',str(cap),'--eta-initial',str(eta),'--gpu',gpu,'--gpu-uuid',uuid,'--out',str(out)]
 with (e/f'{task}_{rn}_sequential_c_calib.driver.log').open('x') as log:
  p=subprocess.run(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT)
 p.check_returncode()
 d=json.loads(out.read_text());assert d['C_converged'] and d['R_target'] is None
 cap,eta=d['kappa'],d['eta']
(e/f'{task}_sequential_c_probes.done.json').write_text(json.dumps({'completed':True,'rounds':2}))
