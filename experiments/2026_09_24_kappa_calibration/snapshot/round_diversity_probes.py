from pathlib import Path
import subprocess,os
root=Path('/mnt/workspace/baojiachun/scout')
E=root/'experiments/2026_09_24_kappa_calibration'
py='/mnt/workspace/baojiachun/.venv_mg/bin/python'
for roundn in [1,2]:
    tasks=[task for task in ['can','coffee','threading','tool_hang'] if not (E/f'round{roundn}'/task/'dp_diversity/summary.json').exists()]
    if not tasks:continue
    reading=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert reading[0].strip()=='GPU-76d481c4-5473-2f05-5171-0a11507e8a0f' and int(reading[1])<128,reading
    command=[py,'-u','scripts/calibration/kappa_diagnostics.py','--root',str(E/f'round{roundn}'),'--tasks',*tasks,'--diversity-samples','8']
    with (E/f'round{roundn}_diversity_probes.stdout').open('w') as log:
        subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
