from pathlib import Path
import subprocess,os,json
root=Path('/mnt/workspace/baojiachun/scout');E=root/'experiments/2026_09_24_kappa_calibration';py='/mnt/workspace/baojiachun/.venv_mg/bin/python'
rows=[]
for n in [1,2]:
    for task in ['can','coffee','threading','tool_hang']:
        source=E/f'round{n}'/task
        reference=json.loads((source/'dp_diversity/summary.json').read_text())
        cap=reference['independent_final_to_anchor']['quantiles']['p50']
        output=source/f'diagnostics_rmatched/k{cap:g}.json'
        if not output.exists():
            reading=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert reading[0].strip()=='GPU-76d481c4-5473-2f05-5171-0a11507e8a0f' and int(reading[1])<128,reading
            command=[py,'-u','scripts/calibration/kappa_diagnostics.py','--root',str(source.parent),'--tasks',task,'--kappas',str(cap),'--target-r','.01']
            with (source/'kl_median_calibration.stdout').open('w') as log:
                subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        d=json.loads(output.read_text());assert d['kappa']==cap and d['R_target']==.01
        rows.append(dict(round=n,task=task,kappa=cap,eta=d['eta'],R_mean=d['R_mean'],R_converged=d['R_converged'],source=str(output)))
        print(rows[-1],flush=True)
(E/'round_median_dose_probes.done.json').write_text(json.dumps(rows,indent=2)+'\n')
