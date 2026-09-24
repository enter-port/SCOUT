import subprocess,os,json
from pathlib import Path
r=Path('/mnt/workspace/baojiachun/scout');e=Path('/mnt/workspace/baojiachun/scout/experiments/2026_09_24_kappa_calibration');s=e/"round2/coffee";py='/mnt/workspace/baojiachun/.venv_mg/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES="5",PYTHONUNBUFFERED="1")
with (e/"coffee_round2_fixed_k1_after_r25_core.log").open("x") as log:
 subprocess.run([py,"scripts/calibration/kappa_diagnostics.py","--root",str(e/"round2"),"--tasks","coffee","--kappas","1","--eta",'1.1066641191457374'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
with (e/"coffee_round2_fixed_k1_after_r25.driver.log").open("x") as log:
 subprocess.run([py,"scripts/calibration/kappa_eval_arm.py","--source",str(s),"--calibration",str(s/"diagnostics/k1.json"),"--output",str(s/"fixed_k1_after_r25"),"--fixed-eta","--gpu","5","--gpu-uuid",'GPU-e6374d2f-198b-b58d-a46b-9d00b47406ea'],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
