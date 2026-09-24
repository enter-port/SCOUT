"""Exercise the actual round calibration blocks without loading models or GPUs."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class PRIntegrationTests(unittest.TestCase):
    def run_block(self, driver, mode="success", dry=False):
        source = (ROOT / driver).read_text(encoding="utf-8")
        start = source.index('if [ "$A" = ATY ]; then\n  ETA_PREV=')
        block = source[start:source.index('\nT0=', start)]
        script = r'''
set -u
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
RDIR="$work" NUM=1 TASK=coffee GPU=0 PY=fake_python
A=ATY ETA0=1 ATT_CAP=2.5 ATY_SCALE=1 CALIB_MODE=pr
DPCKPT='checkpoint with spaces.ckpt' VIBCKPT=dyn.ckpt CORE=core.hdf5
log(){ echo "$*"; }
nvidia-smi(){ echo GPU-test; }
fake_python(){
  if [ "$1" = -c ]; then
    python "$@"
    return $?
  fi
  [ "$1" = -m ] && [ "$2" = scout.calib.joint_pr ] || return 90
  [ "$TEST_MODE" != fail ] || return 7
  while [ "$1" != --out ]; do shift; done
  if [ "$TEST_MODE" = invalid ]; then
    echo '{"eta":2,"kappa":3,"R_mean":0.03,"R_converged":true}' > "$2"
  else
    echo '{"eta":2,"kappa":3,"R_mean":0.01,"R_converged":true}' > "$2"
  fi
}
'''
        script += f'\nTEST_MODE={mode}\nDRY_RUN={int(dry)}\n' + block
        script += '\nprintf "PAIR=%s,%s\\n" "$ATY_SCALE" "$ATT_CAP"\n'
        return subprocess.run([BASH, "-c", script], cwd=ROOT, capture_output=True, text=True)

    def test_both_drivers_use_joint_pair_without_legacy_c(self):
        for driver in ["scripts/coffee/round_cfk5.sh", "scripts/threading/round_thm3.sh"]:
            with self.subTest(driver=driver):
                result = self.run_block(driver)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PAIR=2,3", result.stdout)

    def test_failed_solver_stops_round(self):
        result = self.run_block("scripts/coffee/round_cfk5.sh", mode="fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PAIR=", result.stdout)

    def test_wrong_measured_r_stops_round_even_if_flag_is_true(self):
        result = self.run_block("scripts/threading/round_thm3.sh", mode="invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PAIR=", result.stdout)

    def test_dry_run_does_not_require_gpu_or_models(self):
        result = self.run_block("scripts/coffee/round_cfk5.sh", mode="fail", dry=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN", result.stdout)
        self.assertIn("PAIR=1,2.5", result.stdout)


if __name__ == "__main__":
    unittest.main()
