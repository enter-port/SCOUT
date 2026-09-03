#!/bin/bash
# Fetch wave-1/1b/2/3 result jsons from the server into
# experiments/sq2_wave1/<name>.json, then run the mixing sim per seed.
# Usage: bash soe_scripts/sq2_fetch.sh [--sim]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/sq2_wave1
SSH="ssh -o BatchMode=yes -p 1022 root@106.14.2.243"
BASE=/root/workspace/baojiachun/scout-rand/data/particle
n=0
for name in $($SSH "ls -d $BASE/sq2_*/ 2>/dev/null | xargs -n1 basename"); do
  case $name in sq2_conf_*) continue ;; esac
  j=$BASE/$name/log/square_SCOUT_rollout_exp0.json
  if $SSH "test -f $j" 2>/dev/null; then
    $SSH "cat $j" > "experiments/sq2_wave1/$name.json"
    n=$((n+1))
  fi
done
echo "fetched $n jsons -> experiments/sq2_wave1/"
if [ "${1:-}" = "--sim" ]; then
  python soe_scripts/sq2_mix_sim.py \
    experiments/sq2_wave1/sq2_{plc,att,orb015,par}_s2333.json --target=43
  python soe_scripts/sq2_mix_sim.py \
    experiments/sq2_wave1/sq2_{plc,att,orb015,par}_s23333.json --target=42
  python soe_scripts/sq2_mix_sim.py \
    experiments/sq2_wave1/sq2_{plc,att,orb025,par}_s233.json --target=40
fi
