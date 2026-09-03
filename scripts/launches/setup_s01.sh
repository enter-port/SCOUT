#!/bin/bash
# SCOUT01 chains (user 2026-08-18): guidance_scale 0.1 variant, rounds 1-2 only,
# everything else identical to the current round1 recipe (beta 3e-5, fb 0.005,
# lambda 5, split protocol; e2 = 500 fresh scenes/round, e3 = 100 fixed scenes).
# The tag SCOUT01 keeps rollout dirs / wandb runs / dyn+DP walk-backs separate
# from the scale-0.5 SCOUT chains (all keyed on the a-tag in round_e2.sh).
set -eu
cd /root/workspace/baojiachun/scout

for e in e2 e3; do
  # eval config: only guidance_scale changes (0.5 -> 0.1)
  sed -e 's/^  guidance_scale: .*/  guidance_scale: 0.1             # 2026-08-18 user: SCOUT01 scale-sweep arm (0.5 in the main SCOUT chain)/' \
      configs/eval_can_${e}.yaml > configs/eval_can_${e}s01.yaml
  # driver: SCOUT -> SCOUT01 everywhere (validation, GUIDE branch, dirs,
  # wandb names, walk-back globs) + point at the s01 eval config
  sed -e 's/SCOUT/SCOUT01/g' \
      -e "s/eval_\${TASK}_${e}\.yaml/eval_\${TASK}_${e}s01.yaml/g" \
      soe_scripts/round_${e}.sh > soe_scripts/round_${e}s01.sh
  chmod +x soe_scripts/round_${e}s01.sh
done

echo "=== verify ==="
grep -E 'guidance_scale' configs/eval_can_e2s01.yaml configs/eval_can_e3s01.yaml
grep -E 'DP\|SCOUT01\)|= SCOUT01 \]' soe_scripts/round_e2s01.sh
grep -E 'eval_\$\{TASK\}' soe_scripts/round_e2s01.sh soe_scripts/round_e3s01.sh
