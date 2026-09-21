#!/bin/bash
set -u
cd /root/workspace/baojiachun/scout
bash scripts/coffee/cfk_reanchor_r6probe.sh 1 233   3.8  1.15 &
bash scripts/coffee/cfk_reanchor_r6probe.sh 2 233   3.8  1.49 &
bash scripts/coffee/cfk_reanchor_r6probe.sh 3 2333  2.67 3.694 &
bash scripts/coffee/cfk_reanchor_r6probe.sh 4 23333 4.45 1.982 &
wait
echo REANCHOR_ALLDONE
