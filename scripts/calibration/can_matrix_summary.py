"""Summarize the CAN exploit matrix (dump + best-guess table).

Per (seed, round): the exploit run json (skip-eval explore mode) raw
fields, plus the chain in-chain SR of the SAME ckpt (= json
SCOUT-exp{N+1}.success_rate, since round N+1's eval segment rolled out
DP-SCOUT-exp{N}; r6 has no in-chain eval). CPU-only.

Usage: can_matrix_summary.py
"""
import glob
import json

ENT = "/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy"
REPO = "/root/workspace/baojiachun/scout-exploit"
SEEDS = (233, 2333, 23333)
FIELDS = ("n_success_trajs", "n_all_trajs", "collected_trajs",
          "exploration_rescued", "pass_at_5", "avg_jerk", "jerk_explore")


def jr(pattern):
    js = glob.glob(pattern)
    return json.load(open(js[0])) if js else None


def main():
    for s in SEEDS:
        base = f"{ENT}/CAN-entropy-s{s}/can"
        for n in range(1, 7):
            d = jr(f"{REPO}/data/exploit_can_matrix/s{s}/r{n}/log/*.json")
            nxt = jr(f"{base}/rollout/SCOUT-exp{n + 1}/log/*.json") if n < 6 else None
            if d is None:
                print(f"s{s} r{n}: PENDING  chainSR={nxt and nxt.get('success_rate')}")
                continue
            vals = " ".join(f"{k}={d.get(k)}" for k in FIELDS)
            print(f"s{s} r{n}: {vals}  "
                  f"chainSR={nxt.get('success_rate') if nxt else 'NA'} "
                  f"skip_eval={d.get('skip_eval')} thr={open(f'{REPO}/data/exploit_can_matrix/s{s}/thr_r{n}.txt').read().strip() if glob.glob(f'{REPO}/data/exploit_can_matrix/s{s}/thr_r{n}.txt') else '?'}")


if __name__ == "__main__":
    main()
