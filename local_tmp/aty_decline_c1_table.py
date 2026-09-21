"""Cycle-1: quantify the ATY pass@5 decline from raw_metrics.txt (no server run).
Per seed x arm x round: eval_SR, n_failed, rescued, pass@5,
rescue_rate_on_failed = rescued/n_failed (composition-controlled),
avg_jerk (explore phase), n_success_trajs fed back.
pass@5 == eval_SR + rescued/100 (identity check)."""
import json, re, sys

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
    "experiments/2026_9_21_COFFEE-MG-p1_figs_summary/raw_metrics.txt", encoding="utf-8"))
table = {}
for r in rows:
    f = r.get("file", "")
    m = re.search(r"COFFEE-MG-p1-s(\d+)/coffee/rollout/([A-Z]+)-exp(\d+)/log/(.+)\.json$", f)
    if not m:
        continue
    seed, arm, rnd = m.group(1), m.group(2), int(m.group(3))
    kind = "explore" if "_explore_" in m.group(4) else "eval"
    table.setdefault((seed, arm, rnd), {})[kind] = r

for s in ["233", "2333", "23333"]:
    for a in ["DP", "ATY", "ORBIT"]:
        print(f"\n== seed {s} arm {a} ==")
        print(f"{'rnd':>3} {'evalSR':>7} {'nfail':>6} {'resc':>5} {'p5':>6} {'resc/nfail':>10} {'jerk':>8} {'nSuccFeed':>9} {'idOK':>5}")
        for n in range(1, 7):
            d = table.get((s, a, n), {})
            ev, ex = d.get("eval"), d.get("explore")
            if not ev or not ex:
                print(f"{n:>3}  MISSING")
                continue
            sr = ev["success_rate"]; nf = ex["n_failed"]; rc = ex["exploration_rescued"]
            p5 = ex["pass_at_5"]; jk = ex.get("avg_jerk"); ns = ex.get("n_success_trajs")
            rr = rc / nf if nf else float("nan")
            idok = abs((sr + rc / 100) - p5) < 1e-9
            print(f"{n:>3} {sr:>7.2f} {nf:>6} {rc:>5} {p5:>6.2f} {rr:>10.3f} "
                  f"{(jk if jk is not None else float('nan')):>8.3f} {ns:>9} {str(idok):>5}")
