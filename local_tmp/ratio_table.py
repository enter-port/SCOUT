"""Build the eta*|grad| / |noisy_action| table across 6 tasks.
Inputs (pulled from server into ./rtable_raw/): grad_probe_arm_<tag>.json
per ckpt + anorms.log NORM lines. R(delta) = eta*0.5*g_live(delta)/|a|,
0.5 = sqrt(1-abar) at the mid guided window; delta* = displacement where
base KL crosses kappa=2.5 (linear interp in delta)."""
import json, math, os, re, sys

RAW = os.path.dirname(os.path.abspath(__file__)) + "/rtable_raw"

TASKS = [  # task, eta, base_tag, r5_tag
    ("can",       3.0, "can_base",  "can_e6"),
    ("square",    3.0, "sq_base",   "sq_e6"),
    ("toolhang",  0.5, "th95_base", "th95_aty5"),
    ("transport", 0.5, "tp_base",   "tp_e5"),
    ("threading", 3.0, "th_base",   "th_aty5"),
    ("coffee",    5.6, "roll_dynbase", "roll_aty5"),
]

norms = {}
for line in open(os.path.join(RAW, "anorms.log"), encoding="utf-8", errors="ignore"):
    m = re.search(r"NORM task=(\S+) B=(\d+) \|a\|mean=([\d.]+) \|a\|median=([\d.]+) std\(a\)=([\d.]+)", line)
    if m:
        norms[m.group(1)] = float(m.group(3))

def load(tag):
    return json.load(open(os.path.join(RAW, "grad_probe_arm_%s.json" % tag)))

def ladder(d):
    return {r["delta_rel"]: r for r in d["ladder"]}

def interp(xs, ys, q):
    pairs = sorted(zip(xs, ys))
    for (a, ya), (b, yb) in zip(pairs, pairs[1:]):
        if ya <= q <= yb or ya >= q >= yb:
            if yb == ya:
                return a
            return a + (q - ya) / (yb - ya) * (b - a)
    return None

def g_at(lad, delta):
    xs = sorted(lad)
    for a, b in zip(xs, xs[1:]):
        if a <= delta <= b:
            ga, gb = lad[a]["rowgrad_live_mean"], lad[b]["rowgrad_live_mean"]
            return ga + (delta - a) / (b - a) * (gb - ga)
    return lad[xs[0]]["rowgrad_live_mean"] if delta <= xs[0] else None

print(f"{'task':>9} {'eta':>5} {'|a|':>6} | {'g@.005':>8} {'R@.005':>7} | {'dstar':>6} "
      f"{'g@dst*':>8} {'R@dst*':>7} || r5: {'g@.005':>8} {'R@.005':>7} {'R@dst*':>7} {'R ratio':>7}")
for task, eta, bt, rt in TASKS:
    a = norms.get(task)
    if a is None:
        print(f"{task:>9}  MISSING |a|"); continue
    db, dr = load(bt), load(rt)
    lb, lr = ladder(db), ladder(dr)
    kl = {k: lb[k]["kl_mean"] for k in lb}
    dstar = interp(sorted(kl), [kl[k] for k in sorted(kl)], 2.5)
    g005b = lb[0.005]["rowgrad_live_mean"]; g005r = lr[0.005]["rowgrad_live_mean"]
    if dstar:
        gds_b = g_at(lb, dstar); gds_r = g_at(lr, dstar)
        Rdb = eta * 0.5 * gds_b / a; Rdr = eta * 0.5 * gds_r / a
    else:
        gds_b = gds_r = Rdb = Rdr = float("nan")
    R005b = eta * 0.5 * g005b / a; R005r = eta * 0.5 * g005r / a
    ds = ("%.3f" % dstar) if dstar else "none"
    print(f"{task:>9} {eta:>5} {a:>6.3f} | {g005b:>8.3f} {R005b:>7.3f} | {ds:>6} "
          f"{gds_b:>8.3f} {Rdb:>7.3f} ||     {g005r:>8.3f} {R005r:>7.3f} {Rdr:>7.3f} "
          f"{(Rdr/Rdb if Rdb==Rdb and Rdb!=0 else float('nan')):>7.2f}")
