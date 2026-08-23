import json, glob, re, os

seeds = ["233", "2333", "23333", "233333"]
R = "/root/workspace/baojiachun/scout/data/2026_8_21"

print("== guidance strength |dNLL/da| per dyn retrain (train.log guidance-check) ==")
for S in seeds:
    grads = []
    for e in range(1, 4):
        g = glob.glob(f"{R}/CAN-exp1-{S}/can/train/dyn/dyn-SCOUT-exp{e}/train.log")
        if not g:
            grads.append("."); continue
        txt = open(g[0], errors="ignore").read()
        m = re.findall(r"guidance-check.*?=\s*([0-9.e+]+)", txt)
        grads.append(m[-1] if m else "?")
    print(f"s{S}: dyn-exp1/2/3 grad = {grads}")

print("== eval-phase jerk (jerk_baseline: retrained policy's own successful eval trajs) ==")
for S in seeds:
    for A in ["DP", "SCOUT"]:
        jb = []
        for N in range(1, 7):
            g = glob.glob(f"{R}/CAN-exp1-{S}/can/rollout/{A}-exp{N}/log/*.json")
            if not g:
                jb.append("  .  "); continue
            d = json.load(open(g[0]))
            j = d.get("jerk_baseline")
            jb.append("%.3f" % j if j is not None else "  .  ")
        print(f"s{S:>6} {A:6}: {' '.join(jb)}")

print("== success_accum demo counts (cumulative training set size per round) ==")
for S in seeds:
    for A in ["DP", "SCOUT"]:
        ns = []
        for N in range(1, 7):
            p = f"{R}/CAN-exp1-{S}/can/rollout/{A}-exp{N}/success_accum.hdf5"
            if not os.path.exists(p):
                ns.append("  ."); continue
            import h5py
            with h5py.File(p, "r") as f:
                n = sum(1 for k in f["data"].keys() if str(k).startswith("demo"))
            ns.append("%4d" % n)
        print(f"s{S:>6} {A:6}: {' '.join(ns)}")
