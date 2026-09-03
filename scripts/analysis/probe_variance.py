import json, glob

seeds = ["233", "2333", "23333", "233333"]
hdr = "seed    arm   | eval per round        | expl solved per round | expl avg_jerk per round"
print(hdr)
for S in seeds:
    D = f"/root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-{S}/can/rollout"
    for A in ["DP", "SCOUT"]:
        ev, ex, jk = [], [], []
        for N in range(1, 7):
            g = glob.glob(f"{D}/{A}-exp{N}/log/*.json")
            if not g:
                ev.append("."); ex.append("."); jk.append("."); continue
            d = json.load(open(g[0]))
            ev.append("%3d" % (d["success_rate"] * 100))
            ex.append("%3s" % d.get("explore_solved", "?"))
            j = d.get("avg_jerk")
            jk.append("%5s" % ("%.3f" % j if j is not None else "."))
        print(f"{S:>6} {A:6}| {' '.join(ev)} | {' '.join(ex)} | {' '.join(jk)}")
