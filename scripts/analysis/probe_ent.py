import json, glob

for t, s in [("h1", "0.05"), ("h2", "0.2"), ("h3", "0.5")]:
    g = glob.glob(f"data/entropy_e2e/{t}/log/*.json")
    if not g:
        print(t, s, "no json")
        continue
    d = json.load(open(g[0]))
    print("scale=%s: base=%s/12 rescued=%s pass5=%.3f expl_jerk=%.3f eval_jerk=%.3f trajs=%s" % (
        s, d["baseline_solved"], d["exploration_rescued"], d["pass_at_5"],
        d.get("avg_jerk") or 0, d.get("jerk_baseline") or 0, d["n_all_trajs"]))
