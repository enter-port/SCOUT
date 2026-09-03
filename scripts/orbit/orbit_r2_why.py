# orbit_r2_why.py -- read-only forensic (2026-09-03).
# Q: why is restricted (fb-clamp + sigma-decay + eta-dimless) orbit's square r2
#    eval SR ~10 pts BELOW unrestricted v1, despite MORE rescue demos?
# Method: join rescue demos to scene indices via exact initial-state
#    fingerprints (the 100 eval inits are regenerated with the exact eval
#    seeding path), then cross-tab r1 rescue category vs r2 eval outcome.
import os, sys, json, collections
import numpy as np
import h5py

REP = "/root/workspace/baojiachun/scout-orbit"
V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"
os.chdir(REP); sys.path.insert(0, REP)

def jload(p):
    with open(p) as f: return json.load(f)

v1e1 = jload(V1 + "/rollout/SCOUT-exp1/log/square_SCOUT_rollout_exp1.json")
v1e2 = jload(V1 + "/rollout/SCOUT-exp2/log/square_SCOUT_rollout_exp2.json")
v3e1 = jload(V3 + "/rollout/SCOUT-exp1/log/square_SCOUT_explore_exp1.json")
v3e2 = jload(V3 + "/rollout/SCOUT-exp2/log/square_SCOUT_explore_exp2.json")
F1v1, F2v1 = set(v1e1["failed_init_indices"]), set(v1e2["failed_init_indices"])
F1v3, F2v3 = set(v3e1["failed_init_indices"]), set(v3e2["failed_init_indices"])

# ---------- 1. regenerate the 100 init states (exact eval path) ----------
print("=== [1] regenerating 100 init states (exact eval seeding path)")
import yaml
from scout.eval.rollout import make_robomimic_env_factory
lpb = yaml.safe_load(open("configs/base_dp_square_image.yaml"))
factory = make_robomimic_env_factory(
    dataset_path=V3 + "/rollout/square_core.hdf5",
    shape_meta=lpb["shape_meta"],
    n_obs_steps=int(lpb.get("n_obs_steps", 2)),
    abs_action=bool(lpb.get("abs_action", False)),
)
env = factory()
init_states = []
for i in range(100):
    env.reset(seed=42 + i)
    init_states.append(np.asarray(env.get_state(), dtype=np.float64).copy())
env.close()
STATE = np.stack(init_states)
print(f"    ok: states {STATE.shape}")
np.save("/tmp/init_states_100.npy", STATE)

def scene_of(s0, tol=1e-8):
    d = np.abs(STATE - s0[None, :]).max(axis=1)
    j = int(d.argmin())
    return (j if d[j] <= tol else -1), float(d[j])

# ---------- 2. load demos, fingerprint -> scene ----------
def load_demos(path):
    out = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for k in names:
            g = f["data"][k]
            out.append(dict(
                state=g["states"][0].astype(np.float64),
                acts=g["actions"][:].astype(np.float64),
                steps=int(g["actions"].shape[0]),
                done=int(g["dones"][-1]) if "dones" in g else -1,
                rsum=float(np.sum(g["rewards"][:])) if "rewards" in g else float("nan"),
            ))
    return out

def demo_jerk(a):
    if a.shape[0] < 4: return 0.0
    d3 = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    return float(np.linalg.norm(d3, axis=-1).mean())

def attach_scene(demos, label):
    bad = 0
    for d in demos:
        s, dist = scene_of(d["state"])
        d["scene"], d["fdist"] = s, dist
        if s < 0: bad += 1
    print(f"    {label}: {len(demos)} demos, unmatched={bad}, "
          f"max_fdist={max(d['fdist'] for d in demos):.2e}")
    return demos

print("=== [2] fingerprinting")
v1_succ = attach_scene(load_demos(V1 + "/rollout/SCOUT-exp1/success.hdf5"), "v1 r1 success")
v1_all  = attach_scene(load_demos(V1 + "/rollout/SCOUT-exp1/all.hdf5"), "v1 r1 all")
v3_succ = attach_scene(load_demos(V3 + "/rollout/SCOUT-exp1/success.hdf5"), "v3 r1 success")
v3_all  = attach_scene(load_demos(V3 + "/rollout/SCOUT-exp1/all.hdf5"), "v3 r1 all")

# sanity: dones/rewards semantics on a few demos
for lbl, ds in [("v1succ", v1_succ[:3]), ("v1all", v1_all[:3])]:
    print(f"    sanity {lbl}: " + "; ".join(f"done={d['done']} rsum={d['rsum']:.2f}" for d in ds))

# ---------- 3. per-scene rescue profile ----------
print("=== [3] per-scene r1 rescue profile (shared failed scenes)")
def per_scene(succ, allD):
    m = {}
    for d in succ:
        m.setdefault(d["scene"], []).append(d)
    tries = collections.Counter(d["scene"] for d in allD if d["scene"] >= 0)
    return m, tries
S1, T1 = per_scene(v1_succ, v1_all)
S3, T3 = per_scene(v3_succ, v3_all)

shared = sorted(F1v1 & F1v3)
print(f"    shared r1-failed scenes: {len(shared)}  (v1-only failed {sorted(F1v1-F1v3)}, v3-only {sorted(F1v3-F1v1)})")

def cat(i):
    a, b = i in S1, i in S3
    return "both" if a and b else "hot-only" if a else "cool-only" if b else "neither"

cats = collections.Counter(cat(i) for i in shared)
print(f"    r1 rescue categories (shared scenes): {dict(cats)}")
print(f"    v1-only-failed scenes rescued by v1: {sum(1 for i in F1v1-F1v3 if i in S1)}/{len(F1v1-F1v3)}; "
      f"v3-only-failed rescued by v3: {sum(1 for i in F1v3-F1v1 if i in S3)}/{len(F1v3-F1v1)}")

hdr = f"    {'scene':>5} {'cat':>9} | {'nS1':>3} {'nT1':>3} {'jk1':>5} {'st1':>4} | {'nS3':>3} {'nT3':>3} {'jk3':>5} {'st3':>4} | {'r2v1':>4} {'r2v3':>4}"
print(hdr)
rows = []
for i in shared:
    d1, d3 = S1.get(i, []), S3.get(i, [])
    j1 = float(np.mean([demo_jerk(d["acts"]) for d in d1])) if d1 else float("nan")
    j3 = float(np.mean([demo_jerk(d["acts"]) for d in d3])) if d3 else float("nan")
    s1 = int(np.mean([d["steps"] for d in d1])) if d1 else 0
    s3 = int(np.mean([d["steps"] for d in d3])) if d3 else 0
    r2v1 = "F" if i in F2v1 else "."
    r2v3 = "F" if i in F2v3 else "."
    rows.append((i, cat(i), len(d1), T1.get(i, 0), j1, s1, len(d3), T3.get(i, 0), j3, s3, r2v1, r2v3))
    print(f"    {i:>5} {cat(i):>9} | {len(d1):>3} {T1.get(i,0):>3} {j1:>5.2f} {s1:>4} | {len(d3):>3} {T3.get(i,0):>3} {j3:>5.2f} {s3:>4} | {r2v1:>4} {r2v3:>4}")

# ---------- 4. r2 outcome x r1 category ----------
print("=== [4] r2 eval outcome x r1 rescue category")
lost_by_v3 = sorted((F2v3 - F2v1) & (F1v1 | F1v3))   # failed@v3r2, solved@v1r2, was r1-failed
lost_by_v1 = sorted((F2v1 - F2v3) & (F1v1 | F1v3))
also_new_v3 = sorted((F2v3 - F2v1) - (F1v1 | F1v3))  # newly failed at r2, solved at r1 eval
also_new_v1 = sorted((F2v1 - F2v3) - (F1v1 | F1v3))
def catdist(idxs):
    c = collections.Counter(cat(i) for i in idxs)
    return dict(c)
print(f"    scenes LOST at r2 by v3 only (failed v3r2 & solved v1r2 & r1-failed): {lost_by_v3} -> {catdist(lost_by_v3)}")
print(f"    scenes LOST at r2 by v1 only: {lost_by_v1} -> {catdist(lost_by_v1)}")
print(f"    v3-r2 new failures that PASSED r1 eval (not in rescue pool): {also_new_v3}")
print(f"    v1-r2 new failures that PASSED r1 eval: {also_new_v1}")
# marginal per category: P(fail r2 | cat) for each chain
for c in ["both", "hot-only", "cool-only", "neither"]:
    idxs = [i for i in shared if cat(i) == c]
    if not idxs: continue
    f2v1 = sum(1 for i in idxs if i in F2v1); f2v3 = sum(1 for i in idxs if i in F2v3)
    print(f"    cat={c:9} n={len(idxs):>2}: fail@v1r2 {f2v1:>2} ({f2v1/len(idxs):.0%})  fail@v3r2 {f2v3:>2} ({f2v3/len(idxs):.0%})")

# ---------- 5. aggregate demo quality ----------
print("=== [5] aggregate rescue-demo quality (success demos)")
for lbl, ds in [("v1(hot)", v1_succ), ("v3(cool)", v3_succ)]:
    jk = [demo_jerk(d["acts"]) for d in ds]
    st = [d["steps"] for d in ds]
    print(f"    {lbl}: n={len(ds)} jerk mean={np.mean(jk):.3f} p90={np.percentile(jk,90):.3f} "
          f"steps mean={np.mean(st):.0f} median={np.median(st):.0f}")
# success-per-scene histograms
for lbl, S in [("v1", S1), ("v3", S3)]:
    h = collections.Counter(len(v) for v in S.values())
    print(f"    {lbl} successes-per-rescued-scene hist: {dict(sorted(h.items()))}")

# ---------- 6. within-scene solution diversity ----------
print("=== [6] within-scene diversity of successful retries (scenes both rescued, >=2 succ each)")
def pair_div(demos, T=64):
    if len(demos) < 2: return None
    rs = []
    for d in demos:
        a = d["acts"]; t = np.linspace(0, a.shape[0] - 1, T)
        rs.append(np.stack([np.interp(t, np.arange(a.shape[0]), a[:, k]) for k in range(a.shape[1])], axis=1))
    ds_ = []
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            ds_.append(np.linalg.norm(rs[i] - rs[j], axis=-1).mean())
    return float(np.mean(ds_))
dv1, dv3 = [], []
for i in shared:
    if len(S1.get(i, [])) >= 2 and len(S3.get(i, [])) >= 2:
        a, b = pair_div(S1[i]), pair_div(S3[i])
        if a is not None and b is not None:
            dv1.append(a); dv3.append(b)
if dv1:
    print(f"    scenes with >=2 succ both: n={len(dv1)}; mean pairwise action-dist v1={np.mean(dv1):.4f} v3={np.mean(dv3):.4f}")
else:
    print("    no scenes with >=2 successes in both")

# ---------- 7. retrain configs (equivalence check) ----------
print("=== [7] retrain override diff (v1 vs v3 DP-SCOUT-exp1)")
for tag, T in [("v1", V1), ("v3", V3)]:
    p = os.path.join(T, "train/DP/DP-SCOUT-exp1/.hydra/overrides.yaml")
    print(f"    --- {tag}: " + (open(p).read().replace(chr(10), " | ") if os.path.exists(p) else "MISSING"))
