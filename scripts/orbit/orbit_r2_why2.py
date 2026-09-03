# orbit_r2_why2.py -- read-only forensic v2 (fingerprint-space join, no regen).
# Scene identity = exact initial-state bytes; all files compared in that space.
import os, json, collections, hashlib
import numpy as np
import h5py

V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"

def jload(p):
    with open(p) as f: return json.load(f)

def fp(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]

def load(path):
    """-> {fp: [demo,...]} plus flat list (core excluded later)."""
    out = collections.defaultdict(list)
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for k in names:
            g = f["data"][k]
            a = g["actions"][:].astype(np.float64)
            d = dict(fpr=fp(g["states"][0]), acts=a, steps=int(a.shape[0]))
            out[d["fpr"]].append(d)
    return out

def jerk(a):
    if a.shape[0] < 4: return 0.0
    d3 = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    return float(np.linalg.norm(d3, axis=-1).mean())

core = load(V3 + "/rollout/square_core.hdf5")
CORE_FP = set(core)
print(f"core fps: {len(CORE_FP)}")

r1a_v1 = load(V1 + "/rollout/SCOUT-exp1/all.hdf5");      r1s_v1 = load(V1 + "/rollout/SCOUT-exp1/success.hdf5")
r2a_v1 = load(V1 + "/rollout/SCOUT-exp2/all.hdf5")
r1a_v3 = load(V3 + "/rollout/SCOUT-exp1/all.hdf5");      r1s_v3 = load(V3 + "/rollout/SCOUT-exp1/success.hdf5")
r2a_v3 = load(V3 + "/rollout/SCOUT-exp2/all.hdf5")
def strip(d):
    return {k: v for k, v in d.items() if k not in CORE_FP}
A1, S1 = strip(r1a_v1), strip(r1s_v1)
A2v1 = strip(r2a_v1)
A3, S3 = strip(r1a_v3), strip(r1s_v3)
A2v3 = strip(r2a_v3)
print(f"scene clusters (core-stripped): v1r1={len(A1)} v1r2={len(A2v1)} v3r1={len(A3)} v3r2={len(A2v3)}")
print(f"  (expect v1r1=62 v1r2=38 v3r1=63 v3r2=47 failed scenes)")

F2v1 = set(jload(V1 + "/rollout/SCOUT-exp2/log/square_SCOUT_rollout_exp2.json")["failed_init_indices"])
F2v3 = set(jload(V3 + "/rollout/SCOUT-exp2/log/square_SCOUT_explore_exp2.json")["failed_init_indices"])
print(f"r2 failed counts: v1={len(F2v1)} v3={len(F2v3)}; cluster counts match: {len(A2v1)==len(F2v1)}, {len(A2v3)==len(F2v3)}")

# ---- rescue flags per scene fingerprint ----
def flags(A, S):
    return {k: dict(rescued=k in S, nS=len(S.get(k, [])), nT=len(A[k]),
                    jk=float(np.mean([jerk(d["acts"]) for d in S[k]])) if k in S else float("nan"),
                    st=float(np.mean([d["steps"] for d in S[k]])) if k in S else 0.0)
            for k in A}
V1F, V3F = flags(A1, S1), flags(A3, S3)

shared = sorted(set(A1) & set(A3))
only_v1 = set(A1) - set(A3); only_v3 = set(A3) - set(A1)
print(f"\n=== r1 failed scenes: shared={len(shared)} v1-only={len(only_v1)} v3-only={len(only_v3)}")

def cat(k):
    a, b = k in S1, k in S3
    return "both" if a and b else "hot-only" if a else "cool-only" if b else "neither"

cats = collections.Counter(cat(k) for k in shared)
print(f"r1 rescue categories (shared): {dict(cats)}")
print(f"  v1-only-failed scenes rescued by v1: {sum(1 for k in only_v1 if k in S1)}/{len(only_v1)}")
print(f"  v3-only-failed scenes rescued by v3: {sum(1 for k in only_v3 if k in S3)}/{len(only_v3)}")

# r2 outcome in fingerprint space
lost_v3 = [k for k in A2v3 if k not in A2v1]   # failed@v3r2, solved@v1r2
lost_v1 = [k for k in A2v1 if k not in A2v3]   # failed@v1r2, solved@v3r2
newfail_v3 = [k for k in lost_v3 if k not in A1 and k not in A3]  # solved at r1 eval
newfail_v1 = [k for k in lost_v1 if k not in A1 and k not in A3]
print(f"\n=== r2 outcome: scenes failed@v3r2&solved@v1r2 = {len(lost_v3)}; reverse = {len(lost_v1)}")
print(f"  of v3-losses: were r1-failed = {len([k for k in lost_v3 if k in A1 or k in A3])}, r1-solved (regression) = {len(newfail_v3)}")
print(f"  of v1-losses: r1-failed = {len([k for k in lost_v1 if k in A1 or k in A3])}, r1-solved (regression) = {len(newfail_v1)}")

def catdist(keys):
    return dict(collections.Counter(cat(k) if k in set(A1)|set(A3) else "r1solved" for k in keys))
print(f"  v3-loss scenes x r1 category: {catdist(lost_v3)}")
print(f"  v1-loss scenes x r1 category: {catdist(lost_v1)}")

print("\n=== P(fail at r2 | r1 category) [shared scenes] ===")
for c in ["both", "hot-only", "cool-only", "neither"]:
    ks = [k for k in shared if cat(k) == c]
    if not ks: continue
    f1 = sum(1 for k in ks if k in A2v1); f3 = sum(1 for k in ks if k in A2v3)
    print(f"  {c:9} n={len(ks):>2}: fail@v1r2={f1:>2} ({f1/len(ks):.0%})  fail@v3r2={f3:>2} ({f3/len(ks):.0%})")

print("\n=== per-scene detail (shared failed scenes) ===")
print(f"  {'cat':>9} {'nS1':>3} {'nT1':>3} {'jk1':>5} {'st1':>4} {'nS3':>3} {'nT3':>3} {'jk3':>5} {'st3':>4}  r2v1 r2v3")
for k in shared:
    a, b = V1F[k], V3F[k]
    r2v1 = "F" if k in A2v1 else "."
    r2v3 = "F" if k in A2v3 else "."
    print(f"  {cat(k):>9} {a['nS']:>3} {a['nT']:>3} {a['jk']:>5.2f} {a['st']:>4.0f} "
          f"{b['nS']:>3} {b['nT']:>3} {b['jk']:>5.2f} {b['st']:>4.0f}  {r2v1:>4} {r2v3:>4}")

print("\n=== aggregate success-demo quality (rescue demos only, core stripped) ===")
for lbl, S in [("v1 hot", S1), ("v3 cool", S3)]:
    ds = [d for v in S.values() for d in v]
    jk = [jerk(d["acts"]) for d in ds]; st = [d["steps"] for d in ds]
    print(f"  {lbl}: n={len(ds)} jerk mean={np.mean(jk):.3f} p90={np.percentile(jk,90):.3f} "
          f"steps mean={np.mean(st):.0f} med={np.median(st):.0f}")
for lbl, S in [("v1", S1), ("v3", S3)]:
    h = collections.Counter(len(v) for v in S.values())
    print(f"  {lbl} successes-per-rescued-scene: {dict(sorted(h.items()))}")

print("\n=== within-scene diversity (both-rescued, >=2 succ each) ===")
def pair_div(demos, T=64):
    rs = []
    for d in demos:
        a = d["acts"]; t = np.linspace(0, a.shape[0] - 1, T)
        rs.append(np.stack([np.interp(t, np.arange(a.shape[0]), a[:, k]) for k in range(a.shape[1])], axis=1))
    return float(np.mean([np.linalg.norm(rs[i] - rs[j], axis=-1).mean()
                          for i in range(len(rs)) for j in range(i + 1, len(rs))]))
d1, d3 = [], []
for k in shared:
    if len(S1.get(k, [])) >= 2 and len(S3.get(k, [])) >= 2:
        d1.append(pair_div(S1[k])); d3.append(pair_div(S3[k]))
if d1:
    print(f"  n={len(d1)} scenes; mean pairwise action-L2/step v1={np.mean(d1):.4f} v3={np.mean(d3):.4f}")
