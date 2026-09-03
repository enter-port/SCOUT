# Diagnose the fingerprint mismatch: (a) within-file clustering exactness,
# (b) v1-vs-v3 cross-file cluster match, (c) dim-wise diff vs regen states,
# (d) what the writer actually stores in 'states'.
import os, collections
import numpy as np
import h5py

V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"
STATE = np.load("/tmp/init_states_100.npy")

def s0_list(path):
    out = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for k in names:
            out.append(f["data"][k]["states"][0].astype(np.float64))
    return out

def cluster(states, tol=1e-10):
    reps, counts = [], []
    for s in states:
        for i, r in enumerate(reps):
            if np.abs(r - s).max() <= tol:
                counts[i] += 1; break
        else:
            reps.append(s); counts.append(1)
    return reps, counts

for tag, p in [("v1r1all", V1 + "/rollout/SCOUT-exp1/all.hdf5"),
               ("v3r1all", V3 + "/rollout/SCOUT-exp1/all.hdf5")]:
    ss = s0_list(p)
    reps, cnt = cluster(ss)
    print(f"{tag}: {len(ss)} demos -> {len(reps)} clusters (tol 1e-10); "
          f"counts sorted desc: {sorted(cnt, reverse=True)[:12]}")

r1, c1 = cluster(s0_list(V1 + "/rollout/SCOUT-exp1/all.hdf5"))
r3, c3 = cluster(s0_list(V3 + "/rollout/SCOUT-exp1/all.hdf5"))
print(f"cross-file: v1 {len(r1)} clusters vs v3 {len(r3)} clusters")
dm = []
for i, a in enumerate(r3):
    d = np.array([np.abs(b - a).max() for b in r1])
    dm.append(float(d.min()))
print(f"  v3-cluster -> nearest v1-cluster maxdist: min={min(dm):.2e} med={np.median(dm):.2e} max={max(dm):.2e}")

# dim-wise: take one v1 cluster, its nearest regen state, diff per dim
a = r1[0]
d = np.abs(STATE - a[None, :]).max(axis=1)
j = int(d.argmin())
diff = np.abs(STATE[j] - a)
print(f"regen match attempt: scene {j} dist={d[j]:.3e}; top diff dims: "
      f"{np.argsort(diff)[::-1][:8]} vals={diff[np.argsort(diff)[::-1][:8]].round(5)}")
print(f"  STATE[{j}][:8]={STATE[j][:8].round(4)}  demo[:8]={a[:8].round(4)}")
print(f"  regen inter-scene dist sample: {np.abs(STATE[1]-STATE[2]).max():.3f}")

# check v1-only-failed scene count vs clusters
print(f"expected clusters: v1 62 failed scenes -> {len(r1)}; v3 63 -> {len(r3)}")
