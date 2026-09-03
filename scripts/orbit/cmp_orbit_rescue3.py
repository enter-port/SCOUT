# Read-only recon v3: attr KEYS only (camera_info flooded the print), masks fixed,
# success/all demo counts both chains, DP-retrain input counts, telemetry.
import os, h5py

V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"

def demo_sort(keys):
    return sorted(keys, key=lambda s: int(s.split("_")[-1]))

def peek_h5(path, label):
    print(f"----- {label}")
    if not os.path.exists(path):
        print("   MISSING:", path); return
    with h5py.File(path, "r") as f:
        demos = demo_sort(f["data"].keys())
        print(f"   {path}")
        print(f"   ndemos={len(demos)}")
        a0 = f["data"][demos[0]].attrs
        small = {}
        for k, v in a0.items():
            s = str(v)
            small[k] = s if len(s) <= 120 else f"<len{len(s)}>"
        print(f"   demo0 attrs: {small}")
        if "mask" in f:
            info = {}
            for k, v in f["mask"].items():
                info[k] = len(v[:]) if isinstance(v, h5py.Dataset) else "group"
            print(f"   masks: {info}")

for tag, T in [("v1", V1), ("v3", V3)]:
    for exp in ["SCOUT-exp1", "SCOUT-exp2"]:
        for name in ["success.hdf5", "all.hdf5"]:
            peek_h5(os.path.join(T, "rollout", exp, name), f"{tag} {exp} {name}")
    peek_h5(os.path.join(T, "train", "DP", "DP-SCOUT-exp1", "success_accum.hdf5"), f"{tag} DP-retrain-input r2")
    peek_h5(os.path.join(T, "rollout", "square_core.hdf5"), f"{tag} core")

for tag, T in [("v1", V1), ("v3", V3)]:
    fj = os.path.join(T, "rollout", "SCOUT-exp1", "failed.json")
    if os.path.exists(fj):
        print(f"{tag} failed.json:", open(fj).read()[:200])

print("===== telemetry")
for tag, T in [("v1", V1), ("v3", V3)]:
    for exp in ["SCOUT-exp1", "SCOUT-exp2"]:
        p = os.path.join(T, "rollout", exp, "rollout.stdout")
        if not os.path.exists(p):
            print(f"{tag} {exp}: no rollout.stdout"); continue
        hits = []
        with open(p, errors="replace") as f:
            for line in f:
                if "mean_inject" in line:
                    hits.append(line.rstrip())
        print(f"--- {tag} {exp}: {len(hits)} lines")
        for h in (hits[:1] + hits[-1:] if hits else []):
            print("   ", h.strip()[:240])

print("===== per-episode log lines (first/last few, to find scene/try index format)")
for tag, T in [("v1", V1), ("v3", V3)]:
    p = os.path.join(T, "rollout", "SCOUT-exp1", "rollout.stdout")
    if not os.path.exists(p):
        continue
    with open(p, errors="replace") as f:
        lines = f.readlines()
    cand = [l.rstrip() for l in lines if any(w in l for w in ("init", "scene", "episode", "rescued", "try"))][:8]
    print(f"--- {tag} ({len(lines)} lines total)")
    for c in cand:
        print("   ", c[:220])
