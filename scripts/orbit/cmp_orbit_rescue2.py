# Read-only recon v2: hdf5 schemas (demo_N keys), failed.json, telemetry lines.
import os, h5py

V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"

def demo_sort(keys):
    return sorted(keys, key=lambda s: int(s.split("_")[-1]))

def peek_h5(path, label, max_print=2):
    print(f"----- {label}: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")
    if not os.path.exists(path):
        print("   MISSING"); return
    with h5py.File(path, "r") as f:
        demos = demo_sort(f["data"].keys())
        print(f"   ndemos={len(demos)} file_attrs={ {k: (v if not hasattr(v,'shape') else '<arr>') for k,v in f.attrs.items()} }")
        for dname in demos[:max_print] + (demos[-1:] if len(demos) > max_print else []):
            d0 = f["data"][dname]
            attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d0.attrs.items()}
            print(f"   {dname} attrs={attrs}"[:600])
        d0 = f["data"][demos[0]]
        for k in d0.keys():
            try:
                print(f"     {k}: shape={d0[k].shape} dtype={d0[k].dtype}")
            except Exception as e:
                print(f"     {k}: ({e})")
        if "mask" in f:
            print(f"   masks: { {k: len(v[:]) for k, v in f['mask'].items()} }")

for tag, T in [("v1", V1), ("v3", V3)]:
    for name in ["success.hdf5", "all.hdf5"]:
        peek_h5(os.path.join(T, "rollout", "SCOUT-exp1", name), f"{tag} r1")
    peek_h5(os.path.join(T, "train", "DP", "DP-SCOUT-exp1", "success_accum.hdf5"), f"{tag} DP-retrain-input")

for tag, T in [("v1", V1), ("v3", V3)]:
    fj = os.path.join(T, "rollout", "SCOUT-exp1", "failed.json")
    if os.path.exists(fj):
        print(f"{tag} failed.json:", open(fj).read()[:400])

print("===== telemetry (mean_inject etc.) last occurrences")
import re
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
        print(f"--- {tag} {exp}: {len(hits)} telemetry lines")
        for h in hits[:2] + hits[-2:]:
            print("   ", h.strip()[:220])
