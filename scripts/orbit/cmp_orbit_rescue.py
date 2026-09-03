# Read-only recon: entropy-chain json location, hdf5 schemas, telemetry lines.
import os, glob, h5py

def tree(root, depth=3):
    for dp, dns, fns in os.walk(root):
        if dp.count(os.sep) - root.count(os.sep) >= depth:
            dns[:] = []
            continue
        rel = os.path.relpath(dp, root)
        print("  " * (rel.count(os.sep) + 1) + os.path.basename(dp) + "/")
        for f in sorted(fns)[:14]:
            print("  " * (rel.count(os.sep) + 2) + f)
        if len(fns) > 14:
            print("  " * (rel.count(os.sep) + 2) + f"... +{len(fns)-14} files")

EN = "/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square"
print("===== entropy chain layout (rollout dir)"); tree(os.path.join(EN, "rollout"), 2)

V1 = "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square"
V3 = "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square"
for tag, T in [("v1", V1), ("v3", V3)]:
    print(f"===== {tag} SCOUT-exp1 dir"); tree(os.path.join(T, "rollout", "SCOUT-exp1"), 2)
    print(f"===== {tag} DP-SCOUT-exp1 retrain dir"); tree(os.path.join(T, "train", "DP", "DP-SCOUT-exp1"), 2)

def peek_h5(path, label):
    print(f"----- {label}: {path}")
    if not os.path.exists(path):
        print("   MISSING"); return
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.replace("demo", "")))
        print(f"   ndemos={len(demos)} attrs={dict(f.attrs)}"[:400])
        d0 = f["data"][demos[0]]
        print(f"   demo0 keys={list(d0.keys())} attrs={dict(d0.attrs)}"[:400])
        for k in d0.keys():
            try:
                print(f"     {k}: shape={d0[k].shape} dtype={d0[k].dtype}")
            except Exception as e:
                print(f"     {k}: ({e})")
        if "mask" in f:
            print(f"   masks: {list(f['mask'].keys())}")

for tag, T in [("v1", V1), ("v3", V3)]:
    for name in ["success.hdf5", "all.hdf5"]:
        peek_h5(os.path.join(T, "rollout", "SCOUT-exp1", name), f"{tag} r1 {name}")
    peek_h5(os.path.join(T, "train", "DP", "DP-SCOUT-exp1", "success_accum.hdf5"), f"{tag} DP-retrain-data")
