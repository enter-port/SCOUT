# Read-only: compare per-round eval SR / rescue between orbit v1 (no phase-2
# restriction), orbit v3 (fb soft-clamp + sigma decay + eta-dimless), and the
# atypical entropy chain (cool-dose reference) on SQUARE s233.
import json, glob, os

def scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}

ROOTS = [
 ("v1-orbit", "/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square/rollout"),
 ("v3-orbit", "/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square/rollout"),
 ("aty-S",   "/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square/rollout"),
]
for tag, root in ROOTS:
    for sub in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(sub):
            continue
        name = os.path.basename(sub)
        for j in sorted(glob.glob(os.path.join(sub, "log", "*.json"))):
            try:
                with open(j) as f:
                    d = json.load(f)
            except Exception as e:
                print(f"== [{tag}] {name}: LOAD ERR {e}")
                continue
            print(f"== [{tag}] {name}  ({os.path.basename(j)})")
            print("   top:", json.dumps(scalars(d), ensure_ascii=False))
            for k, v in d.items():
                if isinstance(v, list) and len(v) and isinstance(v[0], (int, float)):
                    print(f"   top.{k} [{len(v)}]: {v if len(v) <= 120 else str(v[:120]) + '...'}")
            ex = d.get("explore")
            if isinstance(ex, dict):
                print("   exp:", json.dumps(scalars(ex), ensure_ascii=False))
                for k, v in ex.items():
                    if isinstance(v, list):
                        print(f"   exp.{k} [{len(v)}]: {v if len(v) <= 120 else str(v[:120]) + '...'}")
    print()
