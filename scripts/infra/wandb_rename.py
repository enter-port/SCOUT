import wandb, sys
api = wandb.Api()
ENTITY = "jiachunbao-sjtu"
PROJ = "scout-base-dp"
DRY = "--apply" not in sys.argv

mapping = {
    "SCOUT-baseDP-lift": "DP-lift-base",
    "SCOUT-baseDP-can": "DP-can-base",
    "SCOUT-baseDP-square": "DP-square-base",
    "SCOUT-baseDP-transport": "DP-transport-base",
    "SCOUT-baseDP-can-exp1": "DP-can-SCOUT-1",
    "SCOUT-baseDP-square-exp1": "DP-square-SCOUT-1",
}

print(f"=== run renames (DRY={'yes' if DRY else 'no'}) ===")
for r in api.runs(f"{ENTITY}/{PROJ}"):
    if r.name in mapping:
        new = mapping[r.name]
        print(f"  {r.name!r} -> {new!r}  (id={r.id}, {r.state})")
        if not DRY:
            r.name = new
            r.update()
    else:
        print(f"  SKIP {r.name!r} (id={r.id})")

print("\n=== project rename probe (scout-base-dp -> dp) ===")
proj = api.project(PROJ, entity=ENTITY)
meths = [m for m in dir(proj) if not m.startswith("_")]
print("  Project attrs:", meths)
rename_like = [m for m in meths if m in ("update","rename","save","edit")]
print("  rename-like methods:", rename_like or "NONE -> 项目改名需在 UI 做")
