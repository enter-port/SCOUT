"""Delete all wandb runs in the two aborted square orbit projects (user order
2026-09-02: chains stopped, clean wandb logs + outputs, do NOT restart).
Prints each run before deleting. Only touches these two projects."""
import wandb

ENTITY = "jiachunbao-sjtu"
PROJECTS = [
    "SQUARE-9-1-orbit-s2333",
    "SQUARE-9-1-orbit-s23333",
]

api = wandb.Api()
for proj in PROJECTS:
    path = f"{ENTITY}/{proj}"
    runs = list(api.runs(path))
    print(f"{path}: {len(runs)} run(s)")
    for r in runs:
        print(f"  deleting id={r.id} name={r.name} state={r.state}")
        r.delete()
print("DONE")
