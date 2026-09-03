"""Delete the aborted s233 round-5 wandb run (user order 2026-09-02: round-5
explore tainted by memory explosion, clean + redo). Only touches run
1rzjjv69 / SCOUT-s233-round5 in SQUARE-9-1-orbit-s233."""
import wandb

api = wandb.Api()
path = "jiachunbao-sjtu/SQUARE-9-1-orbit-s233"
r = api.run(f"{path}/1rzjjv69")
print(f"deleting id={r.id} name={r.name} state={r.state}")
r.delete()
print("DONE")
