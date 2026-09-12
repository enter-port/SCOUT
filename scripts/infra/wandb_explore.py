import wandb
api = wandb.Api()
try:
    print("viewer:", api.viewer)
except Exception as e:
    print("viewer err:", repr(e)[:80])
# probe candidate entity/project combos
for ent in ["jiachunbao-sjtu", "jiachunbao"]:
    for proj in ["scout-base-dp", "SCOUT-baseDP", "scout-basedp", "scout_dynamics", "scout-dynamics"]:
        try:
            runs = list(api.runs(f"{ent}/{proj}"))
            print(f"\nFOUND {ent}/{proj}: {len(runs)} runs")
            for r in runs:
                print(f"   id={r.id}  name={r.name!r}  state={r.state}")
        except Exception as e:
            print(f"miss {ent}/{proj}: {str(e)[:50]}")
