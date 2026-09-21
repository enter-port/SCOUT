"""Cross-campaign ATY rescue-rate x round table (why coffee only?)."""
import json
import re
from collections import defaultdict

rows = json.load(open(
    "experiments/2026_9_21_ATY_decline_analysis/cross_campaign_explore.jsonl",
    encoding="utf-8"))
E = defaultdict(dict)   # (campaign, chain, arm) -> rnd -> rec
META = {}
for r in rows:
    if "err" in r:
        continue
    m = re.search(r"([^/]+)/([^/]+)/([^/]+)/rollout/([A-Za-z0-9_-]+)-exp(\d+)/", r["file"])
    if not m:
        continue
    camp, chain, task, arm, rnd = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
    if not r.get("n_failed"):
        continue
    E[(camp, chain, arm)][rnd] = r
    META[(camp, chain, arm)] = dict(task=task, eta=r.get("eta"), kap=r.get("kap"),
                                    tries=r.get("tries"), orbit=r.get("orbit"))

for (camp, chain, arm) in sorted(E):
    recs = E[(camp, chain, arm)]
    meta = META[(camp, chain, arm)]
    if not re.match(r"ATY|SCOUT(?!-orbit)", arm) and arm not in ("ATY",):
        # print guided entropy arms only (ATY/SCOUT), skip DP/ORBIT for brevity
        if arm not in ("ATY", "SCOUT"):
            continue
    rngs = sorted(recs)
    line = f"{camp:24s} {chain:22s} {arm:6s} eta={str(meta['eta']):>5s} kap={str(meta['kap']):>4s} K={meta['tries']}: "
    for n in rngs:
        r = recs[n]
        rr = r["rescued"] / r["n_failed"] if r["n_failed"] else float("nan")
        line += f"r{n}:{r['rescued']}/{r['n_failed']}({rr:.0%}) "
    vib1 = recs[rngs[1]]["vib"].split("/")[0] if len(rngs) > 1 else "?"
    viblast = recs[rngs[-1]]["vib"].split("/")[0]
    line += f"| vib r2={vib1} r{rngs[-1]}={viblast}"
    print(line)
