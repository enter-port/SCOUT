"""Merge scene-sharded rescue rollout outputs (multicore sharding, 2026-09-01).

Companion to ``run_rollout --scene-slice SLOT:SHARDS`` (orbit-dev): P worker
processes each ran the rescue protocol on scenes with original index
``i % P == SLOT`` and wrote ``-shard{SLOT}of{SHARDS}``-suffixed outputs. This
tool merges them back into the monolithic-equivalent result set:

  * json  -- global success_rate / pass@try_times / rescued / jerk aggregates
             (exact weighted means), unioned failed_init_indices, concatenated
             explore_detail (original scene indices preserved by the workers);
  * hdf5  -- success / all files via :func:`merge_accumulated_hdf5`
             (copy core once + append each shard's added demos, renumbered).

Statistical-equivalence note: each worker draws from its own RNG stream, so
the merged result is NOT bit-identical to a monolithic run of the same seed
(same protocol, same scene set, same initial states; retry randomness is
re-drawn per (seed, SHARDS, SLOT) composition). Bit-replay confirmation runs
should stay monolithic.

Usage:
    python -m scout.eval.merge_sharded \
        --jsons shard0.json shard1.json [...] \
        --out-json merged.json \
        [--success-hdf5s s0.hdf5 s1.hdf5 [...] \
         --core-hdf5 core.hdf5 --out-success merged_success.hdf5] \
        [--all-hdf5s a0.hdf5 a1.hdf5 [...] \
         --core-hdf5 core.hdf5 --out-all merged_all.hdf5]

The hdf5 merge reuses the dyn-accumulation merger: every shard file is
``core + that shard's demos`` (write_rollouts_to_hdf5 include_core=True), so
copying the core once + each shard's appended groups reproduces the union.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional


def _shard_n(d: dict) -> int:
    n = d.get("n_slice")
    if isinstance(n, int) and n > 0:
        return n
    sl = d.get("scene_slice") or d.get("metrics", {}).get("scene_slice")
    ng = d.get("n_eval_global") or d.get("metrics", {}).get("n_eval_global")
    if sl and ng:
        slot, shards = int(sl[0]), int(sl[1])
        return len(range(slot, int(ng), shards))
    # unsliced shard (SHARDS=1 or a monolithic file mixed in): use its own count
    return int(d.get("n_init_states", 0))


def _get(d: dict, key, default=None):
    if key in d:
        return d[key]
    return d.get("metrics", {}).get(key, default)


def merge_metrics(shards: List[dict]) -> dict:
    """Aggregate shard SUMMARY jsons (the dicts run_rollout writes to log/) into
    one merged summary with the monolithic-equivalent fields."""
    if not shards:
        raise ValueError("no shard jsons given")
    slots = [_get(d, "scene_slice") for d in shards]
    if any(s is None for s in slots):
        raise ValueError(
            "every shard json must carry scene_slice (was it produced with "
            "--scene-slice?); got slots "
            f"{[None if s is None else s[0] for s in slots]}")
    shards_seen = {int(s[1]) for s in slots}
    if len(shards_seen) != 1:
        raise ValueError(f"mixed SHARDS counts across shards: {shards_seen}")
    n_shards = slots[0][1]
    seen_slots = sorted(int(s[0]) for s in slots)
    if seen_slots != list(range(n_shards)):
        raise ValueError(
            f"expected slots 0..{n_shards - 1}, got {seen_slots} -- every "
            "shard must be present before merging")

    tt = {_get(d, "explore_try_times") for d in shards}
    if len(tt) != 1:
        raise ValueError(f"inconsistent explore_try_times across shards: {tt}")
    tt = int(tt.pop())

    n_tot = sum(_shard_n(d) for d in shards)
    bs_tot = sum(int(_get(d, "baseline_solved", 0)) for d in shards)
    rescued_tot = sum(int(_get(d, "exploration_rescued",
                               _get(d, "explore_solved", 0)) or 0)
                      for d in shards)
    failed_union = sorted({int(i) for d in shards
                           for i in _get(d, "failed_init_indices", [])})
    detail = sorted(
        (e for d in shards for e in _get(d, "explore_detail", []) or []),
        key=lambda e: int(e["init"]))

    # exact weighted means
    def _wmean(vals_counts):
        num = sum(v * c for v, c in vals_counts if v is not None and c > 0)
        den = sum(c for v, c in vals_counts if v is not None and c > 0)
        return (num / den) if den > 0 else None

    # exact jerk weighting when shards carry explore_jerk_n (T<4 skips make
    # n_failed*try_times only approximate); fall back to the approximation
    # for shard jsons from before the field existed.
    if all(_get(d, "explore_jerk_n") is not None for d in shards):
        _jn = [int(_get(d, "explore_jerk_n")) for d in shards]
        _js = [float(_get(d, "avg_jerk")) * n
               for d, n in zip(shards, _jn) if n > 0]
        avg_jerk = (sum(_js) / sum(_jn)) if sum(_jn) > 0 else None
        avg_jerk_exact = True
    else:
        avg_jerk = _wmean(
            [(_get(d, "avg_jerk"),
              int(_get(d, "n_failed", 0)) * tt) for d in shards])
        avg_jerk_exact = False
    jerk_baseline = _wmean(
        [(_get(d, "jerk_baseline"), int(_get(d, "baseline_solved", 0)))
         for d in shards])

    merged = dict(shards[0])          # inherit task/mode/ckpt provenance
    merged.update({
        "n_init_states": n_tot,
        "success_rate": bs_tot / max(n_tot, 1),
        "baseline_solved": bs_tot,
        "n_failed": n_tot - bs_tot,
        "exploration_rescued": rescued_tot,
        "explore_solved": rescued_tot,
        "explore_total": n_tot - bs_tot,
        "pass_at_5": (bs_tot + rescued_tot) / max(n_tot, 1),  # pass@try_times
        "avg_jerk": avg_jerk,
        "jerk_baseline": jerk_baseline,
        "failed_init_indices": failed_union,
        "explore_detail": detail,
        "collected_trajs": sum(int(_get(d, "collected_trajs", 0) or 0)
                               for d in shards),
        "n_all_trajs": sum(int(_get(d, "n_all_trajs", 0) or 0) for d in shards),
        "scene_slice": None,
        "n_eval_global": n_tot,
        "n_slice": n_tot,
        "shards_merged": len(shards),
        "avg_jerk_exact": avg_jerk_exact,
        # per-shard provenance that would be STALE if inherited from
        # shard 0 (review P2): clear or re-derive -- main() re-fills
        # "outputs" with the merged hdf5 paths when they are merged.
        "n_success_trajs": merged["collected_trajs"],
        "explore_jerk_n": (sum(_jn) if avg_jerk_exact else None),
        "wandb_run_id": None,
        "outputs": None,
    })
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsons", nargs="+", required=True,
                    help="the P shard summary jsons (slots 0..P-1)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--success-hdf5s", nargs="+", default=None,
                    help="shard success.hdf5 files (core + shard successes)")
    ap.add_argument("--all-hdf5s", nargs="+", default=None,
                    help="shard all.hdf5 files (core + shard every-trajs)")
    ap.add_argument("--core-hdf5", default=None,
                    help="the core-only hdf5 (needed for any hdf5 merge)")
    ap.add_argument("--out-success", default=None)
    ap.add_argument("--out-all", default=None)
    args = ap.parse_args()

    shards = []
    for p in args.jsons:
        with open(p) as f:
            shards.append(json.load(f))
    merged = merge_metrics(shards)
    with open(args.out_json, "w") as f:
        json.dump(merged, f, indent=2)
    sr, p10 = merged["success_rate"], merged["pass_at_5"]
    print(f"[merge_sharded] {merged['shards_merged']} shards -> {args.out_json}: "
          f"n={merged['n_init_states']} SR={sr:.3f} "
          f"pass@{merged['explore_try_times']}={p10:.3f} "
          f"rescued={merged['exploration_rescued']}/{merged['n_failed']} "
          f"jerk_b={merged['jerk_baseline']} avg_jerk={merged['avg_jerk']}")

    merged_outputs = {}
    for paths, out, kind in ((args.success_hdf5s, args.out_success, "success"),
                             (args.all_hdf5s, args.out_all, "all")):
        if not paths:
            continue
        if not args.core_hdf5 or not out:
            raise SystemExit(f"--{kind}-hdf5s needs --core-hdf5 and "
                             f"--out-{kind}")
        existing = [q for q in paths if os.path.exists(q)]
        for q in paths:
            if not os.path.exists(q):
                print(f"[merge_sharded] NOTE: {q} absent -- that shard wrote "
                      f"no {kind} hdf5 (0 rescued / no failed scenes); "
                      f"skipping it")
        if not existing:
            # mirrors the monolithic run's skip semantics for empty outputs
            print(f"[merge_sharded] no shard {kind} hdf5s exist -- skipping "
                  f"the {kind} merge")
            continue
        from scout.eval.hdf5_writer import merge_accumulated_hdf5
        stats = merge_accumulated_hdf5(args.core_hdf5, existing, out)
        print(f"[merge_sharded] {kind} hdf5 -> {out}: {stats}")
        merged_outputs[kind] = out
    if merged_outputs:
        with open(args.out_json) as f:
            m = json.load(f)
        m["outputs"] = merged_outputs
        with open(args.out_json, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
