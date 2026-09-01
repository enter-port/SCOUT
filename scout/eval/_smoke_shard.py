"""Hermetic smoke for scene-sharded rescue outputs (orbit-dev, 2026-09-01).

No robomimic / LPB stack needed -- exercises the PURE merge/slice logic:

  1. merge_metrics on synthetic shard summaries (slot-parity split of a
     10-scene set): global SR / pass@try / rescued / weighted jerks /
     failed-index union / explore_detail concatenation, all against
     hand-computed expectations;
  2. the n_slice fallback derivation from (scene_slice, n_eval_global) alone;
  3. the hdf5 union merge (core copy + per-shard appended demos renumbered,
     scout_aug mask rebuilt) on tiny synthetic files;
  4. error paths: duplicate slot and mixed SHARDS must refuse.

The rollout-side slicing itself (index math inside _run_rescue) shares the
``range(slot, n_eval, shards)`` selection asserted here and needs a real env
(server validation).
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np


def _shard_json(slot, shards, bs, n_slice, rescued, jerk_b, avg_jerk,
                failed, detail, n_eval_global=10):
    return {
        "task": "square", "mode": "orbit", "exp_num": 1, "seed": 42,
        "n_init_states": n_eval_global, "try_times": 10,
        "protocol": "rescue", "explore_try_times": 10,
        "success_rate": bs / n_slice, "baseline_solved": bs,
        "n_failed": n_slice - bs,
        "exploration_rescued": rescued, "explore_solved": rescued,
        "explore_total": n_slice - bs,
        "pass_at_5": (bs + rescued) / n_slice,
        "avg_jerk": avg_jerk, "jerk_baseline": jerk_b,
        "failed_init_indices": failed, "explore_detail": detail,
        "collected_trajs": rescued * 2, "n_all_trajs": (n_slice - bs) * 10,
        "scene_slice": [slot, shards],
        "n_eval_global": n_eval_global,
        "n_slice": n_slice,
    }


def main():
    # ---- 0) structural guard (review P0 class: a misplaced module-level
    # helper once made _run_rescue parse as dead nested code -- py_compile
    # passes that! assert the class really owns its methods). ------------- #
    import ast
    import scout.eval.rollout_pipeline as _rp
    _tree = ast.parse(open(_rp.__file__, encoding="utf-8").read())
    _cls = next(n for n in ast.walk(_tree)
                if isinstance(n, ast.ClassDef) and n.name == "RolloutPipeline")
    _methods = {n.name for n in _cls.body if isinstance(n, ast.FunctionDef)}
    assert {"run", "_run_split", "_run_rescue"} <= _methods, _methods
    print(f"[smoke_shard] RolloutPipeline methods intact: "
          f"{sorted(_methods)}: OK")

    from scout.eval.merge_sharded import merge_metrics

    # ---- 1) two shards over a 10-scene set (slots 0/1 parity) ------------- #
    # slot0 scenes 0,2,4,6,8 -- 1 failed (4); slot1 scenes 1,3,5,7,9 -- 2
    # failed (3). rescued 1 and 2. jerk_baseline weighted by bs, avg_jerk
    # weighted by n_failed*try_times.
    s0 = _shard_json(0, 2, bs=4, n_slice=5, rescued=1, jerk_b=0.20,
                     avg_jerk=0.30, failed=[4],
                     detail=[{"init": 4, "solved": True,
                              "first_success_try": 3}])
    s1 = _shard_json(1, 2, bs=3, n_slice=5, rescued=2, jerk_b=0.30,
                     avg_jerk=0.50, failed=[1, 7],
                     detail=[{"init": 1, "solved": False,
                              "first_success_try": 10},
                             {"init": 7, "solved": True,
                              "first_success_try": 7}])
    m = merge_metrics([s0, s1])
    assert m["n_init_states"] == 10, m["n_init_states"]
    assert m["baseline_solved"] == 7 and m["success_rate"] == 0.7
    assert m["exploration_rescued"] == 3
    assert abs(m["pass_at_5"] - 1.0) < 1e-12, m["pass_at_5"]
    assert m["failed_init_indices"] == [1, 4, 7]
    assert [e["init"] for e in m["explore_detail"]] == [1, 4, 7]
    assert m["collected_trajs"] == 6 and m["n_all_trajs"] == 30
    assert abs(m["jerk_baseline"] - (0.20 * 4 + 0.30 * 3) / 7) < 1e-12
    assert abs(m["avg_jerk"] - (0.30 * 10 + 0.50 * 20) / 30) < 1e-12
    assert m["scene_slice"] is None and m["shards_merged"] == 2
    print("[smoke_shard] merge_metrics: counts/rates/unions/weighted "
          "jerks/detail concat all exact: OK")

    # ---- 2) n_slice fallback from (scene_slice, n_eval_global) ------------ #
    s0b = dict(s0)
    del s0b["n_slice"]
    s1b = dict(s1)
    del s1b["n_slice"]
    m2 = merge_metrics([s0b, s1b])
    assert m2["n_init_states"] == 10, "fallback slot-count derivation broken"
    print("[smoke_shard] n_slice fallback (slot-count from range math): OK")

    # ---- 2b) exact explore_jerk_n weighting ------------------------------- #
    # shards carrying explore_jerk_n merge by sum/n exactly (T<4 skips make
    # n_failed*tt only approximate); avg_jerk_exact records which path ran.
    s0c = dict(s0)
    s0c.update({"avg_jerk": 0.30, "explore_jerk_n": 8})   # 2 of 10 trajs skipped
    s1c = dict(s1)
    s1c.update({"avg_jerk": 0.50, "explore_jerk_n": 15})
    m3 = merge_metrics([s0c, s1c])
    assert m3["avg_jerk_exact"] is True
    assert abs(m3["avg_jerk"] - (0.30 * 8 + 0.50 * 15) / 23) < 1e-12, \
        m3["avg_jerk"]
    # mixed presence falls back to the approximate weights
    m4 = merge_metrics([s0c, s1])
    assert m4["avg_jerk_exact"] is False
    print("[smoke_shard] explore_jerk_n exact weighting + mixed fallback: OK")

    # ---- 3) hdf5 union merge on synthetic core+shard files ---------------- #
    import h5py
    from scout.eval.hdf5_writer import merge_accumulated_hdf5
    tmp = tempfile.mkdtemp(prefix="smoke_shard_")
    core = os.path.join(tmp, "core.hdf5")
    with h5py.File(core, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
        g.create_dataset("abs_actions", data=np.zeros((4, 7), dtype=np.float32))
    shard_paths = []
    for k in range(2):
        p = os.path.join(tmp, f"shard{k}.hdf5")
        with h5py.File(core, "r") as src, h5py.File(p, "w") as dst:
            src.copy("data/demo_0", dst["data"] if "data" in dst else
                     dst.create_group("data"))
            g = dst["data"].create_group(f"demo_{k + 1}")
            g.create_dataset("actions",
                             data=np.ones((3, 7), dtype=np.float32) * (k + 1))
            g.create_dataset("abs_actions",
                             data=np.ones((3, 7), dtype=np.float32) * (k + 1))
        shard_paths.append(p)
    out = os.path.join(tmp, "merged.hdf5")
    stats = merge_accumulated_hdf5(core, shard_paths, out)
    assert stats["demos_copied"] == 2 and stats["total_demos"] == 3, stats
    with h5py.File(out, "r") as f:
        names = sorted(f["data"].keys())
        assert names == ["demo_0", "demo_1", "demo_2"], names
        # accum-file mask semantics: the merged file IS the training set
        # (core + every appended demo), so scout_aug marks everything True --
        # same convention as merge_accumulated_hdf5's round-accumulation use.
        mask = f["mask/scout_aug/mask"][()]
        assert mask.tolist() == [True, True, True], mask.tolist()
        # renumbered copies preserve payload: shard0's demo -> demo_1 (ones)
        assert float(np.abs(f["data/demo_1/actions"][()] - 1.0).max()) == 0
        assert float(np.abs(f["data/demo_2/actions"][()] - 2.0).max()) == 0
    print("[smoke_shard] hdf5 union: core copy + renumbered shard demos + "
          "rebuilt scout_aug mask: OK")

    # ---- 4) error paths ---------------------------------------------------- #
    try:
        merge_metrics([s0, s0])
        raise AssertionError("duplicate slot must refuse")
    except ValueError:
        pass
    s_bad = dict(s1)
    s_bad["scene_slice"] = [1, 3]
    try:
        merge_metrics([s0, s_bad])
        raise AssertionError("mixed SHARDS must refuse")
    except ValueError:
        pass
    try:
        merge_metrics([s0])
        raise AssertionError("missing slot must refuse")
    except ValueError:
        pass
    print("[smoke_shard] guards: duplicate slot / mixed SHARDS / missing "
          "slot all refuse: OK")
    print("ALL SMOKE SHARD PASSED")


if __name__ == "__main__":
    main()
