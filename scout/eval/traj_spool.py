"""Bounded-memory trajectory spool for the explore phase (user 2026-09-04 OOM fix).

Problem: ``evaluate_exploration_vec`` keeps EVERY finalized rollout trajectory
(per-frame float obs + next_obs ~200MB/traj at horizon 700) in memory until the
whole phase returns, and the CLI then writes success/all hdf5 in one shot. A
rescue round with 89 failed inits x 10 retries accumulates ~200GB per worker --
the OOM risk this module removes.

Design (user spec: "every n=100 rolled-out trajectories, write to hdf5 and
clear memory", multi-worker compatible):

  * the engine's ``on_done`` calls :meth:`TrajSpool.on_traj` with the RAW traj
    reference BEFORE the engine drops it (entries keep lightweight copies), so
    the spool converts each traj immediately to the compact core-storage
    format (uint8 HWC images, 7-dim axis-angle actions, ~60MB/traj) and the
    raw float arrays are freed on return;
  * every ``flush_every`` kept trajectories the buffer is appended to per-
    target STAGING hdf5 files (``<out>.spool``) and the buffer cleared --
    memory is bounded by O(flush_every) converted trajs (+ the <= n_failed
    pending first-retry records of the rescue rule);
  * multi-worker: each shard worker process owns its own suffixed output
    paths (``--scene-slice`` suffix), hence its own staging files -- no cross-
    process sharing, no locks;
  * :meth:`finalize` assembles the FINAL success/all hdf5 from the staging
    files in the EXACT demo order the one-shot ``write_rollouts_to_hdf5``
    produced (init-major, within-init finalize order; zero-horizon "phantom"
    trajs burn a demo id without writing, matching the one-shot ``continue``)
    -- value-identical outputs, so downstream merges/retrains are unchanged.

Selection rules mirror the pipeline's exactly:

  * rescue: successes -> BOTH success.hdf5 and all.hdf5; an all-failed init
    contributes its FIRST retry (try_idx 0) -> all.hdf5 only; every other
    failure is discarded at receipt (never written);
  * split (fresh scenes): every trajectory -> all.hdf5; successes also ->
    success.hdf5.

Smoke: ``python -m scout.eval.traj_spool`` replays out-of-order arrivals and
asserts finalize() == write_rollouts_to_hdf5() on the same trajs.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from scout.eval.hdf5_writer import (
    _acts_to_storage,
    _demo_list,
    _stack_obs,
    get_aa_transformer,
)


class TrajSpool:
    """Incremental trajectory spool with one-shot-identical final assembly.

    Parameters
    ----------
    core_path, success_path, all_path, core_filter_key, aug_mask_key:
        same meaning as :func:`scout.eval.hdf5_writer.write_rollouts_to_hdf5`
        (both outputs are core + appended, i.e. ``include_core=True``).
    rule:
        ``"rescue"`` or ``"split"`` -- the per-traj data-selection rule (see
        module docstring).
    try_times:
        retries per failed init (rescue rule only) -- when an init has
        received this many trajs its pending first-retry record is resolved.
    flush_every:
        buffer size in kept trajectories before a staging flush (user: 100).
    """

    def __init__(self, core_path: str, success_path: str, all_path: str,
                 core_filter_key: str = "train",
                 aug_mask_key: str = "scout_aug", rule: str = "rescue",
                 try_times: int = 10, flush_every: int = 100,
                 verbose: bool = True):
        if rule not in ("rescue", "split"):
            raise ValueError(f"rule must be 'rescue' or 'split' (got {rule!r})")
        if int(flush_every) < 1:
            raise ValueError(f"flush_every must be >= 1 (got {flush_every})")
        if int(try_times) < 1:
            raise ValueError(f"try_times must be >= 1 (got {try_times})")
        self.core_path = core_path
        self.success_path = success_path
        self.all_path = all_path
        self.core_filter_key = core_filter_key
        self.aug_mask_key = aug_mask_key
        self.rule = rule
        self.try_times = int(try_times)
        self.flush_every = int(flush_every)
        self.verbose = verbose
        # staging state (files created lazily on first append; mode "w" also
        # clobbers any leftover staging from a crashed earlier attempt)
        self._stage_s_path = success_path + ".spool"
        self._stage_a_path = all_path + ".spool"
        self._stage_s = None          # h5py.File, "data" group append target
        self._stage_a = None
        self._stage_s_n = 0
        self._stage_a_n = 0
        # conversion state
        self._obs_keys: Optional[List[str]] = None   # core ∩ first-traj keys
        self._rot = None             # lazy RotationTransformer (pytorch3d)
        # buffers / bookkeeping
        self._seq = 0                 # global arrival counter (within-init order)
        self._buf_s: List[dict] = []  # records -> success staging (successes)
        self._buf_a: List[dict] = []  # records -> all staging (kept trajs)
        self._pending: dict = {}      # init_idx -> record (rescue try-0 firsts)
        self._seen: dict = {}         # init_idx -> trajs received
        self._solved = set()          # init_idx with >=1 success (rescue rule)
        self._n_succ = 0              # kept successes (both files)
        self._n_all = 0               # kept trajs (all file)
        self._n_succ_records = 0      # incl. phantoms (file-written check)
        self._n_all_records = 0
        self._flushes = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------ #
    # engine-facing sink
    # ------------------------------------------------------------------ #
    def on_traj(self, traj: dict, init_idx: int, try_idx: int) -> None:
        """Engine ``on_done`` hook: receive one finalized rollout trajectory.

        ``init_idx`` is the ORIGINAL scene index (engine job tag) and
        ``try_idx`` the 0-based retry of that init; both feed the rescue
        selection rule and the canonical final ordering.
        """
        rec = self._convert(traj, init_idx)
        if self.rule == "rescue":
            self._seen[init_idx] = self._seen.get(init_idx, 0) + 1
            if rec["success"]:
                self._solved.add(init_idx)
                self._pending.pop(init_idx, None)
                self._keep(rec, to_success=True)
            else:
                if try_idx == 0 and init_idx not in self._solved:
                    # candidate dyn "first retry": held until the init's full
                    # try_times are known (discarded if any retry succeeds)
                    self._pending[init_idx] = rec
                if self._seen[init_idx] >= self.try_times:
                    self._resolve_init(init_idx)
        else:                          # split: every traj -> all; succ -> both
            self._keep(rec, to_success=rec["success"])
        if len(self._buf_a) + len(self._pending) >= self.flush_every:
            self._flush()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _keep(self, rec: dict, to_success: bool) -> None:
        if to_success:
            self._buf_s.append(rec)
            self._n_succ_records += 1
        self._buf_a.append(rec)
        self._n_all_records += 1
        self._n_succ += int(to_success)   # kept records (== one-shot list len)
        self._n_all += 1

    def _resolve_init(self, init_idx: int) -> None:
        """All try_times retries of an init arrived: resolve its pending first."""
        rec = self._pending.pop(init_idx, None)
        if rec is not None and init_idx not in self._solved:
            self._buf_a.append(rec)
            self._n_all_records += 1
            self._n_all += 1

    def _convert(self, traj: dict, init_idx: int) -> dict:
        """RAW traj -> compact storage-format record (mirrors the one-shot
        writer's per-demo prep exactly; same helpers)."""
        rec = {"seq": self._seq, "init": int(init_idx), "success":
               bool(traj.get("success", True)), "phantom": False}
        self._seq += 1
        ep_len = int(traj.get("horizon", 0))
        rec["ep_len"] = ep_len
        if ep_len == 0:
            # one-shot writer burns a demo id and writes nothing (continue)
            rec["phantom"] = True
            return rec
        obs_list = traj.get("obs") or []
        if len(obs_list) < ep_len:
            raise ValueError(
                f"traj init={init_idx} seq={rec['seq']} missing obs (need "
                f"record_obs=True); got {len(obs_list)} frames, need {ep_len}")
        keys = set(obs_list[0].keys())
        if self._obs_keys is None:
            self._set_obs_keys(keys)
        elif keys != self._key_union:
            raise ValueError(
                f"traj init={init_idx} seq={rec['seq']} obs keys {sorted(keys)} "
                f"!= earlier trajs' {sorted(self._key_union)} (the one-shot "
                f"writer would fail the same inconsistent batch)")
        rec["obs"] = {k: _stack_obs(obs_list, k, ep_len) for k in self._obs_keys}
        next_obs_list = traj.get("next_obs") or []
        rec["next_obs"] = ({k: _stack_obs(next_obs_list, k, ep_len)
                            for k in self._obs_keys} if next_obs_list else None)
        acts = np.asarray(traj["actions"], dtype=np.float32)
        if acts.shape[-1] in (10, 20):
            if self._rot is None:
                self._rot = get_aa_transformer()
            acts = _acts_to_storage(acts, self._rot)
        rec["actions"] = acts
        rec["dones"] = np.asarray(traj["dones"], dtype=bool)
        states = traj.get("states")
        rec["states"] = (np.asarray(states[:ep_len], dtype=np.float32)
                         if states is not None and len(states) >= ep_len
                         else None)
        return rec

    def _set_obs_keys(self, keys: set) -> None:
        """core ∩ rollout keys, decided from the FIRST kept traj. The one-shot
        writer intersects core with the union over ALL rollouts -- identical
        for the consistent key sets the engine always produces; on an
        inconsistent batch the one-shot writer instead fails later at the
        per-rollout stack (KeyError on the missing key) -- same failure class,
        just earlier here."""
        import h5py
        with h5py.File(self.core_path, "r") as f:
            core_demos = _demo_list(f, self.core_filter_key)
            if not core_demos:
                raise RuntimeError(
                    f"no core demos under mask='{self.core_filter_key}' in "
                    f"{self.core_path}")
            core_keys = set(f["data"][core_demos[0]]["obs"].keys())
        self._key_union = keys
        self._obs_keys = sorted(core_keys & keys) if keys else sorted(core_keys)

    def _ensure_stage(self, which: str):
        import h5py
        if which == "s" and self._stage_s is None:
            self._stage_s = h5py.File(self._stage_s_path, "w")
            self._stage_s.create_group("data")
        if which == "a" and self._stage_a is None:
            self._stage_a = h5py.File(self._stage_a_path, "w")
            self._stage_a.create_group("data")

    def _write_rec_group(self, stage, stage_which: str, rec: dict) -> None:
        """Append one record to a staging file as ``data/demo_<k>``."""
        name = f"demo_{self._stage_s_n if stage_which == 's' else self._stage_a_n}"
        grp = stage["data"].create_group(name)
        grp.attrs["spool_seq"] = int(rec["seq"])
        grp.attrs["spool_init"] = int(rec["init"])
        grp.attrs["num_samples"] = int(rec["ep_len"])
        if rec["phantom"]:
            grp.attrs["spool_phantom"] = 1
            return
        obs_grp = grp.create_group("obs")
        for k, arr in rec["obs"].items():
            obs_grp.create_dataset(k, data=arr)
        if rec["next_obs"] is not None:
            next_grp = grp.create_group("next_obs")
            for k, arr in rec["next_obs"].items():
                next_grp.create_dataset(k, data=arr)
        grp.create_dataset("actions", data=rec["actions"])
        # abs_actions: identical 7-dim aa (the one-shot writer stores BOTH; the
        # LPB loader reads THIS key when abs_action=true -- omitting it would
        # KeyError every downstream retrain; review P0-2)
        grp.create_dataset("abs_actions", data=rec["actions"])
        grp.create_dataset("done", data=rec["dones"])
        grp.create_dataset("success",
                           data=np.full(int(rec["ep_len"]), rec["success"],
                                        dtype=bool))
        if rec["states"] is not None:
            grp.create_dataset("states", data=rec["states"])

    def _flush(self) -> None:
        """Append the buffers to the staging files and clear them."""
        if not self._buf_s and not self._buf_a:
            return
        if self._buf_s:
            self._ensure_stage("s")
            for rec in self._buf_s:
                self._write_rec_group(self._stage_s, "s", rec)
                self._stage_s_n += 1
            self._stage_s.flush()
        if self._buf_a:
            self._ensure_stage("a")
            for rec in self._buf_a:
                self._write_rec_group(self._stage_a, "a", rec)
                self._stage_a_n += 1
            self._stage_a.flush()
        n = len(self._buf_a) + len(self._pending)
        self._buf_s.clear()
        self._buf_a.clear()
        self._flushes += 1
        if self.verbose:
            print(f"[traj-spool] flush #{self._flushes}: kept={n} "
                  f"(succ={self._n_succ} all={self._n_all}) "
                  f"elapsed={time.time() - self._t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ #
    # final assembly
    # ------------------------------------------------------------------ #
    def finalize(self) -> dict:
        """Write the FINAL success/all hdf5 (canonical one-shot order) and
        remove the staging files. Mirrors the CLI's old ``if trajs:`` /
        ``else: skip`` file-existence semantics."""
        import h5py
        import shutil
        self._flush()
        # close the staging WRITE handles BEFORE reopening the files for
        # reading below -- a second (read) h5py handle on a file that is still
        # open for writing can observe a stale/partial image (no SWMR); the
        # server smoke caught a half-written demo group this way.
        self.close()
        res = {"success_written": False, "success_demos": 0,
               "all_written": False, "all_demos": 0, "flushes": self._flushes}
        for which, out_path, n_records in (
                ("success", self.success_path, self._n_succ_records),
                ("all", self.all_path, self._n_all_records)):
            if n_records == 0:
                continue
            shutil.copyfile(self.core_path, out_path)
            stage_path = (self._stage_s_path if which == "success"
                          else self._stage_a_path)
            added = 0
            with h5py.File(out_path, "r+") as f, \
                    h5py.File(stage_path, "r") as sf:
                core_set = set(_demo_list(f, self.core_filter_key))
                existing_ids = [int(d.split("_")[-1]) for d in f["data"].keys()
                                if d.startswith("demo_")
                                and d.split("_")[-1].isdigit()]
                next_id = (max(existing_ids) + 1) if existing_ids else 0
                # canonical order: init-major, within-init finalize order
                staged = [(sf["data"][k].attrs["spool_init"],
                           sf["data"][k].attrs["spool_seq"], k)
                          for k in sf["data"].keys()]
                staged.sort()
                new_names = []
                for _init, _seq, src in staged:
                    phantom = int(sf["data"][src].attrs.get("spool_phantom", 0))
                    name = f"demo_{next_id}"
                    next_id += 1
                    if phantom:
                        continue          # burn the id, write nothing
                    sf.copy(f"data/{src}", f["data"], name=name)
                    grp = f["data"][name]
                    for attr in ("spool_seq", "spool_init", "spool_phantom"):
                        if attr in grp.attrs:
                            del grp.attrs[attr]
                    new_names.append(name)
                added = len(new_names)
                all_demos_after = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo")])
                new_set = set(new_names)
                mask = np.array(
                    [d in core_set or d in new_set for d in all_demos_after],
                    dtype=bool)
                if f"mask/{self.aug_mask_key}" in f:
                    del f[f"mask/{self.aug_mask_key}"]
                aug_grp = f.create_group(f"mask/{self.aug_mask_key}")
                aug_grp.create_dataset("mask", data=mask)
                aug_grp.attrs["num"] = int(mask.sum())
                f.attrs["num_demos_added"] = added
            res[f"{which}_written"] = True
            res[f"{which}_demos"] = added
        self.close()
        for p in (self._stage_s_path, self._stage_a_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        if self.verbose:
            print(f"[traj-spool] finalize: success={res['success_demos']} "
                  f"all={res['all_demos']} (kept succ={self._n_succ} "
                  f"all={self._n_all}, flushes={self._flushes}, "
                  f"elapsed={time.time() - self._t0:.0f}s)", flush=True)
        return res

    def close(self) -> None:
        """Close staging handles (finalize calls this; also safe on crash paths)."""
        for attr in ("_stage_s", "_stage_a"):
            f = getattr(self, attr)
            if f is not None:
                try:
                    f.close()
                finally:
                    setattr(self, attr, None)

    @property
    def n_success(self) -> int:
        """Kept successful trajectories (== len of the one-shot trajs list)."""
        return self._n_succ

    @property
    def n_all(self) -> int:
        """Kept all-file trajectories (== len of the one-shot all_trajs list)."""
        return self._n_all


# --------------------------------------------------------------------------- #
# smoke: out-of-order arrivals -> finalize() == write_rollouts_to_hdf5()
# --------------------------------------------------------------------------- #
def _smoke():
    import h5py
    import shutil
    import tempfile

    from scout.eval.hdf5_writer import write_rollouts_to_hdf5

    tmp = tempfile.mkdtemp(prefix="traj_spool_smoke_")
    rng = np.random.default_rng(0)

    # ---- synthetic core (robomimic-ish: HWC uint8 images + low_dim) -------- #
    core = os.path.join(tmp, "core.hdf5")
    with h5py.File(core, "w") as f:
        d = f.create_group("data")
        for i in range(3):
            g = d.create_group(f"demo_{i}")
            g.create_dataset("obs/agentview_image",
                             data=(rng.random((5, 4, 4, 3)) * 255).astype(np.uint8))
            g.create_dataset("obs/robot0_eef_pos",
                             data=rng.random((5, 3)).astype(np.float32))
            g.create_dataset("actions", data=rng.random((5, 7)).astype(np.float32))
            g.attrs["num_samples"] = 5
        m = f.create_group("mask/train")
        m.create_dataset("mask", data=np.array([True, True, True]))

    def mk_traj(T, init_i, try_j, succ):
        """One raw engine-style traj: CHW float obs lists + (T,10) rot6d acts."""
        obs = [{"agentview_image": rng.random((3, 4, 4)).astype(np.float32),
                "robot0_eef_pos": rng.random(3).astype(np.float32)}
               for _ in range(T)]
        nxt = [{"agentview_image": rng.random((3, 4, 4)).astype(np.float32),
                "robot0_eef_pos": rng.random(3).astype(np.float32)}
               for _ in range(T)]
        return {"actions": rng.random((T, 10)).astype(np.float32),
                "rewards": np.zeros(T, dtype=np.float32),
                "dones": np.zeros(T, dtype=bool),
                "states": [rng.random(6).astype(np.float32) for _ in range(T)],
                "obs": obs, "next_obs": nxt, "horizon": T, "success": succ,
                "initial_state_dict": None,
                "_tag": (init_i, try_j)}

    def replay(events, rule, try_times, flush_every, tag):
        """Feed arrival events to a spool; return (spool result, paths)."""
        s_path = os.path.join(tmp, f"succ_{tag}.hdf5")
        a_path = os.path.join(tmp, f"all_{tag}.hdf5")
        sp = TrajSpool(core, s_path, a_path, core_filter_key="train",
                       aug_mask_key="scout_aug", rule=rule, try_times=try_times,
                       flush_every=flush_every, verbose=False)
        try:
            for traj, i, j in events:
                sp.on_traj(traj, i, j)
            return sp.finalize(), s_path, a_path, sp
        finally:
            sp.close()

    def canonical_lists(events, rule):
        """Reconstruct the pipeline's one-shot trajs/all_trajs from the same
        arrival stream (init-major, within-init arrival order)."""
        per_init = {}
        for traj, i, j in events:
            per_init.setdefault(i, []).append(traj)
        trajs, all_trajs = [], []
        for i in sorted(per_init):
            tries = per_init[i]
            if rule == "rescue":
                succ = [t for t in tries if t["success"]]
                trajs.extend(succ)
                if succ:
                    all_trajs.extend(succ)
                else:
                    all_trajs.extend(t for t in tries if t["_tag"][1] == 0)
            else:
                trajs.extend(t for t in tries if t["success"])
                all_trajs.extend(tries)
        return trajs, all_trajs

    def assert_files_equal(p1, p2, ctx):
        with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
            d1 = sorted(f1["data"].keys())
            d2 = sorted(f2["data"].keys())
            assert d1 == d2, f"{ctx}: demo names {d1} != {d2}"
            assert int(f1.attrs["num_demos_added"]) == \
                int(f2.attrs["num_demos_added"]), f"{ctx}: num_demos_added"
            for k in ("mask/scout_aug",):
                np.testing.assert_array_equal(
                    f1[k + "/mask"][()], f2[k + "/mask"][()],
                    err_msg=f"{ctx}: mask {k}")
                assert int(f1[k].attrs["num"]) == int(f2[k].attrs["num"])
            for demo in d1:
                g1, g2 = f1["data"][demo], f2["data"][demo]
                # dataset-name-set parity FIRST (review P1-1: comparing only a
                # fixed tuple is blind to missing keys -- e.g. the abs_actions
                # omission this assert now exists to catch)
                assert set(g1.keys()) == set(g2.keys()), \
                    f"{ctx}:{demo} keys {set(g1.keys())} vs {set(g2.keys())}"
                assert int(g1.attrs["num_samples"]) == \
                    int(g2.attrs["num_samples"]), f"{ctx}:{demo} num_samples"
                assert set(g1.attrs.keys()) == set(g2.attrs.keys()), \
                    f"{ctx}:{demo} attrs {set(g1.attrs)} vs {set(g2.attrs)}"
                for ds in ("actions", "abs_actions", "done", "success"):
                    np.testing.assert_array_equal(
                        g1[ds][()], g2[ds][()], err_msg=f"{ctx}:{demo}/{ds}")
                if "states" in g2:
                    np.testing.assert_array_equal(
                        g1["states"][()], g2["states"][()],
                        err_msg=f"{ctx}:{demo}/states")
                for sub in ("obs", "next_obs"):
                    for k in g2[sub]:
                        np.testing.assert_array_equal(
                            g1[f"{sub}/{k}"][()], g2[f"{sub}/{k}"][()],
                            err_msg=f"{ctx}:{demo}/{sub}/{k}")

    # ---- scenario 1: rescue, out-of-order arrivals, mixed outcomes --------- #
    # every failed init gets EXACTLY try_times=4 trajs (engine invariant);
    # init 10: fail,fail,succ,succ -> 2 succ in both files
    # init 11: 4x fail            -> first (try0) only -> all file
    # init 12: succ,fail,fail,fail -> 1 succ in both files
    ev = []
    e10 = [(mk_traj(6, 10, 0, False), 10, 0), (mk_traj(5, 10, 1, False), 10, 1),
           (mk_traj(7, 10, 2, True), 10, 2), (mk_traj(4, 10, 3, True), 10, 3)]
    e11 = [(mk_traj(5, 11, 0, False), 11, 0), (mk_traj(6, 11, 1, False), 11, 1),
           (mk_traj(4, 11, 2, False), 11, 2), (mk_traj(5, 11, 3, False), 11, 3)]
    e12 = [(mk_traj(4, 12, 0, True), 12, 0), (mk_traj(5, 12, 1, False), 12, 1),
           (mk_traj(6, 12, 2, False), 12, 2), (mk_traj(4, 12, 3, False), 12, 3)]
    for batch in zip(e10, e11, e12):        # INTERLEAVED (parallel slots)
        ev.extend(batch)
    trajs, all_trajs = canonical_lists(ev, "rescue")
    assert len(trajs) == 3 and len(all_trajs) == 4, (len(trajs), len(all_trajs))
    ref_s = os.path.join(tmp, "ref_s1.hdf5")
    ref_a = os.path.join(tmp, "ref_a1.hdf5")
    write_rollouts_to_hdf5(core, ref_s, trajs, core_filter_key="train")
    write_rollouts_to_hdf5(core, ref_a, all_trajs, core_filter_key="train")
    for fe in (1, 2, 3, 100):
        res, s_p, a_p, sp = replay(ev, "rescue", try_times=4, flush_every=fe,
                                   tag=f"r1_fe{fe}")
        assert res["success_demos"] == 3 and res["all_demos"] == 4, res
        assert sp.n_success == 3 and sp.n_all == 4
        assert_files_equal(s_p, ref_s, f"rescue fe={fe} success")
        assert_files_equal(a_p, ref_a, f"rescue fe={fe} all")
        assert not os.path.exists(s_p + ".spool"), "staging unlinked"
    print("[1] rescue out-of-order == one-shot (fe=1,2,3,100) OK")

    # ---- scenario 2: rescue, ZERO successes -> success file skipped -------- #
    ev0 = [(mk_traj(5, 1, 0, False), 1, 0), (mk_traj(6, 1, 1, False), 1, 1)]
    res, s_p, a_p, sp = replay(ev0, "rescue", try_times=2, flush_every=1,
                               tag="r2")
    assert not res["success_written"] and res["all_demos"] == 1, res
    assert sp.n_success == 0 and sp.n_all == 1
    _t, _a = canonical_lists(ev0, "rescue")
    ref_a2 = os.path.join(tmp, "ref_a2.hdf5")
    write_rollouts_to_hdf5(core, ref_a2, _a, core_filter_key="train")
    assert_files_equal(a_p, ref_a2, "rescue zero-succ all")
    print("[2] rescue zero-success skip OK")

    # ---- scenario 3: split rule (every traj -> all; succ -> success) ------- #
    ev3 = [(mk_traj(5, 3, 0, True), 3, 0), (mk_traj(4, 4, 0, False), 4, 0),
           (mk_traj(6, 4, 0, True), 4, 0), (mk_traj(3, 3, 0, False), 3, 0)]
    trajs, all_trajs = canonical_lists(ev3, "split")
    assert len(trajs) == 2 and len(all_trajs) == 4
    ref_s3 = os.path.join(tmp, "ref_s3.hdf5")
    ref_a3 = os.path.join(tmp, "ref_a3.hdf5")
    write_rollouts_to_hdf5(core, ref_s3, trajs, core_filter_key="train")
    write_rollouts_to_hdf5(core, ref_a3, all_trajs, core_filter_key="train")
    res, s_p, a_p, sp = replay(ev3, "split", try_times=1, flush_every=2,
                               tag="r3")
    assert res["success_demos"] == 2 and res["all_demos"] == 4, res
    assert_files_equal(s_p, ref_s3, "split success")
    assert_files_equal(a_p, ref_a3, "split all")
    print("[3] split == one-shot OK")

    # ---- scenario 4: phantom (horizon 0) burns a demo id like the one-shot -- #
    ph = dict(mk_traj(3, 5, 0, True))
    ph["horizon"] = 0
    ev4 = [(ph, 5, 0), (mk_traj(4, 5, 0, True), 5, 0)]
    trajs, all_trajs = canonical_lists(ev4, "split")
    ref_s4 = os.path.join(tmp, "ref_s4.hdf5")
    write_rollouts_to_hdf5(core, ref_s4, trajs, core_filter_key="train")
    res, s_p, a_p, sp = replay(ev4, "split", try_times=1, flush_every=1,
                               tag="r4")
    # one-shot: demo_3 burned, real demo -> demo_4 (names must match exactly)
    with h5py.File(ref_s4, "r") as fr, h5py.File(s_p, "r") as fs:
        assert sorted(fr["data"].keys()) == sorted(fs["data"].keys()), \
            (sorted(fr["data"].keys()), sorted(fs["data"].keys()))
        assert max(int(d.split("_")[-1]) for d in fs["data"].keys()) == 4, \
            "phantom must burn demo id 3 exactly like the one-shot writer"
    assert_files_equal(s_p, ref_s4, "phantom success")
    print("[4] phantom id-burn parity OK")

    # ---- scenario 5: rescue pending released ONLY after try_times seen ----- #
    # try0 fail arrives, then try1 SUCCESS: pending must be discarded, and a
    # late try_idx==0-ish duplicate pattern must not resurrect it.
    ev5 = [(mk_traj(5, 7, 0, False), 7, 0), (mk_traj(4, 7, 1, True), 7, 1),
           (mk_traj(4, 7, 2, False), 7, 2)]
    res, s_p, a_p, sp = replay(ev5, "rescue", try_times=3, flush_every=1,
                               tag="r5")
    assert res["success_demos"] == 1 and res["all_demos"] == 1, res
    assert sp.n_all == 1
    print("[5] pending discard on late success OK")

    shutil.rmtree(tmp)
    print("[smoke] traj_spool.py OK")


if __name__ == "__main__":
    _smoke()
