"""Per-denoise-step force decomposition for the orbit chains (user order
2026-09-02): why does orbit's explore die (square r5, CAN r2) while atypical
survives on the same VIB drift?

Part 1  production telemetry -> per-tick time series (the stdout lines are
        CUMULATIVE means; difference them back into per-tick values).
Part 2  DDPM schedule profile: s_t = sqrt(1-abar_t) per denoise step, the
        per-step climb force eta*s_t*|dKL/da| and tangential noise
        sigma*s_t*E||xi_perp|| (E[chi_{d-1}], d=80 chunk dims).
Part 3  VIB gain probe: for every (round DP, round dyn) pair of both
        chains, measure on core-demo states the atypical-KL level and its
        gradient norm w.r.t. the action chunk, as a function of the
        deviation ||delta a|| from the reference chunk -- the round-over-
        round drift that saturates the kappa shell. Same recipe as
        can_dose_probe.py (frames of core demo_0, obs adapter, E_s tied to
        the round's DP ckpt).

Run:  cd /root/workspace/baojiachun/scout-rand && CUDA_VISIBLE_DEVICES=1 \
      /root/workspace/baojiachun/.venv/bin/python /tmp/orb_step_analysis.py
"""
import math
import re
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-rand")

import numpy as np

SQ = ("/root/workspace/baojiachun/scout-rand/data/"
      "2026_9_1_orbchain/ORBIT-s233/square")
CAN = ("/root/workspace/baojiachun/scout-rand/data/"
       "2026_9_2_orbchain/ORBIT-s233/can")
ATY = "/root/workspace/baojiachun/scout-rand/data/aty_test_s233_r4trio"
ETA, KAPPA, LAM, DELTA, SIGMA = 3.0, 2.5, 0.5, 0.25, 0.25

ORBIT_RE = re.compile(
    r"calls=(\d+) p2_rows=(\d+)/(\d+) mean\|fb\|/p2row=([\d.eE+-]+) "
    r"mean\|noise\|/p2row=([\d.eE+-]+)")
GUID_RE = re.compile(
    r"n=(\d+) mean_inject=([\d.eE+-]+) max_inject=([\d.eE+-]+)")


def decompose(path):
    """stdout -> list of per-tick dicts for orbit/guidance telemetry."""
    orb, gui = [], []
    with open(path, errors="replace") as f:
        for line in f:
            m = ORBIT_RE.search(line)
            if m:
                orb.append(tuple(float(x) for x in m.groups()))
            m = GUID_RE.search(line)
            if m:
                gui.append(tuple(float(x) for x in m.groups()))
    ticks = []
    for i in range(1, len(orb)):
        c0, p0, r0, fb0, nz0 = orb[i - 1]
        c1, p1, r1, fb1, nz1 = orb[i]
        dp2, drows = p1 - p0, r1 - r0
        if drows <= 0:
            continue
        ticks.append(dict(
            calls=int(c1), p2=100.0 * dp2 / drows,
            fb=(fb1 * p1 - fb0 * p0) / dp2 if dp2 > 0 else float("nan"),
            noise=(nz1 * p1 - nz0 * p0) / dp2 if dp2 > 0 else float("nan")))
    gticks = []
    for i in range(1, len(gui)):
        n0, m0, _ = gui[i - 1]
        n1, m1, x1 = gui[i]
        if n1 - n0 <= 0:
            continue
        gticks.append(dict(n=int(n1),
                           inj=(m1 * n1 - m0 * n0) / (n1 - n0),
                           max_inj=x1))
    return ticks, gticks


def fmt_ticks(name, ticks, every=5):
    if not ticks:
        return f"{name}: no orbit telemetry"
    rows = ticks[:3] + ticks[3::every]
    if rows[-1] is not ticks[-1]:
        rows.append(ticks[-1])
    out = [f"  {name}  ({len(ticks)} ticks)"]
    out.append("    {:>9s} {:>7s} {:>7s} {:>7s}".format(
        "calls", "p2%", "|fb|", "|noise|"))
    for t in rows:
        out.append("    {:>9d} {:>7.1f} {:>7.3f} {:>7.3f}".format(
            t["calls"], t["p2"], t["fb"], t["noise"]))
    p2s = np.array([t["p2"] for t in ticks])
    out.append("    per-round tick mean: p2={:.1f}% fb={:.3f} "
               "noise={:.3f}".format(
                   p2s.mean(),
                   np.nanmean([t["fb"] for t in ticks]),
                   np.nanmean([t["noise"] for t in ticks])))
    return "\n".join(out)


def fmt_gticks(name, gticks, every=5):
    if not gticks:
        return f"{name}: no guidance telemetry"
    rows = gticks[:3] + gticks[3::every]
    if rows[-1] is not gticks[-1]:
        rows.append(gticks[-1])
    out = [f"  {name}  ({len(gticks)} ticks)"]
    out.append("    {:>9s} {:>10s} {:>10s}".format("n", "mean_inj", "max(cum)"))
    for t in rows:
        out.append("    {:>9d} {:>10.3f} {:>10.1f}".format(
            t["n"], t["inj"], t["max_inj"]))
    inj = np.array([t["inj"] for t in gticks])
    out.append("    per-round tick mean inj={:.3f}  first3={:.3f}  "
               "last3={:.3f}".format(
                   inj.mean(), inj[:3].mean(), inj[-3:].mean()))
    return "\n".join(out)


def kl_closed(mu1, lv1, mu0, lv0):
    """Diagonal-Gaussian KL(N1||N0), summed over dims -> (B,)."""
    v0 = lv0.exp()
    return (0.5 * ((lv1.exp() + (mu1 - mu0) ** 2) / v0
                   + lv0 - lv1 - 1.0)).sum(dim=-1)


def probe(cfg_path, dp, vibp, core, dev, n_dirs=3):
    """KL and |dKL/da| vs ||delta a|| on core-demo states (one VIB)."""
    import h5py
    import torch
    from omegaconf import OmegaConf
    from scout.eval.factories import make_scout_vib_factory
    from scout.eval.rollout import make_obs_adapter
    from diffusion_policy.dataset.robomimic_replay_image_dataset import \
        _convert_actions
    from diffusion_policy.model.common.rotation_transformer import \
        RotationTransformer

    cfg = OmegaConf.load(cfg_path)
    cfg.vib.base_dp_ckpt = dp
    vib = make_scout_vib_factory(cfg, dev)(vibp).eval()
    views, props = list(cfg.eval.view_names), list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)
    rt = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    with h5py.File(core, "r") as f:
        d = f["data/demo_0"]
        T = d["actions"].shape[0]
        fr = np.arange(0, T - 8, 20)[:12]
        obs = {v: torch.from_numpy(np.ascontiguousarray(
                   d[f"obs/{v}"][:][fr])).permute(0, 3, 1, 2).float()
               .unsqueeze(1) for v in views}
        for k in props:
            obs[k] = torch.from_numpy(np.ascontiguousarray(
                d[f"obs/{k}"][:][fr])).float().unsqueeze(1)
        acts7 = d["abs_actions"][:]
    es = adapter(obs)
    es = {"visual": {v: x.to(dev) for v, x in es["visual"].items()},
          "proprio": es["proprio"].to(dev)}
    with torch.no_grad():
        s_bar = vib.encode(es)
        a0 = np.stack([_convert_actions(acts7[t:t + 8].astype(np.float64),
                                        True, rt).reshape(-1) for t in fr])
        a0 = torch.from_numpy(a0).float().to(dev)
        mu0, lv0 = vib.vib_enc(s_bar, a0)
    d_chunk = a0.shape[1]
    res = dict(scale=float(a0.std()), sigma0=float((0.5 * lv0).exp().mean()))
    for r in (0.05, 0.1, 0.25, 0.5, 1.0):
        kls, gns = [], []
        for s in range(n_dirs):
            g = torch.Generator(device="cpu").manual_seed(42 + s)
            u = torch.randn(d_chunk, generator=g)
            u = (u / u.norm()).to(dev)
            a = (a0 + r * u).clone().requires_grad_(True)
            mu1, lv1 = vib.vib_enc(s_bar, a)
            kl = kl_closed(mu1, lv1, mu0, lv0)
            gr = torch.autograd.grad(kl.sum(), a)[0]
            kls.append(kl.detach())
            gns.append(gr.norm(dim=-1))
        kl = torch.stack(kls)
        gn = torch.stack(gns)
        res[r] = (float(kl.mean()), float(gn.mean()),
                  float((kl >= KAPPA - DELTA).float().mean()),
                  float((kl >= KAPPA).float().mean()))
    return res


def main():
    print("=" * 78)
    print("PART 1  production telemetry, per-tick decomposition")
    print("=" * 78)
    sources = [(f"SQ-r{n}", f"{SQ}/rollout/SCOUT-exp{n}/rollout.stdout")
               for n in range(1, 6)]
    sources += [(f"CAN-r{n}", f"{CAN}/rollout/SCOUT-exp{n}/rollout.stdout")
                for n in (1, 2)]
    sources += [("SQ-aty(r4 trio)", f"{ATY}/rollout.stdout")]
    for name, path in sources:
        try:
            ticks, gticks = decompose(path)
        except FileNotFoundError:
            print(f"  {name}: MISSING")
            continue
        print(fmt_ticks(name, ticks))
        print(fmt_gticks(name, gticks))
        print()

    print("=" * 78)
    print("PART 2  per-denoise-step force profile (eta=3, sigma=0.25, "
          "kappa=2.5)")
    print("=" * 78)
    n_ts, b0, b1 = 100, 1e-4, 0.02        # LPB/DP square DDPM constants
    betas = np.linspace(b0, b1, n_ts)
    s_t = np.sqrt(1.0 - np.cumprod(1.0 - betas))
    e_chi = math.sqrt(2.0) * math.gamma(80.0) / math.gamma(79.5)  # E||xi_perp||
    print(f"  E||xi_perp|| (d={80}) = {e_chi:.2f}")
    print("  {:>4s} {:>8s} {:>18s} {:>12s}".format(
        "t", "s_t", "noise/step/p2row", "climb/eta"))
    for t in (99, 90, 75, 50, 25, 10, 0):
        print("  {:>4d} {:>8.3f} {:>18.3f} {:>12.3f}".format(
            t, s_t[t], SIGMA * s_t[t] * e_chi, ETA * s_t[t]))
    print("  mean over steps: s_t={:.3f} noise={:.3f}".format(
        s_t.mean(), SIGMA * s_t.mean() * e_chi))

    print()
    print("=" * 78)
    print("PART 3  VIB gain probe: KL & |dKL/da| vs ||delta a|| "
          "(per round, E_s = round DP)")
    print("=" * 78    )
    sq_pairs = [("base", f"{SQ}/train/DP/DP-base/checkpoints/599.ckpt",
                 f"{SQ}/train/dyn/dyn-base/20260826-112119/scout_vib.ckpt")]
    for n, ts in enumerate(["20260901-165102", "20260901-213840",
                            "20260902-023448", "20260902-075528"], 1):
        sq_pairs.append((f"exp{n}",
                         f"{SQ}/train/DP/DP-SCOUT-exp{n}/checkpoints/299.ckpt",
                         f"{SQ}/train/dyn/dyn-SCOUT-exp{n}/{ts}/"
                         "scout_vib.ckpt"))
    can_pairs = [("base", f"{CAN}/train/DP/DP-base/checkpoints/599.ckpt",
                  f"{CAN}/train/dyn/dyn-base/20260824-232156/"
                  "scout_vib.ckpt"),
                 ("exp1", f"{CAN}/train/DP/DP-SCOUT-exp1/checkpoints/299.ckpt",
                  f"{CAN}/train/dyn/dyn-SCOUT-exp1/20260902-125340/"
                  "scout_vib.ckpt")]
    import torch
    dev = torch.device("cuda")
    for tag, pairs, cfg, core in (
            ("SQUARE", sq_pairs,
             "configs/eval_square_entropy.yaml", f"{SQ}/rollout/square_core.hdf5"),
            ("CAN", can_pairs,
             "configs/eval_can_entropy.yaml", f"{CAN}/rollout/can_core.hdf5")):
        print(f"\n  --- {tag} (chunk std & posterior sigma0 in header) ---")
        print("  {:>6s} {:>6s} {:>6s}".format("r", "KL", "|g|/row")
              + " {:>6s} {:>6s}".format("p2%", "cap%"))
        for name, dp, vibp in pairs:
            try:
                res = probe(cfg, dp, vibp, core, dev)
            except Exception as e:                      # noqa: BLE001
                print(f"  {tag}-{name}: FAILED {type(e).__name__}: {e}")
                continue
            print(f"  == {name}: chunk_std={res['scale']:.3f} "
                  f"sigma0={res['sigma0']:.3f}")
            for r in (0.05, 0.1, 0.25, 0.5, 1.0):
                kl, gn, p2, cap = res[r]
                print("  {:>6.2f} {:>6.2f} {:>8.3f} {:>6.0f} {:>6.0f}".format(
                    r, kl, gn, 100 * p2, 100 * cap))
    print("\nDONE")


if __name__ == "__main__":
    main()
