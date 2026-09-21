"""Deterministic per-round recalibration of (eta, kappa) for the KL-cost
guidance (param-dev, 2026-09-21). Three MUTUALLY DISTINCT methods, one
subcommand each; every one is closed-form, deterministic, and probe-free
(no synthetic-displacement delta-ladder, no gradient measurement):

  m1  information-budget feedforward:  m = KL_train(r) / KL_train(base);
      eta_r = eta0 / m,  kappa_r = kappa0 * m.
      Signal: the dyn training summary's final kl term (logged by train_vib
      every retrain -- zero extra compute).

  m2  posterior-precision renormalization: v = geomean(var_r / var_base) of
      the ANCHOR posteriors on the frozen core batch (torch.manual_seed(0),
      num_workers=0); eta_r = eta0 * v, kappa_r = kappa0 / v.
      Signal: ONE no-grad forward pass of both encoders on real data points
      (reads only mu/logvar; no displacements, no autograd).

  m3  field-telemetry feedback controller (PPO adaptive-KL literal port):
      ratio = J / J_star (J = latest explore avg_jerk from the rollout json,
      J_star = calibration-day flight jerk at (eta0, kappa0) from the grid
      csv); ratio > 1.5 -> eta <- eta * 0.5; ratio < 1/1.5 -> eta * 1.5;
      else hold. kappa unchanged (its semantics is the dimensionless tilt
      bound e^kappa of the guided distribution, invariant to cost-surface
      rescaling).

Usage (server, .venv_mg):
  python scripts/coffee/cfk_param_calib.py m1 --base-summary <dyn-base/summary.yaml> \
      --round-summary <dyn-r/summary.yaml> --eta0 5.6 --kappa0 2.5 --out <json>
  python scripts/coffee/cfk_param_calib.py m2 --seed-data-root <COFFEE-MG-p1-s23333> \
      --base-vib <dyn-base ckpt> --round-vib <dyn-r ckpt> --eta0 5.6 --kappa0 2.5 --out <json>
  python scripts/coffee/cfk_param_calib.py m3 --explore-json <latest explore json> \
      --jerk-star 0.5065 --eta0 5.6 --kappa0 2.5 --out <json>
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, ".")


def _load_summary_kl(path):
    """train_vib summary.yaml carries the final KL term as 'kl: <float>'."""
    import yaml
    with open(path) as f:
        s = yaml.safe_load(f)
    # summary nests sections; search for any mapping value named 'kl'
    stack = [s]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == "kl" and isinstance(v, (int, float)):
                    return float(v)
                stack.append(v)
    raise KeyError(f"no 'kl' numeric field in {path}")


def _find_summary(dyn_dir):
    hits = sorted(glob.glob(os.path.join(dyn_dir, "*", "summary.yaml")))
    if not hits:
        raise FileNotFoundError(f"no summary.yaml under {dyn_dir}")
    return hits[-1]


def cmd_m1(args):
    kl0 = _load_summary_kl(_find_summary(args.base_dyn_dir))
    klr = _load_summary_kl(_find_summary(args.round_dyn_dir))
    m = klr / kl0
    eta = args.eta0 / m
    kap = args.kappa0 * m
    return {"method": "m1", "kl_base": kl0, "kl_round": klr, "m": m,
            "eta": eta, "kappa": kap,
            "rule": "eta=eta0/m, kappa=kappa0*m  (information-budget rescale)"}


def cmd_m2(args):
    import torch
    import yaml
    from easydict import EasyDict

    sdr = args.seed_data_root
    task_dir = os.path.join(sdr, args.task)
    core = args.core_hdf5 or os.path.join(task_dir, "rollout", args.core_name)
    for p in (core, args.base_vib, args.round_vib, args.dp_ckpt):
        assert os.path.exists(p), f"missing {p}"

    dev = torch.device("cuda")
    from scout.train_vib import make_dataloader, _slice_transition
    from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet

    vib_cfg = yaml.safe_load(open(args.vib_config))
    vib_cfg["dataset"]["zarr_path"] = core
    vib_cfg["dataset"]["num_workers"] = 0
    vib_cfg["dataset"]["feature_cache"] = False
    vib_cfg["batch_size"] = args.batch_size
    torch.manual_seed(0)
    loader, _ds = make_dataloader(EasyDict(vib_cfg))
    batch = next(iter(loader))
    t_crop = get_eval_crop_transform_resnet(84, 76)
    obs_t, a_t, _, _ = _slice_transition(batch, dev, t_crop)

    from scout.eval.factories import load_cfg, make_scout_vib_factory
    ecfg = load_cfg(args.eval_config)

    def anchor_stats(vib_ckpt):
        ecfg.vib.ckpt_path = vib_ckpt
        ecfg.vib.base_dp_ckpt = args.dp_ckpt
        model = make_scout_vib_factory(ecfg, dev)(vib_ckpt)
        model.eval()
        with torch.no_grad():
            s_bar = model.encode(obs_t)
            mu, lv = model.vib_enc(s_bar, a_t)
        return mu.detach(), lv.detach()

    # anchor posterior at the DATA action (a0 convention of the probe batch)
    mu0_b, lv0_b = anchor_stats(args.base_vib)
    mu0_r, lv0_r = anchor_stats(args.round_vib)
    # v = geomean over batch x dims of var_round / var_base
    log_v = (lv0_r - lv0_b).mean().item()
    v = float(torch.exp(torch.tensor(log_v)))
    eta = args.eta0 * v
    kap = args.kappa0 / v
    return {"method": "m2", "log_v": log_v, "v": v,
            "sig_base_mean": float(torch.exp(0.5 * lv0_b).mean()),
            "sig_round_mean": float(torch.exp(0.5 * lv0_r).mean()),
            "eta": eta, "kappa": kap,
            "rule": "eta=eta0*v, kappa=kappa0/v  (posterior-precision renorm)"}


def cmd_m3(args):
    with open(args.explore_json) as f:
        d = json.load(f)
    jerk = float(d["avg_jerk"])
    ratio = jerk / args.jerk_star
    if ratio > 1.5:
        factor = 0.5
    elif ratio < 1.0 / 1.5:
        factor = 1.5
    else:
        factor = 1.0
    eta = args.eta_current * factor
    return {"method": "m3", "jerk": jerk, "jerk_star": args.jerk_star,
            "ratio": ratio, "factor": factor, "eta": eta,
            "kappa": args.kappa0,
            "rule": "PPO-style multiplicative update on eta; kappa unchanged"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="method", required=True)

    p1 = sub.add_parser("m1")
    p1.add_argument("--base-dyn-dir", required=True,
                    help="e.g. <seed>/coffee/train/dyn/dyn-base")
    p1.add_argument("--round-dyn-dir", required=True)
    p1.add_argument("--eta0", type=float, required=True)
    p1.add_argument("--kappa0", type=float, required=True)
    p1.add_argument("--out", default=None)

    p2 = sub.add_parser("m2")
    p2.add_argument("--seed-data-root", required=True)
    p2.add_argument("--base-vib", required=True)
    p2.add_argument("--round-vib", required=True)
    p2.add_argument("--dp-ckpt", required=True)
    p2.add_argument("--vib-config", default="configs/vib_coffee_exp1.yaml")
    p2.add_argument("--eval-config", default="configs/eval_coffee_entropy.yaml")
    p2.add_argument("--task", default="coffee")
    p2.add_argument("--core-name", default="coffee_core.hdf5")
    p2.add_argument("--core-hdf5", default=None)
    p2.add_argument("--batch-size", type=int, default=128)
    p2.add_argument("--eta0", type=float, required=True)
    p2.add_argument("--kappa0", type=float, required=True)
    p2.add_argument("--out", default=None)

    p3 = sub.add_parser("m3")
    p3.add_argument("--explore-json", required=True,
                    help="latest field explore json (avg_jerk source)")
    p3.add_argument("--jerk-star", type=float, required=True,
                    help="calibration-day flight jerk at (eta0,kappa0)")
    p3.add_argument("--eta-current", type=float, required=True,
                    help="eta deployed in the flight that produced the json")
    p3.add_argument("--kappa0", type=float, required=True)
    p3.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.method == "m1":
        res = cmd_m1(args)
    elif args.method == "m2":
        res = cmd_m2(args)
    else:
        res = cmd_m3(args)

    print("[calib:%s] eta=%.6g kappa=%.6g" % (res["method"], res["eta"],
                                              res["kappa"]))
    for k, v in res.items():
        if k not in ("method", "eta", "kappa"):
            print("  %-14s %s" % (k, v))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("[calib] saved -> %s" % args.out)


if __name__ == "__main__":
    main()
