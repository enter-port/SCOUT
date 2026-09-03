"""Decompose the guidance-gradient blowup on the e2-SCOUT chain (2026-08-21).

Question (user): is the |dNLL/da| rise (9 -> 6 -> 10 -> 39 -> 109 across
dyn-exp1..5) a VIB collapse driven by growing data?

Method:
  A. data drift  -- per-round explore trajectories (the accumulating training
     data): count / success / SOE jerk / action std.
  B. fixed batch -- ONE fixed (obs, a_chunk) batch from the CORE hdf5 (same
     raw inputs for every model); for each dyn ckpt (base, exp1..5), with its
     paired DP as the E_s source:
       |mu|, KL(q||N(0,I)), sigma stats, |mu - z| (z ~ N(0,I), seeded),
       |dmu/da|_F, and |dNLL/da| in three variants:
         plain  : d/da 0.5*sum (z-mu)^2           (= (mu-z) . dmu/da)
         sigw   : d/da 0.5*sum (z-mu)^2/sig^2     (sigma detached)
         full   : d/da 0.5*sum [(z-mu)^2/sig^2 + log sig^2]  (the real cost)
Collapse would show KL -> 0, |mu| -> 0, sigma -> 1, ALL grads -> 0.

Run:  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python soe_scripts/diag_grad_decompose.py
"""
import glob
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml
from easydict import EasyDict

DEV = torch.device("cuda")


def soe_jerk(a):
    if len(a) < 4:
        return 0.0
    d3 = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    return float(np.linalg.norm(d3, axis=-1).mean())


def part_a():
    print("== A. per-round explore trajectories (the accumulating train data) ==")
    for r in range(1, 7):
        p = f"data/experiment2/can/rollout/SCOUT-exp{r}/all.hdf5"
        if not os.path.exists(p):
            continue
        js, succ, n, stds = [], 0, 0, []
        import h5py
        with h5py.File(p, "r") as f:
            for k in f["data"].keys():
                if not str(k).startswith("demo"):
                    continue
                if int(k.split("_")[1]) < 20:
                    continue
                a = np.asarray(f["data"][k]["actions"][()], dtype=np.float64)
                js.append(soe_jerk(a))
                succ += bool(f["data"][k]["success"][0])
                n += 1
                stds.append(float(a.std(axis=0).mean()))
        print(f"  r{r}: n={n:3d} succ={succ:3d}  data_jerk={np.mean(js):.3f}  "
              f"act_std={np.mean(stds):.3f}")


def part_b():
    from scout.eval.factories import load_cfg, make_scout_vib_factory
    from scout.train_vib import make_dataloader, _slice_transition
    from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet

    vib_cfg = yaml.safe_load(open("configs/vib_can_image_e2.yaml"))
    vib_cfg["dataset"]["feature_cache"] = False
    vib_cfg["dataset"]["num_workers"] = 0
    torch.manual_seed(0)
    loader, _ds = make_dataloader(EasyDict(vib_cfg))
    batch = next(iter(loader))
    t_crop = get_eval_crop_transform_resnet(84, 76)
    obs_t, a_t, _, _ = _slice_transition(batch, DEV, t_crop)
    print(f"== B. fixed core batch: B={a_t.shape[0]} a_dim={a_t.shape[1]} ==")

    ecfg = load_cfg("configs/eval_can_e2.yaml")
    ecfg.dataset.path = "data/experiment2/can/rollout/can_core.hdf5"

    def newest_ckpt(d):
        return sorted(glob.glob(os.path.join(d, "checkpoints", "*.ckpt")),
                      key=os.path.getmtime)[-1]

    def newest_vib(d):
        return sorted(glob.glob(os.path.join(d, "*", "scout_vib.ckpt")),
                      key=os.path.getmtime)[-1]

    pairs = [("base", "data/experiment2/can/train/dyn/dyn-base",
              "data/experiment2/can/train/DP/DP-base")]
    for r in range(1, 6):
        pairs.append((f"exp{r}",
                      f"data/experiment2/can/train/dyn/dyn-SCOUT-exp{r}",
                      f"data/experiment2/can/train/DP/DP-SCOUT-exp{r}"))

    print("  %-5s %7s %6s %8s %8s %8s %8s %9s %9s %9s %9s" % (
        "tag", "|mu|", "KL", "sig_mean", "sig_min", "sig_max", "|mu-z|",
        "|dmu/da|", "|g_plain|", "|g_sigw|", "|g_full|"))
    for tag, dyn_dir, dp_dir in pairs:
        try:
            vib = newest_vib(dyn_dir)
            ecfg.vib.ckpt_path = vib
            ecfg.vib.base_dp_ckpt = newest_ckpt(dp_dir)
            model = make_scout_vib_factory(ecfg, DEV)(vib)
        except Exception as e:
            print(f"  {tag}: SKIP ({e})")
            continue
        model.eval()
        with torch.no_grad():
            s_bar = model.encode(obs_t)
        a_g = a_t.clone().requires_grad_(True)
        mu, logvar = model.vib_enc(s_bar, a_g)
        sig2 = logvar.exp()
        mu_n = float(mu.norm(dim=1).mean())
        kl = float((0.5 * (mu ** 2 + sig2 - 1.0 - logvar)).sum(1).mean())
        sig = sig2.sqrt()
        g_full, g_sigw, g_plain, muz = [], [], [], []
        for zi in range(3):
            gen = torch.Generator(device="cpu").manual_seed(100 + zi)
            z = torch.randn(mu.shape, generator=gen).to(DEV)
            nll = 0.5 * ((z - mu) ** 2 / sig2 + logvar).sum(1).mean()
            g_full.append(float(torch.autograd.grad(nll, a_g, retain_graph=True)[0].norm()))
            lw = 0.5 * ((z - mu) ** 2 / sig2.detach()).sum(1).mean()
            g_sigw.append(float(torch.autograd.grad(lw, a_g, retain_graph=True)[0].norm()))
            lp = 0.5 * ((z - mu) ** 2).sum(1).mean()
            g_plain.append(float(torch.autograd.grad(lp, a_g, retain_graph=True)[0].norm()))
            muz.append(float((z - mu).norm(dim=1).mean()))
        jsq = torch.zeros(mu.shape[0], device=DEV)
        for i in range(mu.shape[1]):
            g = torch.autograd.grad(mu[:, i].sum(), a_g, retain_graph=True)[0]
            jsq += g.pow(2).sum(dim=1)
        jac = float(jsq.sqrt().mean())
        print("  %-5s %7.3f %6.3f %8.4f %8.4f %8.4f %8.3f %9.3f %9.3f %9.3f %9.3f" % (
            tag, mu_n, kl, float(sig.mean()), float(sig.min()), float(sig.max()),
            float(np.mean(muz)), jac, np.mean(g_plain), np.mean(g_sigw),
            np.mean(g_full)))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    part_a()
    part_b()
