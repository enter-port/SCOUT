"""Model / env factory constructors + config loader for the eval pipeline.

Extracted from ``self_improvement.py`` so that :class:`EvalPipeline`,
:class:`RolloutCollector`, and :class:`SelfImprovementLoop` all share one
source of truth for how a base DP / ScoutVIB / robomimic env is built. None of
these constructors carry eval-vs-rollout bias -- they are pure plumbing.

Public API:

  * :func:`load_cfg`            -- read a YAML eval config (utf-8; EasyDict).
  * :func:`make_lpb_dp_factory` -- ``dp_factory(ckpt_path) -> ScoutPolicy``.
  * :func:`make_scout_vib_factory` -- ``scout_vib_factory() -> ScoutVIB``.
  * :func:`make_default_env_factory` -- robomimic env factory from a base-DP cfg.
  * Type aliases ``DPFactory`` / ``VIBFactory`` / ``EnvFactory`` / ``RetrainFn``.

All robomimic / hydra / mujoco imports are LAZY (inside the factory closures)
so this module stays importable without those deps -- the dry-runs rely on it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

import torch
import yaml
from easydict import EasyDict


# --------------------------------------------------------------------------- #
# type aliases (shared across evaluator / collector / self-improvement loop)
# --------------------------------------------------------------------------- #
DPFactory = Callable[[str], torch.nn.Module]
VIBFactory = Callable[[], torch.nn.Module]
EnvFactory = Callable[[], Any]
RetrainFn = Callable[
    [EasyDict, int, List[dict], str],  # cfg, round_idx, successful_rollouts, prev_dp_ckpt
    str,                                # new DP ckpt path
]


# --------------------------------------------------------------------------- #
# config loader
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> EasyDict:
    """Read a YAML eval config into an EasyDict.

    ``encoding=utf-8``: configs carry §/β/smart-quotes in comments; the Windows
    locale default (GBK) would otherwise UnicodeDecodeError on them.
    """
    with open(path, "r", encoding="utf-8") as f:
        return EasyDict(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# model / env factories (real run only -- lazy imports)
# --------------------------------------------------------------------------- #
def make_lpb_dp_factory(device: Optional[torch.device] = None) -> DPFactory:
    """Build a ``dp_factory(ckpt_path) -> ScoutPolicy`` for the real run.

    Loads the LPB base DP ckpt (saved by ``TrainDiffusionUnetHybridWorkspace``)
    into a :class:`scout.guidance.policy.ScoutPolicy` (subclass; same state_dict
    structure -- no new params). Fits the action normalizer from the sibling
    ``normalizer.pth`` (saved by the LPB workspace alongside ``checkpoints/``).

    Lazy-imported so this module stays importable without robomimic/hydra. The
    dry-runs do NOT use this -- they inject a mock.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def factory(ckpt_path: str) -> torch.nn.Module:
        import hydra
        from omegaconf import OmegaConf
        from scout.guidance.policy import ScoutPolicy
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload["cfg"] if isinstance(payload, dict) and "cfg" in payload \
            else payload
        # Build a ScoutPolicy (LPB DP subclass) -- NOT the parent -- so its
        # guided_conditional_sample override is the one predict_action_dyn_guided
        # resolves to. Override _target_ on the policy cfg; ScoutPolicy.__init__
        # forwards all params to the LPB parent (scout_planner defaults None).
        pcfg = OmegaConf.to_container(cfg.policy, resolve=True)
        pcfg["_target_"] = "scout.guidance.policy.ScoutPolicy"
        policy = hydra.utils.instantiate(pcfg)

        # state_dict slot: the LPB workspace stores the policy under the "model"
        # attribute (TrainDiffusionUnetHybridImageWorkspace: self.model), so
        # payload["state_dicts"] keys are "model"/"ema_model"/"optimizer" -- NOT
        # "policy". Fall back to "policy"/"state_dict" for other savers.
        if "state_dicts" in payload:
            sds = payload["state_dicts"]
            sd = sds.get("model", sds.get("policy", sds.get("ema_model")))
        else:
            sd = payload.get("model", payload.get("policy", payload.get("state_dict")))
        policy.load_state_dict(sd, strict=False)

        # normalizer (sibling normalizer.pth next to the ckpt's checkpoints/ dir)
        normalizer_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "normalizer.pth")
        if os.path.isfile(normalizer_path):
            nstate = torch.load(normalizer_path, map_location="cpu")
            policy.normalizer = LinearNormalizer()
            policy.normalizer.load_state_dict(nstate)

        return policy.to(dev).eval()

    return factory


def make_scout_vib_factory(cfg: EasyDict,
                           device: Optional[torch.device] = None) -> VIBFactory:
    """Build a ``scout_vib_factory() -> ScoutVIB`` for the real run.

    Reconstructs ``E_s`` via :meth:`StateEncoder.from_base_dp_ckpt` (frozen
    base-DP ResNet + trained proprio Conv1d) using the E_s params in
    ``cfg.vib`` (``base_dp_ckpt``, ``view_names``, ``proprio_dim``,
    ``proprio_emb_dim``), then loads the chosen-β VIB ckpt
    (``cfg.vib.ckpt_path``). Lazy-imported; the dry-runs inject a mock.

    The E_s params MUST match the E1 VIB training config (``configs/vib_lift_*``)
    -- duplicate them in the eval config's ``vib:`` block so the consumer is
    self-contained.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def factory(vib_ckpt=None) -> torch.nn.Module:
        from scout.model.encoder import StateEncoder
        from scout.model.scout_vib import ScoutVIB

        vcfg = cfg.vib
        E_s = StateEncoder.from_base_dp_ckpt(
            base_dp_ckpt=vcfg.base_dp_ckpt,
            view_names=list(vcfg.view_names),
            proprio_dim=int(vcfg.proprio_dim),
            proprio_emb_dim=int(getattr(vcfg, "proprio_emb_dim", 64)),
        )
        ckpt_path = vib_ckpt if vib_ckpt is not None else vcfg.ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # The VIB encoder was trained on the FLATTENED fs-step action chunk
        # (train_vib._slice_transition: a_t = first `frameskip` per-step actions
        # flattened, e.g. 8x10 = 80-dim), NOT the per-step action. So its action
        # input dim is whatever was saved -- inferred here as (encoder first-
        # layer width - s_bar_dim), NOT cfg.action_dim (10). Building with the
        # per-step dim would mismatch the saved Linear weights (1168 vs 1098).
        sd = ckpt["state_dict"]
        enc_in = sd["vib_enc.net.encoder.0.weight"].shape[1]
        action_dim = enc_in - int(E_s.s_bar_dim)
        model = ScoutVIB(
            action_dim=action_dim,
            E_s=E_s,
            style_dim=int(vcfg.style_dim),
            hidden_dim=int(getattr(vcfg, "hidden_dim", 128)),
            beta=float(ckpt.get("beta", 1.0e-3)),
        )
        model.load_state_dict(sd)
        return model.to(dev).eval()

    return factory


def make_default_env_factory(cfg: EasyDict) -> EnvFactory:
    """Build the real robomimic env factory (LPB robomimic_image_runner reuse).

    Lazy-imports :func:`scout.eval.rollout.make_robomimic_env_factory`; reads
    dataset_path + shape_meta from a base-DP LPB config (resolved via the
    cfg.base_dp config_name). The dry-runs do NOT use this -- they inject a mock.
    """
    from scout.eval.rollout import make_robomimic_env_factory
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = os.path.join(cfg.base_dp.config_dir,
                               cfg.base_dp.config_name + ".yaml")
    with open(config_path) as f:
        lpb_cfg = OmegaConf.create(yaml.safe_load(f))
    # n_obs_steps + abs_action come from the base-DP config (the policy's
    # obs-stacking + the env's abs controller must match training).
    # dataset_path (env-meta source) = the eval/rollout core hdf5
    # (cfg.dataset.path, set by the CLI's --core-hdf5); fall back to the base-DP
    # config's task.dataset_path only when the former is unset/placeholder.
    ds = cfg.dataset.path
    if (not ds) or str(ds).startswith("<"):
        ds = OmegaConf.select(lpb_cfg, "task.dataset_path", default=ds)
    return make_robomimic_env_factory(
        dataset_path=ds,
        shape_meta=OmegaConf.to_container(lpb_cfg.shape_meta, resolve=True),
        n_obs_steps=int(OmegaConf.select(lpb_cfg, "n_obs_steps", default=2)),
        abs_action=bool(OmegaConf.select(lpb_cfg, "abs_action", default=False)),
    )
