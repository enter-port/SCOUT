# 从一个 robomimic low_dim hdf5 自动生成两份 config:
#   (1) base DP low_dim config  -> 用 SOE 的 train_single_gpu.py 训练冻结的 base DP
#   (2) SCOUT config            -> 用 scout/src/train_scout.py 训练 VIB 动力学模型
#
# 为什么要这个脚本:robomimic 各任务的 low_dim 观测维度(object 等)不同,
# 直接硬编码容易写错。本脚本直接从 hdf5 读 obs key 的真实形状,避免猜维度。
#
# 用法:
#   python scout/src/make_configs.py \
#       --dataset SOE/simulation/datasets/lift/ph/low_dim_v141.hdf5 \
#       --task lift --out_dir scout/configs
# 然后:
#   cd SOE/src && python train_single_gpu.py --config <abs>/scout/configs/dp_lift_lowdim.json
#   python scout/src/train_scout.py            --config <abs>/scout/configs/scout_lift.json

import os
import sys
import json
import argparse

import h5py

# 仓库根目录(scout/src 的上两级),用于生成绝对路径的 log_dir,与运行目录无关
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def read_dataset_info(dataset_path):
    """读 obs key 列表与各自维度、动作维度。"""
    f = h5py.File(dataset_path, "r")
    d0 = list(f["data"].keys())[0]
    obs_keys = list(f["data/" + d0 + "/obs"].keys())
    key_dims = {k: int(f["data/" + d0 + "/obs/" + k].shape[1]) for k in obs_keys}
    action_dim = int(f["data/" + d0 + "/actions"].shape[1])
    f.close()
    return obs_keys, key_dims, action_dim


def make_base_dp_config(dataset_path, obs_keys, key_dims, action_dim, task, seed=233):
    """生成 base DP 的 low_dim config(走 SOE train_single_gpu.py)。"""
    obs_shape_meta = {k: {"shape": [key_dims[k]], "type": "low_dim"} for k in obs_keys}
    return {
        "seed": seed,
        "log_dir": os.path.join(_REPO_ROOT, "scout/out/dp_{}_lowdim".format(task)),
        "batch_size": 256,
        "num_workers": 4,
        "num_epochs": 600,
        "save_epochs": 100,
        "resume_epoch": -1,
        "resume_ckpt": None,
        "clip_grad_norm": 1.0,
        "dataset": {
            "name": "robomimic",
            "params": {
                "path": os.path.abspath(dataset_path),
                "train_filter_key": "train",
                "num_obs": 1,
                "num_action": 20,
                "output_keys": ["actions"],
                "success_only": False,
                "normalize_actions": False,
            },
        },
        "policy": {
            "name": "DP",
            "params": {
                "num_action": 20,
                "action_dim": action_dim,
                "obs_shape_meta": obs_shape_meta,
                "use_group_norm": True,
                "resnet_out_features": 64,
                "readout_dim": None,
                "hidden_dim": None,
                "weight_type": None,
                # 以下透传给 DiffusionUNetPolicy
                "num_inference_steps": 20,
                "diffusion_step_embed_dim": 256,
                "down_dims": [256, 512],
                "kernel_size": 5,
                "n_groups": 8,
                "cond_predict_scale": True,
                "noise_scheduler_type": "ddim",
            },
        },
        "optimizer": {"name": "AdamW", "params": {"lr": 1e-4, "weight_decay": 1e-6}},
        "lr_scheduler": {"name": "cosine_with_warmup"},
    }


def make_scout_config(dataset_path, obs_keys, key_dims, action_dim, task, seed=233,
                      kl_weight=1e-3, style_dim=16):
    """生成 SCOUT VIB 训练 config(走 scout/src/train_scout.py)。"""
    state_dim = int(sum(key_dims[k] for k in obs_keys))
    return {
        "seed": seed,
        "log_dir": os.path.join(_REPO_ROOT, "scout/out/scout_{}".format(task)),
        "batch_size": 256,
        "num_workers": 4,
        "num_epochs": 500,
        "save_epochs": 100,
        "resume_epoch": -1,
        "resume_ckpt": None,
        "clip_grad_norm": 1.0,
        "dataset": {
            "name": "scout_transition",
            "params": {
                "path": os.path.abspath(dataset_path),
                "obs_keys": list(obs_keys),
                "train_filter_key": "train",
                "action_offset": 1,   # 与 SOE 对齐:actions[t+1] 驱动 obs[t]->obs[t+1]
                "normalize": False,   # 须与 base DP 保持一致
                "success_only": False,
            },
        },
        "policy": {
            "name": "SCOUT",
            "params": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "style_dim": style_dim,
                "hidden_dim": 256,
                "kl_weight": kl_weight,   # beta —— make-or-break 旋钮,建议扫 1e-4..1e-1
            },
        },
        "optimizer": {"name": "AdamW", "params": {"lr": 1e-4, "weight_decay": 1e-6}},
        "lr_scheduler": {"name": "cosine_with_warmup"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="robomimic low_dim hdf5 路径")
    parser.add_argument("--task", type=str, default="lift", help="任务名(用于命名 config/输出目录)")
    parser.add_argument("--out_dir", type=str, default="scout/configs", help="config 输出目录")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="beta (KL 权重)")
    parser.add_argument("--style_dim", type=int, default=16)
    args = parser.parse_args()

    obs_keys, key_dims, action_dim = read_dataset_info(args.dataset)
    print("obs_keys:", obs_keys)
    print("key_dims:", key_dims)
    print("action_dim:", action_dim)
    state_dim = sum(key_dims.values())
    print("=> state_dim:", state_dim)

    os.makedirs(args.out_dir, exist_ok=True)
    dp_cfg = make_base_dp_config(args.dataset, obs_keys, key_dims, action_dim, args.task, args.seed)
    scout_cfg = make_scout_config(args.dataset, obs_keys, key_dims, action_dim, args.task,
                                  args.seed, args.kl_weight, args.style_dim)

    dp_path = os.path.join(args.out_dir, "dp_{}_lowdim.json".format(args.task))
    scout_path = os.path.join(args.out_dir, "scout_{}.json".format(args.task))
    with open(dp_path, "w") as f:
        json.dump(dp_cfg, f, indent=4)
    with open(scout_path, "w") as f:
        json.dump(scout_cfg, f, indent=4)
    print("wrote:", dp_path)
    print("wrote:", scout_path)
    print("\n下一步:")
    print("  cd SOE/src && python train_single_gpu.py --config {}".format(os.path.abspath(dp_path)))
    print("  python scout/src/train_scout.py --config {}".format(os.path.abspath(scout_path)))


if __name__ == "__main__":
    main()
