# SCOUT VIB 训练入口。结构对齐 SOE 的 src/train_single_gpu.py
# (AdamW + cosine warmup + ckpt + loss plot + 自定义 backward 触发),
# 区别只在数据集(转移三元组)与策略(SCOUT)的分支。

import os
import sys

# 路径:自身目录(scout 模块)+ SOE/src(基类、utils)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SOE_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "SOE", "src"))
if _SOE_SRC not in sys.path:
    sys.path.insert(0, _SOE_SRC)

import time
import glob
import json
import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.training import set_seed  # noqa: E402  (SOE)


def plot_history(history, epoch, ckpt_dir, name):
    plt.figure()
    plt.plot(history, label=name)
    plt.xlabel("epoch")
    plt.ylabel(name)
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, name + ".png"))
    plt.close()


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    log_root = os.path.join(cfg.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(log_root, exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = open(os.path.join(log_root, "log.txt"), "w")

    # --- dataset ---
    print("Loading dataset ...")
    if cfg.dataset.name == "scout_transition":
        from transition_dataset import ScoutTransitionDataset, collate_fn, pre_process_data
        dataset = ScoutTransitionDataset(**cfg.dataset.params)
    else:
        raise NotImplementedError("unsupported dataset.name: {}".format(cfg.dataset.name))
    print("dataset size:", len(dataset))

    if not hasattr(cfg, "num_iters_per_epoch") or cfg.num_iters_per_epoch is None:
        num_samples = len(dataset)
    else:
        num_samples = cfg.num_iters_per_epoch * cfg.batch_size
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        collate_fn=collate_fn, sampler=sampler,
    )

    # --- policy ---
    print("Loading policy ...")
    if cfg.policy.name == "SCOUT":
        from scout_policy import SCOUT
        policy = SCOUT(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError("unsupported policy.name: {}".format(cfg.policy.name))
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # --- checkpoint ---
    if cfg.resume_ckpt is not None:
        resume_ckpt = cfg.resume_ckpt.format(seed=cfg.seed)
        resume_ckpt_list = glob.glob(resume_ckpt)
        assert len(resume_ckpt_list) > 0, "Checkpoint {} not found.".format(resume_ckpt)
        resume_ckpt = sorted(resume_ckpt_list)[-1] if len(resume_ckpt_list) > 1 else resume_ckpt_list[0]
        policy.load_state_dict(torch.load(resume_ckpt, map_location=device), strict=False)
        cfg.resume_ckpt = resume_ckpt
        print("Checkpoint {} loaded.".format(resume_ckpt))

    ckpt_dir = os.path.join(log_root, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- optimizer / scheduler ---
    print("Loading optimizer and scheduler ...")
    assert cfg.optimizer.name == "AdamW"
    optimizer = torch.optim.AdamW(policy.parameters(), **cfg.optimizer.params)
    assert cfg.lr_scheduler.name == "cosine_with_warmup"
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=2000,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )
    lr_scheduler.last_epoch = len(dataloader) * (cfg.resume_epoch + 1) - 1

    with open(os.path.join(log_root, "config.json"), "w") as f:
        json.dump(dict(cfg), f, indent=4)

    # --- training ---
    train_history = {}
    policy.train()
    for epoch in range(cfg.resume_epoch + 1, cfg.num_epochs):
        print("Epoch {}".format(epoch))
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader)
        avg_loss = {"loss": 0.0}

        for data in pbar:
            batch = pre_process_data(data, device)
            loss = policy(**batch)
            if isinstance(loss, dict):
                for key in loss:
                    avg_loss[key] = avg_loss.get(key, 0.0) + loss[key].item()
                final_loss = loss["loss"]
            else:
                avg_loss["loss"] += loss.item()
                final_loss = loss
            # SCOUT 自带 backward(直链,无需隔离)
            if hasattr(policy, "backward"):
                policy.backward(loss)
            else:
                final_loss.backward()
            if hasattr(cfg, "clip_grad_norm") and cfg.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        for key in avg_loss:
            avg_loss[key] /= num_steps
        for key in avg_loss:
            train_history.setdefault(key, []).append(avg_loss[key])

        print("Train loss: {:.6f}".format(avg_loss["loss"]), end=" ")
        for key in train_history:
            print("{}: {:.6f}".format(key, train_history[key][-1]), end=" ")
        print("")

        if (epoch + 1) % cfg.save_epochs == 0 \
                or (epoch + 1) in list(range(cfg.num_epochs - 100, cfg.num_epochs, 10)):
            torch.save(policy.state_dict(),
                       os.path.join(ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, cfg.seed)))
            for key in train_history:
                plot_history(train_history[key], epoch, ckpt_dir, key + "_seed_{}.png".format(cfg.seed))

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy_last.ckpt"))

    sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file (json)")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    print(cfg)
    train(cfg)
