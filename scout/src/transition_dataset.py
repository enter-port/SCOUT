# SCOUT transition dataset: produces single-step (S_t, A_t, S_{t+1}) triples.
# 单步转移数据集:产出 (S_t, A_t, S_{t+1})。
#
# 与 SOE 的 RoboMimicDataset (dataset/robomimic_v2.py) 的关系:
#   - 复用其 hdf5 整文件加载工具 (load_entire_hdf5 / get_from_loaded_hdf5);
#   - 但 RoboMimicDataset 只输出 obs[t] 与未来动作 chunk,不输出 S_{t+1};
#     SCOUT 的 VIB 动力学模型需要显式的下一状态,故单独写一个数据集。
#
# 动作对齐约定(与 SOE 一致):SOE 的 RoboMimicDataset 用 obs[cur_idx] 预测
# actions[cur_idx+1 ...],即 actions[t+1] 是「在 obs[t] 处执行、驱动 obs[t]->obs[t+1]」
# 的动作。因此本数据集对帧 f 输出:
#       S_t   = concat(obs[key][f]    for key in obs_keys)
#       A_t   = actions[f + action_offset]        # 默认 offset=1,与 SOE 对齐
#       S_{t+1}= concat(obs[key][f + 1] for key in obs_keys)

import os as _os
import sys as _sys

# 把 SOE/src 加进路径,以复用其 hdf5 工具与 collate_fn
_S = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "SOE", "src"))
if _S not in _sys.path:
    _sys.path.insert(0, _S)

import h5py
import numpy as np
import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset

from dataset.robomimic_v2 import load_entire_hdf5, get_from_loaded_hdf5, collate_fn  # noqa: E402


class ScoutTransitionDataset(Dataset):
    """
    单步转移数据集。

    Args:
        path: robomimic hdf5 路径(low_dim 即可,不需要图像)。
        obs_keys: 用于拼接成状态向量的 obs key 列表(须与 hdf5 中 data/<demo>/obs/ 下的 key 一致)。
        train_filter_key: 用 mask/<key> 选 demo(None=用全部 data);默认 "train"。
        action_offset: 动作帧偏移;1=与 SOE 对齐(actions[t+1] 驱动 obs[t]->obs[t+1])。
        normalize: 是否对 state / action 做 z-score(均值方差在全部帧上统计)。
            注意:若 base DP 不做归一化,这里也应保持 False,以保证 train/test 一致。
        demo_num_limit: 只用前 N 条 demo(调试用)。
    """

    def __init__(
        self,
        path,
        obs_keys,
        train_filter_key="train",
        success_only=False,
        action_offset=1,
        normalize=False,
        demo_num_limit=None,
    ):
        # 排序:与 SOE MultiImageObsEncoder 的 low_dim 拼接顺序(sorted)保持一致,
        # 这样 SCOUT 的 state 向量 == base DP 的 readout 向量(同序拼接)。
        self.obs_keys = sorted(obs_keys)
        self.action_offset = int(action_offset)
        self.normalize = bool(normalize)

        # --- 加载 hdf5 ---
        self.hdf5_file_path = path if isinstance(path, str) else None
        self.hdf5_file = h5py.File(path, "r") if isinstance(path, str) else path
        self.hdf5_loaded = load_entire_hdf5(self.hdf5_file)
        print("hdf5 file loaded:", self.hdf5_file_path or "<file handle>")

        # --- 选 demo ---
        if train_filter_key is None:
            all_demos = [d.encode("utf8") for d in self.hdf5_file["data"].keys()]
        else:
            all_demos = list(self.hdf5_file["mask/" + train_filter_key])
        if success_only:
            kept = []
            for d in all_demos:
                dp = "data/" + d.decode("utf8")
                if np.any(self.hdf5_file[dp + "/dones"][...]):
                    kept.append(d)
            all_demos = kept
        if demo_num_limit is not None:
            all_demos = all_demos[:demo_num_limit]
        self.all_demos = all_demos
        print("num_demos:", len(all_demos))

        # --- 从第一条 demo 推断维度 ---
        d0 = "data/" + all_demos[0].decode("utf8")
        self.key_dims = {k: int(self.hdf5_file[d0 + "/obs/" + k].shape[1]) for k in self.obs_keys}
        self.state_dim = int(sum(self.key_dims.values()))
        self.action_dim = int(self.hdf5_file[d0 + "/actions"].shape[1])
        print("obs_keys:", self.obs_keys)
        print("key_dims:", self.key_dims)
        print("state_dim:", self.state_dim, "action_dim:", self.action_dim)

        # --- 建索引:(demo_path, frame_id) ---
        # 需要 f 与 f+1、f+offset 都存在 => f < n - max(1, offset)
        self.index = []
        step = max(1, self.action_offset)
        for d in all_demos:
            dp = "data/" + d.decode("utf8")
            n = int(self.hdf5_file[dp].attrs["num_samples"])
            for f in range(0, n - step):
                self.index.append((dp, f))
        print("num_transitions:", len(self.index))

        # --- 可选归一化统计 ---
        if self.normalize:
            self._compute_stats()
        else:
            self.state_mean = self.state_std = None
            self.action_mean = self.action_std = None

    def _get_state(self, demo_path, frame_id):
        parts = [
            get_from_loaded_hdf5(self.hdf5_loaded, demo_path + "/obs/" + k)[frame_id]
            for k in self.obs_keys
        ]
        return np.concatenate(parts).astype(np.float32)

    def _compute_stats(self):
        # 在所有帧上累积 state / action 的均值与方差(state_dim/action_dim 都很小,全量统计)。
        s_sum = np.zeros(self.state_dim, dtype=np.float64)
        s_sqsum = np.zeros(self.state_dim, dtype=np.float64)
        a_sum = np.zeros(self.action_dim, dtype=np.float64)
        a_sqsum = np.zeros(self.action_dim, dtype=np.float64)
        count = 0
        for demo_path, f in self.index:
            s = self._get_state(demo_path, f)
            a = get_from_loaded_hdf5(self.hdf5_loaded, demo_path + "/actions")[f + self.action_offset].astype(np.float32)
            s_sum += s
            s_sqsum += s ** 2
            a_sum += a
            a_sqsum += a ** 2
            count += 1
        self.state_mean = (s_sum / count).astype(np.float32)
        self.state_std = (np.sqrt(s_sqsum / count - (s_sum / count) ** 2) + 1e-6).astype(np.float32)
        self.action_mean = (a_sum / count).astype(np.float32)
        self.action_std = (np.sqrt(a_sqsum / count - (a_sum / count) ** 2) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        demo_path, f = self.index[i]
        s_t = self._get_state(demo_path, f)
        a_t = get_from_loaded_hdf5(self.hdf5_loaded, demo_path + "/actions")[f + self.action_offset].astype(np.float32)
        s_tp1 = self._get_state(demo_path, f + 1)
        if self.normalize:
            s_t = (s_t - self.state_mean) / self.state_std
            s_tp1 = (s_tp1 - self.state_mean) / self.state_std
            a_t = (a_t - self.action_mean) / self.action_std
        return {
            "state_t": torch.from_numpy(s_t),
            "action_t": torch.from_numpy(a_t),
            "state_tp1": torch.from_numpy(s_tp1),
        }


def pre_process_data(data, device):
    """把 collate 后的 batch 搬到 device(SCOUT 只需三个张量)。"""
    return {k: v.to(device) for k, v in data.items()}
