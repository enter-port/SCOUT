# SCOUT 训练流程提速报告（2026-08-14）

> **目标**：DP 训练 + VIB dynamics 训练 + rollout 三段串行一轮 ≤ 6 小时。
> **结论**：can ≈ 1h53m、square ≈ 4h30m，**均达标**（优化前 ~23.7h / ~27.5h，提速 12.6× / 6.1×）。
> 提速主要来自两处：VIB 的**冻结 ResNet 特征缓存**（19.5h → 13min）与 DP 的 **DataLoader / mid-train eval** 优化（3.5–5.8h → 56min–2h10）。rollout 代码未改动。
> 计时方法：每段任务 ≤5 分钟抽样、按 epoch 线性外推（除 VIB 快版为 300 epoch 完整真实跑完验证）。

---

## 1. 优化前后耗时对比

| 阶段 | 优化前 | 优化后 | 提速 |
|---|---|---|---|
| DP 训练 can（600ep） | ~3.5 h | **≈56 min**（105.5s/20ep 外推 + 2.9min 启动） | 3.8× |
| DP 训练 square（600ep） | ~5.8 h | **≈2h10**（254.5s/20ep + 3.0min 启动） | 2.7× |
| VIB 训练（300ep） | ~19.5 h | **13 min**（含 bank 构建 31–35s；18:12→18:25 完整真实跑完） | ~90× |
| rollout can | 39 min | 39 min（代码未动） | 1× |
| rollout square | 2h09 | 2h09（代码未动） | 1× |
| **串行一轮 can** | ~23.7 h | **≈1h53m** | 12.6× |
| **串行一轮 square** | ~27.5 h | **≈4h30m** | 6.1× |

lift 更短（base 已饱和、探索段无收益）；transport 未计时（4 相机 + 双臂、模型 263M+45M，DP 段预计更慢，如超预算见 §6）。

---

## 2. DP 训练提速（DataLoader + 跳过 mid-train eval）

**瓶颈定位**：config `num_workers: 0`（此前因服务器 `torch_shm_manager` 二进制受限被退回 0）→ 主进程单线程取数，4.6 it/s，GPU 大量空转。

**改动**：
1. `train.py`：`torch.multiprocessing.set_sharing_strategy('file_system')` —— /dev/shm（870G）本身正常，失效的只是 torch 的 shm 二进制；改文件系统共享策略后 DataLoader 多 worker 可用。
2. 训练 config：`num_workers 0→8` + `persistent_workers`（避免每 epoch 重建 worker）。
3. `training.rollout_every=0` 跳过 mid-train eval（原每 20 epoch 一次 50 test + 6 train rollout；纯观测、不触碰权重）。
4. `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`：env_runner **惰性构建**（跳过 eval 时不再实例化 28 个 env，省 ~5.8min 启动）。

**结果**：can 13–14 it/s，GPU 稳态利用率 77–88%，转为 GPU-bound。

**数值等价**：workers 数不影响训练结果——batch 顺序与 RandomCrop 的 RNG 都在主进程 policy 内生成；跳过的 eval 不影响权重。

**试过但已回滚**：`cudnn.benchmark` + TF32 matmul `high`——稳态已 GPU-bound，实测零收益，按最小 diff 原则还原。

**正式轮沿用方式**：同 `soe_scripts/fast_round_local/fast_round_<task>.sh` 里的 override（`training.rollout_every=0 training.num_workers=8 ...`），或直接把这些值写进正式 config。

---

## 3. VIB 训练提速（冻结 ResNet 特征缓存 —— 最大头）

**原理**：E_s 的 ResNet-18 冻结 + eval → 每个 (帧, 视角, crop offset) 的输出特征是**常数**。84×84 图上 RandomCrop(76) 恰有 9×9=81 个可能位置 → 一次性预计算特征库 `(N, 9, 9, 512)/视角` 存磁盘；训练时按 offset 查表，每步只跑 ~469k 可训练参数（proprio embed + VIB enc + D_s），完全跳过在线 ResNet 前向。

**实现**（新文件 `scout/feat_cache.py`）：
- `build_feature_bank`：按 (r,c) 遍历 81 个 offset，**全部视角一次性**过 ResNet（注意 `ResNetEncoder.forward` 遍历其自身全部 view_names，单视角 dict 会 KeyError）；预处理逐行镜像 base dataset。
- `get_or_build_bank`：磁盘缓存 `<zarr路径>.featbank.<md5(DP ckpt)[:8]>.<view>.npy` + json 指纹（帧数/视角/尺寸/DP ckpt），mmap 加载。
- `CachedFeatureTransitionDataset`：镜像 base 数据集索引（含末 chunk 重复项）；**train 每帧独立均匀采 81 个 offset**（与原 RandomCrop 的边缘分布一致，数据增广等价）；val 用中心 offset（= CenterCrop）。
- `verify_bank`：随机锚点 bank vs 在线比对，atol=5e-3。
- 配套方法：`encoder.forward_from_feats`、`scout_vib.encode_from_feats / forward_feats`（与原路径同序融合、同 loss 含 target `.detach()`）；`train_vib.py` 增加 `feature_cache: true` 分支。

**另一处独立优化**：训练循环每步 3 次 `.item()`（GPU→CPU 同步）改为 GPU tensor 累加、epoch 末一次同步 —— VIB 每 epoch 7.1s → 2.04s。

**等价性验证**：
- 同 batch 形状下 bank vs 在线 ResNet **逐位相等（0.00e+00）**；
- 不同形状（b1 vs b512）差 3–6e-4 —— 这是在线管线自身也存在的 cuDNN/TF32 形状噪声（在线 b1 vs b512 同量级），故 `verify_bank` atol=5e-3 并在代码注释中写明依据；
- `python -m scout.feat_cache --smoke` 全绿（bank 回环、dataset schema、`forward == forward_from_feats`、offset 多样性）。

**结果**：19.5h → **13 min**（300 epoch 完整真实运行验证，train/val loss 与慢版同水平）。

**使用注意**：`configs/vib_{can,square}_image_fast.yaml`（`feature_cache: true`）的 `es_base_dp_ckpt` 指向 timing-test 的 DP ckpt（已随测试数据删除）。**正式跑快版时把该路径改回正式 DP ckpt 即可**，bank 按 ckpt md5 自动重建（~35s/5.3k 帧）。

---

## 4. Rollout：代码未动 + 剩余空间分析（未实施）

实测 can 39min / square 2h09。square 耗时构成（估算）：~75% explore 段（guided 去噪循环的 UNet 前向 + 后 50 步 `autograd.grad` 反向，与 env offscreen 渲染约各一半）+ ~12% baseline 段 + ~8% 启动/写盘。

已识别但**未实施**的候选（都会改变结果采样，实施前需拍板）：
1. **`n_envs 50→100`**：policy 批调用次数减半，预期 policy 段 −30~50%。但 RNG 走全局 torch 流，batching 组合改变 → 100 个 init 场景不变、探索轨迹的采样会不同（同分布、不同样本）。
2. **replan/step 线程重叠**：env 步进藏进 GPU 推理窗口，收益上限 = max(policy, env) 替代二者之和；z-per-slot 语义可保持，但 DDPM 噪声采样组合同样会变。
3. **减少 DDPM 去噪步数**：改变语义，已排除。

分项 bench 脚本 `soe_scripts/fast_round_local/bench_rollout_components.py` 因 ScoutPolicy 在 bench 进程内落到无 robomimic 的 import fallback 壳（`TypeError: object.__init__() ...`）未跑通——**square rollout 的进一步提速工作未完成**，上述收益均为估算，实测前不要当作依据。

---

## 5. 验证记录汇总

- `scout/feat_cache.py --smoke`：全绿。
- `verify_bank`：随机锚点抽检通过（atol 5e-3，依据见 §3）。
- VIB 快版：300 epoch 完整真实跑完（非外推），train/val loss 正常。
- DP 计时测试：20 epoch 抽样 × 外推；`logging.mode=offline` 不写线上 wandb。
- 回归：`rollout_vec` smoke、dry-run 等原有测试未受影响。

## 6. 磁盘警示（下一轮前必须处理）

一轮 DP 训练写 ~143GB（31 ckpt × 4.6GB）；共享盘配额紧张（本轮工作开始时仅剩 ~8.6GB）→ 下一轮启动前要么清理旧 ckpt，要么 `training.checkpoint_every=100`（~28GB/轮）。

## 7. 改动文件清单与状态

提速相关（本地与服务器同步、**未 git commit**）：
- `scout/feat_cache.py`（新）
- `scout/model/encoder.py`、`scout/model/scout_vib.py`、`scout/train_vib.py`
- `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`
- `train.py`
- `configs/vib_can_image_fast.yaml`、`configs/vib_square_image_fast.yaml`（新）
- `soe_scripts/fast_round_local/`：`fast_round_can.sh`、`fast_round_square.sh`（串行一轮驱动脚本）、`bench_rollout_components.py`（未跑通，见 §4）

遗留事项：
1. `bench_rollout_components.py` import fallback 问题未解决（square rollout 提速探索中断）；
2. fast 配置的 `es_base_dp_ckpt` 需改回正式 ckpt（见 §3）；
3. 全部改动未 commit。
