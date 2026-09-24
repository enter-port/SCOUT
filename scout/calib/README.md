# η / κ 标定

算法统一放在 `SCOUT/scout/calib/`，从仓库根目录用 `python -m scout.calib.<模块>` 调用。
这里的 `κ` 对应代码参数 `kappa`；用户此前所说的“KL-mean”在新方法中明确指 **KL 中位数（median）**。
原来的 C 标定仍使用 KL 均值，二者分别保留。

## 结构与算法

```text
scout/calib/
├── __init__.py
├── core.py       # 固定 core 观测、模型加载、R/C 测量、GPU 检查
├── search.py     # 有界 κ 搜索；括区间与对数二分，只返回测过的点
├── eta_r.py      # 原 η 标定：固定 κ，按 R 比例更新 η
├── kappa_c.py    # 原 κ 标定：固定 η，使 C 接近固定 base 的 C
├── joint_pr.py   # P6：η=P/κ，在 core 上求 R≈0.01；保留研究用 C/R 模式
├── dp_kl.py      # 独立 DP 样本 KL 中位数 → κ，再固定 κ 标定 η
└── README.md
```

| 入口 | 输入/约束 | 标定规则 | 默认测量预算与失败行为 |
|---|---|---|---|
| `eta_r` | 当前 DP/dyn、上一轮 η/κ | `η ← η × target / R`，κ 固定 | 初始测量后最多 3 次更新；初始已在区间也更新一次。未收敛仍保存最后值并成功退出，保留旧语义 |
| `kappa_c` | base DP/dyn/η/κ、当前 DP/dyn/η、κ 起点 | `κ ← κ × C_base / C_current` | 默认最多 3 次当前模型测量；另测一次 base。`--require-converged` 才会在未收敛时非零退出 |
| `joint_pr` | 当前 DP/dyn；`P=6`、`R=0.01` | 每个 κ 点设 `η=P/κ`，求 R 进入目标带 | 最多 16 个 κ 点；未收敛保存诊断并非零退出 |
| `dp_kl` | 当前 DP/dyn；η 起点 | DP KL 的 pooled median 给出 κ；再按 R 比例调整 η | 8 次无 guidance DP 采样 + 最多 9 次 η 测量；初始已达标即停。未收敛保存诊断并非零退出 |

这些入口只做 core 标定。任务名支持 `can`、`square`、`coffee`、`threading`、`tool_hang`。
P6 的近似共同经验仅来自 can/threading/coffee 的已测 base，不能据此保证跨轮。
KL 中位数也仍是有任务适用范围的经验方法。实证记录见
[已有结论](../../idea/kappa_pr_base_calibration.md) 和
[实验记录](../../experiments/2026_09_24_kappa_calibration/RESULTS.md)。

## 测量定义

- **core batch**：demo 名字按字典序排列，步长 `max(1, 总帧数 // (B×4))`，取各 demo 的 `t=1,...,n-2`，直到 B 个观测。每个观测堆叠 `(t-1,t)` 两帧；默认 `B=128`。图像转 CHW 并除以 255，进入 VIB adapter 时还原尺度。
- **R**：从 guidance 开始，逐步计算 `η × noise_scale × mean(abs(capped_gradient))`，再对步取均值，除以整个 core 的 `mean(abs(raw abs_actions))`。分母不是 noisy action；`R_noisy_mean` 只是额外诊断字段。默认目标区间 `[0.009,0.011]`。
- **C**：guided 轨迹上所有步、所有样本的 **uncapped KL 均值**。旧 C 保留“每步先 Torch 均值，再对步取 NumPy 均值”的浮点归约顺序。`base_kappa` 固定 base 参考点，`kappa0` 只是当前轮起点。
- 所有方法使用 `KLCostPlanner`、`eta_dimless=False`；每次 R/C 测量前 `torch.manual_seed(0)`，保持噪声配对。

**KL 中位数的精确定义**：在同一组 128 个观测上，以 seed 0–7 各运行一次 η=0 的 DP 采样。
每个样本记录最后一次 clean estimate 的 posterior `q_final`，以及 guidance 首步所建 anchor 的 posterior `q_anchor`。
对每个观测 s、每个有序样本对 `i≠j` 计算：

```text
KL_s,i,j = KL(q_final(s, draw_i) || q_anchor(s, draw_j))
κ = median({KL_s,i,j : s=1..128, i,j=0..7, i≠j})
```

默认共有 `128×8×7=7168` 个值。KL 沿 latent 维求和，按上述方向计算；不混合不同观测，
不包含同一次 draw 的配对。直接对全部值取中位数，不先做每个观测的均值。
代码用 float64 的对角高斯 KL 公式计算。`same_draw_final_to_anchor`、`final_to_final`、
`median_state_mean` 均为诊断，κ 只取 `independent_final_to_anchor.quantiles.p50`。

## 直接使用

以下为服务器 Bash 示例，先在目标端口查看 `nvidia-smi` 确认卡空闲；1022 的 GPU7 不使用。
在仓库根目录和相应任务的 Python 环境执行，checkpoint/core 路径换成自己的。
P6 和中位数入口在加载模型前检查物理 GPU 编号、UUID、显存占用，并拒绝已知故障卡。
原 η/C 入口沿用 `CUDA_VISIBLE_DEVICES` 的调用方式。

```bash
PY=python
TASK=can
GPU=0                      # 换成已确认空闲的物理编号
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
DP_CKPT=/path/to/current_dp.ckpt
DYN_CKPT=/path/to/current_scout_vib.ckpt
CORE=/path/to/core.hdf5
OUT_DIR=/path/to/your/calibration
mkdir -p "$OUT_DIR"
```

### 1. 原 η 标定

`--eta-prev`、`--kappa-prev` 使用上一轮实际值，base 时使用指定起点。下列数值仅示范参数格式：

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scout.calib.eta_r \
  --eval-config "configs/eval_${TASK}_entropy.yaml" \
  --dp-ckpt "$DP_CKPT" --vib-ckpt "$DYN_CKPT" --core-hdf5 "$CORE" \
  --eta-prev 2 --kappa-prev 2.5 \
  --target 0.01 --band-lo 0.009 --band-hi 0.011 --max-repeat 2 \
  --out "$OUT_DIR/eta.json"
```

输出主要字段：`eta,kappa,R_mean,converged,n_updates,history`。

### 2. 原 κ 的 C 标定

在 η 标定后执行。`BASE_ETA` 要取 base 自己已标定的 η，`--base-kappa` 是其对应 κ。

```bash
BASE_DP=/path/to/base_dp.ckpt
BASE_DYN=/path/to/base_scout_vib.ckpt
BASE_ETA=2                  # 换成该 base 的实际 η
ROUND_ETA=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["eta"])' "$OUT_DIR/eta.json")
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scout.calib.kappa_c \
  --eval-config "configs/eval_${TASK}_entropy.yaml" --core-hdf5 "$CORE" \
  --base-dp-ckpt "$BASE_DP" --base-vib-ckpt "$BASE_DYN" \
  --base-eta "$BASE_ETA" --base-kappa 2.5 \
  --round-dp-ckpt "$DP_CKPT" --round-vib-ckpt "$DYN_CKPT" \
  --round-eta "$ROUND_ETA" --kappa0 2.5 \
  --solver ratio --max-repeat 2 --band 0.1 \
  --out "$OUT_DIR/kappa_c.json"
```

可加 `--require-converged`，在失败时阻止后续流程。可用
`--solver bracket --max-probes 12 --kappa-min 0.001 --kappa-max 100` 替代比例更新。
输出主要字段：`eta,kappa,C_target,C_target_base,C_mean,converged,history`。
顺序 R→C 方法在第二步固定 η 改变 κ，最终 R 可能偏离目标。

### 3. P6 联合标定

```bash
"$PY" -m scout.calib.joint_pr \
  --task "$TASK" --eval-config "configs/eval_${TASK}_entropy.yaml" \
  --dp-ckpt "$DP_CKPT" --vib-ckpt "$DYN_CKPT" --core-hdf5 "$CORE" \
  --potential-cap 6 --target-r 0.01 --band 0.1 --initial-kappa 2.5 \
  --gpu "$GPU" --gpu-uuid "$GPU_UUID" --out "$OUT_DIR/p6.json"
```

默认 κ 范围 `[0.001,100]`，最多 16 个点。输出主要字段：
`eta,kappa,R_mean,R_converged,potential_cap,potential_target,history,termination_reason`。
P6 同时给出 η/κ；将这对参数直接用于 rollout。

保留原研究参数：用 `--c-reference <base测量JSON>` 替代 `--potential-cap`，可运行联合 C/R 标定；
加 `--sequential-c` 则先在初始 κ 匹配 R，再固定 η 匹配 C。`--eta-initial` 和
`--eta-max-probes` 只用于这些 C 模式。联合 C/R 内层最多 12 次 η 测量，区别于上述原 η 入口。

### 4. DP KL 中位数

```bash
"$PY" -m scout.calib.dp_kl \
  --task "$TASK" --eval-config "configs/eval_${TASK}_entropy.yaml" \
  --dp-ckpt "$DP_CKPT" --vib-ckpt "$DYN_CKPT" --core-hdf5 "$CORE" \
  --eta-initial 2 --samples 8 --target-r 0.01 \
  --gpu "$GPU" --gpu-uuid "$GPU_UUID" --out "$OUT_DIR/kl_median.json"
```

`--eta-initial` 是 η 求解起点，可取上一轮 η；κ 由当前 DP/dyn 的样本重新确定。
默认配置复现已研究的方法；改变 `--samples` 会改变 κ 的估计。
输出包括最终 `eta,kappa,R_mean,R_converged`、采样统计来源和完整 η 历史。
如果耗尽预算仍不满足 R 区间，会保存失败结果并非零退出。

直接模式产物：

```text
kl_median.json
kl_median.artifacts/
├── dp_diversity/summary.json    # 包含用于 κ 的 p50
├── dp_diversity/samples.npz     # posterior 与所有 KL 配对值
├── kl_median_reference.json
└── diagnostics_rmatched/k<κ>.json + k<κ>.npz
```

P6/中位数入口拒绝覆盖已有 `--out`；中位数直接模式也拒绝复用已有 `.artifacts` 目录。
原 η/C 入口保持旧的输出覆盖行为。四个入口均可用 `--help` 查看参数，无需安装模型依赖即可打印帮助。

## 与现有 round 的连接

```text
scripts/coffee/cfk5_chain.sh        scripts/threading/thm3_chain.sh
              ↓                                  ↓
scripts/coffee/round_cfk5.sh        scripts/threading/round_thm3.sh
                      ATY 标定阶段
             ┌────────────┴────────────┐
      CALIB_MODE=rc（默认）       CALIB_MODE=pr
      scout.calib.eta_r          scripts/calibration/calibrate_pr.sh
                ↓                         ↓
      scout.calib.kappa_c        scout.calib.joint_pr → search
             └────────────┬────────────┘
                  ATY_SCALE=η, ATT_CAP=κ
                          ↓
       GEXTRA: --guidance-scale η --atypical-cap κ
                          ↓
              shard_rollout.sh → run_rollout.py
```

例如在准备好自己的实验目录后：

```bash
CALIB_MODE=pr CALIB_P=6 CALIB_R=0.01 CALIB_BAND=0.1 CALIB_KAPPA0=2.5 \
  GPU="$GPU" TSEED=233 DATA_ROOT=/path/to/your/prepared/campaign \
  bash scripts/coffee/round_cfk5.sh coffee ATY 1 full
```

这会继续执行完整 round；只标定请用上面的 Python 命令。
Threading 使用 `scripts/threading/round_thm3.sh threading ATY 1 full`。
链入口使用 `SEED`、`ARM`，会向 round 导出 CALIB 变量；`BASE_ETA` 在 ATY 的 RC 模式需要提供。
PR 在 rollout 前校验实际 P/R 并同时回填两参数，失败停止该轮。
`DRY_RUN=1` 只打印计划，不加载模型。输出名仍为
`calib_pr_r<N>.json`，或 RC 的 `calib_rcalib_r<N>.json` / `calib_kcal_r<N>.json`。

`dp_kl` 目前为独立入口，没有新增 `CALIB_MODE` 或自动切换“base P6、后续中位数”。
将其 JSON 中的 η/κ 一起用于 `--guide atypical --guidance-scale ... --atypical-cap ...`。
历史 Can 的无量纲 η/orbit 入口使用不同剂量定义，不能直接替换成这里的 η。

## 旧入口与研究目录

| 原路径 | 现在的调用 |
|---|---|
| `scripts/coffee/cfk_rcalib.py` | 薄封装 → `scout.calib.eta_r.main`，原 CLI 不变 |
| `scripts/threading/thm2_c_calib.py` | 薄封装 → `scout.calib.kappa_c.main`，原 CLI 不变 |
| `scripts/calibration/kappa_pcalib.py` | 薄封装 → `scout.calib.joint_pr`，原 CLI 不变 |
| `scripts/calibration/kappa_search.py` | 转导出 `scout.calib.search.solve_kappa` |
| `scripts/calibration/kappa_diagnostics.py` | 保留批量诊断编排；采样、R 标定调用 `core` / `dp_kl` |
| `scripts/calibration/kappa_diversity_eval.py` | 保留“标定后可选评估”的编排；算法调用 `scout.calib.dp_kl` |

评估、报告、checkpoint 探针等研究脚本留在 `scripts/calibration/`。
`joint_pr`、`dp_kl` 都支持 `--source experiments/<研究>/<task>`，与直接 checkpoint 参数互斥；
输出需位于该 source 内。source 包含 `checkpoints.json` 或已有的 `diagnostics/natural.json`，
至少含 `task,dp_ckpt,vib_ckpt,core_hdf5`。
中位数模式的初始 η 从 `eta_initial` 或 `eta.json` 读取，也可显式传 `--eta-initial`。
source 模式会校验 checkpoint/core 身份后复用已有测量。

```bash
"$PY" -m scout.calib.dp_kl --source experiments/<研究>/can \
  --gpu "$GPU" --gpu-uuid "$GPU_UUID" \
  --out experiments/<研究>/can/calib_dp_kl.json
```

研究模式保留 `dp_diversity/`、`kl_median_reference.json`、`diagnostics_rmatched/` 产物路径。
旧 `kappa_diversity_eval.py --source ... --gpu ... --gpu-uuid ... --calibrate-only` 仍只标定；
去掉 `--calibrate-only` 会继续调用原评估器，需要已有初始评估及失败集。
迁移后该编排器还写一份 `source/calib_dp_kl.json` 供评估器使用。

## 本次迁移验证

运行 `python -m unittest discover -s tests -v`：覆盖旧参数入口、迭代与失败语义、固定观测、
R/C 测量、KL 方向和配对、P/R 搜索及真实 round 标定分支的参数传递。
另外用已有 13 组 base/round1/round2 posterior 存档重新计算 KL 与中位数，
与保存值在 `rtol=atol=1e-12` 内一致（全部数组最大绝对差约 `2.3e-13`）。
本地没有 PyTorch；未重新运行真实 checkpoint 的 GPU 采样或环境 rollout。
本次为本地迁移，未向服务器部署。
