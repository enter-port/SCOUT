# 标定研究脚本与 round 适配层

标定算法已迁移至 [`scout/calib`](../../scout/calib/README.md)。该文档包含四种方法的定义、
直接 checkpoint 命令、输出及兼容入口。本目录保留研究编排、评估、报告与 round 的 shell 适配层。

本页描述2026-09-24接入的PR模式及现有RC模式。三任务base的实证结论见
`../../idea/kappa_pr_base_calibration.md`。代码接入不代表跨轮效果已经通过验证。

## 实验流程如何调用

```text
coffee/cfk5_chain.sh                 threading/thm3_chain.sh
          ↓                                    ↓
coffee/round_cfk5.sh                 threading/round_thm3.sh
          └──────── ATY arm 的标定阶段 ──────────┘
                         ↓ CALIB_MODE
        ┌────────────────┴─────────────────┐
        rc（默认，旧流程）                  pr（显式启用）
        scout.calib.eta_r                  calibration/calibrate_pr.sh
          固定κ，用R调η                     scout.calib.joint_pr
                    ↓                        η=P/κ，在core上求R≈目标
        scout.calib.kappa_c              ↓
          固定η，用C调κ                     scout.calib.search
        └────────────────┬─────────────────┘
                    ATY_SCALE=η，ATT_CAP=κ
                         ↓
        GEXTRA: --guidance-scale η --atypical-cap κ
                         ↓
        shard_rollout.sh → scout/eval/run_rollout.py
                         ↓
        scout/guidance/policy.py + entropy_costs.py
```

图中的 shell 路径相对 `scripts/`；`scout.calib.*` 用 `python -m` 调用。标定在组装GEXTRA前完成。
PR同时给出η和κ，不再执行后面的C标定；DP arm仍走guide=off。
历史`*_launch.sh`含旧路径、固定GPU和campaign配置，不作为新PR实验的启动入口。
新实验从已配置checkpoint/core的round driver或chain入口显式传入模式。

## 入口和文件职责

| 文件 | 职责 |
|---|---|
| `calibrate_pr.sh` | round driver共用适配层；取GPU UUID，调用PR求解器，校验实际P/R，回填η/κ |
| `kappa_pcalib.py` | 兼容封装，转调 `scout.calib.joint_pr` |
| `kappa_search.py` | 兼容导出 `scout.calib.search.solve_kappa` |
| `../coffee/cfk_rcalib.py` | 兼容封装，转调 `scout.calib.eta_r` |
| `../threading/thm2_c_calib.py` | 兼容封装，转调 `scout.calib.kappa_c` |
| `kappa_diversity_eval.py` | 调用 `scout.calib.dp_kl` 后可选评估；`--calibrate-only` 只标定 |
| `kappa_natural_transfer.py` | 自然KL分位数转移候选，不属于PR调用链 |
| `kappa_eval_arm.py` / `kappa_report.py` | 研究评估和配对审计，不是生产标定器 |

## 直接给 checkpoint 做 P6 标定

在仓库根目录运行，先确认GPU空闲。`--gpu`为物理编号；UUID与占用会在加载模型前核对，
已知故障卡UUID被拒绝。输出父目录须已存在；已有输出会拒绝覆盖。

```bash
python -m scout.calib.joint_pr \
  --task can --eval-config configs/eval_can_entropy.yaml \
  --dp-ckpt /path/to/DP-base.ckpt --vib-ckpt /path/to/scout_vib.ckpt \
  --core-hdf5 /path/to/can_core.hdf5 \
  --potential-cap 6 --target-r 0.01 --band 0.1 --initial-kappa 2.5 \
  --gpu <空闲物理编号> --gpu-uuid <对应UUID> --out /path/to/calib_pr.json
```

输出含`eta,kappa,R_mean,R_target,R_converged,potential_cap,potential_target,history`、
checkpoint/core身份和终止原因。不收敛会保存诊断并返回非零；不能拿失败点继续rollout。
将输出η/κ用于`--guide atypical --guidance-scale ... --atypical-cap ...`，
本方法使用`eta_dimless=False`。Can历史`can_aty_launch.sh → aty_chain.sh`使用
无量纲η和orbit特殊配置，尚未自动接入PR，不能直接把这里的η替换进那个旧入口。

原研究接口`--source experiments/<研究>/<task>`仍保留；它与直接checkpoint参数互斥。
研究模式可按P/R误差带复用core测量；直接checkpoint模式重新测量，不复用历史结果。

## 在已有round/chain中选择PR

在现有调用前加：

```bash
CALIB_MODE=pr CALIB_P=6 CALIB_R=0.01 CALIB_BAND=0.1 CALIB_KAPPA0=2.5 \
  GPU=<空闲卡> TSEED=233 DATA_ROOT=<已准备的新实验目录> \
  bash scripts/coffee/round_cfk5.sh coffee ATY 1 full
```

Threading对应`round_thm3.sh threading ATY 1 full`。Round1消费base checkpoint；
后续 round 消费重新训练的 checkpoint；已有跨轮结果不支持 P6 的普遍迁移，见算法文档与实验记录。
上述命令会继续执行完整round，不是只标定；只标定用前面的Python入口。
Chain入口使用`SEED`和`ARM`，并显式导出这些CALIB变量给round子进程。
`BASE_ETA`仅在ATY的RC模式必需，PR不依赖base C参考。`ROOT`默认从chain位置解析，
可显式覆盖。历史launch脚本没有改写。

PR每轮从`CALIB_KAPPA0`起步，不依赖旧RC参数文件。产物：

```text
$RDIR/calib_pr_r<N>.json   # 同一组η/κ及实测P/R、求解历史
$RDIR/calib_pr.stdout     # 模型加载与core求解日志
```

旧RC仍写`calib_rcalib_r<N>.json`及`calib_kcal_r<N>.json`。PR输出独立命名，
已存在时拒绝覆盖；重试前需检查该文件，使用新的实验目录或明确归档旧诊断。
`DRY_RUN=1`只打印计划，PR适配层不访问GPU、不加载模型。

## 验证与部署状态

`tests/test_pr_integration.py`执行两个真实round脚本中的标定分支，使用模拟模型调用
检查参数传递、求解失败拒绝、实测R校验和DRY_RUN；`tests/test_kappa_search.py`
覆盖非线性、不可达、跳跃响应与预算终止。本次接入只修改本地代码，未重启服务器
campaign、未部署覆盖其运行脚本，也没有为接入测试启动新的GPU实验。
