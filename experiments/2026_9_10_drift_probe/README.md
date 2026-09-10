# cloudrep(方案一:drifting 云参照排斥)round0 A/B 探针

2026-09-10,分支 `drift-dev`(tip 见 git log),服务器 port1024 `/tmp/scout-drift`
(⚠️ CPFS 0 可用,探针整体跑在 pod 本地 1TB NVMe,产物易失——ledger/shard tally
已拷贝到本目录持久化;完整 hdf5 轨迹未拷)。

## 协议

- 资产:CAN-8-24-entropy-s233 / SQUARE-8-26-entropy-s233 各自 round0 三件套
  (DP-base 599.ckpt + dyn-base,square 与 orbchain 的 DP ckpt md5 相同已验证)。
- 失败集:can 43(SR .57)/ square 63(SR .37),缓存自 gae 探针根。
- A/B:同失败集、同 seed 流、每臂 P=4 CPU worker × env25、ETRIES=5(pass@5)、
  η3.0/κ2.5/gst100、45min/轮 backstop;两臂唯一差异 = `--guide atypical` vs
  `--guide cloudrep`(τ/agg 各代变体)。square 因 crep 慢 ~1.4-2× 走 16 场景
  窗口轮转。
- **读数纪律**:合并 explore.json 出现过坏值(r4 aty 合并报 0,分片 grep 实为
  14)——**权威 = 分片 json 的 exploration_rescued grep 求和**
  (shard_tally.csv)。A/B 为同失败集统计对比(起点门使两臂 RNG 流自分叉,
  非逐位配对)。

## 方法迭代

1. **iter-1 softmin(τ=0.5 fixed)**:S=−τ·logΣexp(−KL_j/τ) 于池{本 chunk 锚}∪
   {本场景历次重试 chunk-0 锚云};梯度=softmax(−KL_j/τ)(drifting 归一化核)。
   起点门(try_started_gate)保证 retry k 的 chunk-0 见 retries<k 的锚。
2. **iter-2 adapt**:τ̃ᵢ=clamp(ρ·spread_i, τ_min, τ₀=0.5) 逐行自适应
   (反思:绝对 τ 在 square 半幅 KL 尺度上等效大 τ → 权重平坦 J_eff 3.7-5.5 →
   j 轴力被平均稀释)。
3. **iter-3 add(GAElike add 教训)**:S = KL_anchor + λ·softmin_cloud,λ=0.15,
   锚点逃逸力全权保留,云排斥作纯加性扰动。

## 结果(rescued@5,分片 grep 硬账)

| 任务 | aty 基线 | crep iter-1 | crep iter-2(adapt) | crep iter-3(add) |
|---|---|---|---|---|
| can 43 失败 | 14 | 18 | **21** | r3 在跑 |
| square 63 失败 | 23(r1 9 + r4 14) | 9([0:32],未续) | 16 | **20** |

- **can:cloudrep 胜**(adapt 21 vs 14,+7;iter-1 的 McNemar 10/4/8,单侧 p=.19;
  iter-2 的 +7 更强)。pass@5 = .78 vs .71。
- **square:cloudrep 负**(最好 add 20 vs 23,−3;迭代轨迹 9→16→20 改善但未过线)。
  pass@5 = .53 vs .60。
- 机制读数:can(KL 尺度大)上云排斥是净收益;square(KL 压缩、aty 本身强)上
  任何形式的云项都净亏——j 轴反重复的价值是任务依赖的。
- jerk:crep 系全程 ≈ aty(can .365-.380 vs .381;square add .22-.29 vs .26)。

## 遗留

- 迭代硬上限(3 轮改进)已用尽,square 未超 → 按协议停在此处如实上报。
- 下一步候选(未做):square 专属剂量换算(η_sq ×2 按梯度尺度)、λ 扫描
  (0.05-0.3)、云构成扩展(执行 chunk 后验)。
- 服务器 /tmp 产物易失;要复评需重跑(脚本+分支都在)。
