# WalkCloud(去噪步 x̂₀ 累积云)round0 三任务探针

2026-09-10,分支 `drift-dev`(4133cca 实现 + 1f4d78f 探针变体 + 1e896c9 链启动器),
port1024 `/tmp/scout-drift`(易失;本目录已持久化 ledger + probe log)。

## 协议

- 与 drift campaign 逐字同协议:rescue×5(pass@5)、失败集冻结、P 分片 × env25、
  η3.0/κ2.5/gst100;两臂唯一差异 = `--guide atypical` vs `--guide walkcloud`
  (adapt τ̃=clamp(0.3·行池极差, 0.02, 0.5),hist_max=0 全累积)。
- walkcloud 零 RNG 且 t=2 逐位 aty → 两臂同 RNG 流 = 配对 A/B(非 cloudrep 的统计对比)。
- 失败集零派生:can 43(SR .57)/square 63(SR .37)复用 drift 缓存;toolhang 69
  (SR .31)取自 TH9-4 round0 r1 DP-base eval 的 failed.json。
- 窗口:can 全 43 一轮;square 32+31 两轮;toolhang 48+21 两轮(45min/臂 backstop)。
- 每臂 45min backstop(用户令);**toolhang r1 两臂均撞 backstop(rc=124),该轮
  分片账为下界**,两臂同等截断。

## 硬账(rescued@5,分片 grep 求和;aty 对照 = 配对同跑,非历史数)

| 任务(失败集) | aty | wc | 判定 | wc 遥测 mean_S / mean_inject(aty) | wc jerk(aty) |
|---|---|---|---|---|---|
| can 43(全) | **14** | 10 | aty 胜 | −1.21 / 0.57(1.06) | 0.125(0.381) |
| square 63(r1+r2) | **23**(12+11) | 18(7+11) | aty 胜 | −1.23/−1.40 / 0.40、0.68(1.56、1.65) | 0.118、0.145(0.278、0.263) |
| toolhang 69(r1 部分+r2) | **18**(9+9) | 5(2+3) | aty 大胜 | +5.27/+5.21 / **10.7、5.2(0.20、0.12)** | 0.199(0.079) |

**对照臂有效性**:aty can=14、square=23 与 drift campaign 基线逐数一致 ✓。

## 判读(预注册证伪线触发)

1. **can wc=10 ≤ 14 且 mean_S 正常(−1.21,非退化)→ i 轴证伪正式扩展到核聚合
   形式,按 walkcloud_plan §5 停**。square(18<23)、toolhang(5≪18)同向。
   GAElike(γ^k 求和)与 WalkCloud(soft-min 核)两种 i 轴聚合先后证伪——
   「去噪迭代间反重复」这一方向在 rescue 语境下不产生价值。
2. **剂量不可通约是最大混杂**(审查 P2-4 预警,三任务三种表现):
   - can/square:wc mean_inject 比 aty 低 2-4×(soft-min 重归一化,S≤min_j KL_j)=欠剂量;
   - toolhang:反向爆炸,wc mean_inject 10.7/5.2 vs aty 0.20/0.12(≈50×/43×过剂量),
     mean_S≈+5.2≫κ=2.5(th 的 VIB 梯度尺度大,η=3.0 的 S 单位与 KL 单位不同构),
     jerk 翻倍、救回崩到 5。共享原始 η 的 A/B 测的是「η3.0 下的 walkcloud」
     而非「位移匹配的 walkcloud」;若要翻案唯一干净路径 = eta_dimless 位移匹配重跑。
3. **机制签名**:mean_S 在 can/square 恒为负(候选陷在自身云内,云=wolk 自己的
   收敛轨迹,排斥力与去噪收敛对抗)= plan §6 风险 2 的实测印证;jerk 三任务全低
   (欠剂量侧的软轨迹)。
4. 产物:本目录 3×ledger.csv + 5×probe.log;/tmp/scout-drift 原始 hdf5 未拷(易失,
   判决已定无需复评)。

## 结论

**三任务全负,预注册证伪线触发,campaign 停在此处。** i 轴(去噪迭代内聚合)在
求和形式(GAElike)与核形式(WalkCloud)下均不优于单锚 atypical;唯一保留的
翻案条件是 eta_dimless 剂量匹配重跑(未做,须用户令)。
