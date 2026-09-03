# 真机阶段任务调研(2026-08-28)

> 目的:进入阶段 4(真机)前,调研"别人做过 + 能凸显我们 exploration 优势"的任务,≥10 个,按推荐度排序。
> 结论速览:§2 排序表;§3 给出建议的实验组合与里程碑顺序。

---

## 0. 筛选标准:从我们的优势面反推

我们的方法 = 冻结 base DP + VIB 动力学模型 + entropy cost 引导(guidance = 相对 DP 自身意图后验的 atypical 定向偏移,κ=2.5 封顶信任域)。仿真 campaign 已经确立的优势面,决定了任务必须满足什么条件才能"凸显"它们:

| # | 仿真已确立的优势 | 真机任务侧的对应判据 |
|---|---|---|
| 1 | **rescue 救回 / pass@10**(对 base DP 失败的初始态、同初始态重试 10 次,引导式重试把"至少成功一次"的比例大幅抬升:can +10/+9/+9、square +10/+13,pass@10 逐轮 12/12 领先) | 任务 base 成功率处于**中等带(约 0.3–0.7)**,且**初始态异质**(不同摆位难度分化)——饱和任务(lift 型)和几乎不可行任务都没有 rescue 空间 |
| 2 | **定向锥 > 随机噪声 / 喷雾**(entropy cost 把动作往"DP 自己不会做的方向"推,比随机扰动和 SOE 喷雾救回效率更高) | 任务存在**DP 习惯性失败模式**(固定 approach 方向/高度、插孔错位顶推、挂钩偏差)——接触类精度任务最典型 |
| 3 | **真机安全性**(冻结 DP + κ 封顶 + jerk 可控 0.2–0.3,动作永远在 DP 流形附近;不需要在硬件上跑 RL 梯度) | 任务**接触敏感/易碰坏**,以凸显我们 vs RL fine-tuning(SERL/HIL-SERL/DPPO)的差异化卖点 |
| 4 | **管线匹配**(84×84 双摄 agentview+wrist、7D abs EE 控制、horizon 16 chunk、二值成功信号驱动 rescue 记账) | 桌面级、单臂、**成功判定可自动化**(位姿/接触/小型分类器,SERL 的 reward classifier 是现成先例) |
| 5 | **multi-round 自提升闭环**(每轮 explore→回灌→retrain,数百次 rollout) | **reset 便宜、耗材耐用便宜**(每轮 100 场景 × 多轮,人工 reset 要快;不能用易碎/贵重物) |

**一句话**:理想任务 = "base DP 能做但会以固定习惯方式失败在一部分初始态上的、便宜耐用、成功判定自动化的接触型桌面任务"。

---

## 1. 对标工作的真机任务全景

| 工作 | 真机任务 | 平台 | 关键数字 |
|---|---|---|---|
| **SOE**(arXiv 2509.19292,学长) | mug hanging(挂杯)、toaster loading(放吐司机)、lamp capping(盖灯罩) | 机械臂 + Robotiq 夹爪 | 真机 3 任务,成功率/平滑度优于对比探索方法 |
| **SIME**(arXiv 2505.01396,同班底) | cup stacking(纸杯套金属杯,杯位每轮随机) | Flexiv Rizon + Robotiq 2F-85(软指)+ 双 D435i(腕+侧) | 50 demos;π0 **0.34** → naive 自提升 0.48 → SIME **0.74** |
| **LPB**(arXiv 2508.05941,NeurIPS'25,我们的代码底座) | **ToolHang** 真机 | 单臂 | 真机 ToolHang;仿真 Push-T/Square/Transport/ToolHang |
| **SERL**(arXiv 2401.16013) | **USB insertion**、**cube flip**、**clip hanging** | Franka | RL 25–50 分钟,自动 reset + reward classifier |
| **HIL-SERL**(arXiv 2410.21845,Science Robotics) | object repositioning、laptop lid opening、cable routing 等 | Franka | 1–2 小时近满分 |
| **DPPO**(arXiv 2409.00588,ICLR'25 Oral) | 真机 long-horizon **peg insertion**(sim2real zero-shot);仿真含 robomimic **Square/Transport** + Furniture-Bench | 单臂 | 仿真 Square 57%→97% |
| **RISE**(arXiv 2602.11075,RSS'26,world-model 自提升) | **dynamic brick sorting**(+35%)、**backpack packing**(+45%)等 3 个真机任务 | 单臂/双臂 | 与 IL/RL 基线全面对比 |
| **V-GPS**(arXiv 2410.13816,CoRL'24) | WidowX 6 任务:put **pepper in pot**、put **mushroom on cloth**、put **sushi in pot** 等 | WidowX + Octo | 冻结 base + 测试期 value 重排 → 成功率 ×2~×3 |
| **ACT/ALOHA**(arXiv 2304.13705) | Thread Velcro(**20%**)、Prep Tape(64%)、Open Cup(84%)、Slot Battery(96%)等 6 任务 | 双臂 ALOHA | 插入相位是瓶颈(Insert 20%) |
| **Diffusion Policy**(RSS'23) | **Push-T** 等 15 任务 | 多平台 | Push-T 成为事实标准 |
| **REBOOT**(CoRL'23) | 接触丰富真机技能(wiping、sorting 类,自动 reset) | 单臂 | 数据复用 2× 样本效率 |

---

## 2. 推荐排序(12 个任务)

排序权重:优势凸显度 40% + 可比性 25% + 工程可行性 25% + 叙事价值 10%。

### 🥇 1. Mug hanging(挂马克杯到挂钩架)— SOE 真机原任务
- **谁做过**:SOE(arXiv 2509.19292),真机三任务之一。
- **为什么凸显我们**:①挂钩是"定向锥"的教科书场景——base DP 习惯从固定方向/高度靠近挂钩,差几厘米就挂空;entropy cost 的 atypical 定向偏移正中这个失败模式。②与学长 SOE **同任务同协议直接同台**,我们已有 `SOE_scripts_2` 全套协议对齐基建;在 SOE 的主场上按 SOE 口径(rescue×10)赢 SOE,是论文里最硬的一张表。③成功判定=杯柄挂上杆(位姿+夹爪释放),可自动化。
- **工程注意**:用 demo 数量把 base SR 压到 0.4–0.6 的甜点带(SOE 的 demo 协议可直接缩放);挂钩架固定,reset 快。

### 🥈 2. Cup stacking(纸杯套入金属杯)— SIME 真机原任务
- **谁做过**:SIME(arXiv 2505.01396),Flexiv+Robotiq 软指+双 D435i(腕+侧),与我们双摄布局一致。
- **为什么凸显我们**:①文献基值 π0=**0.34** 恰在甜点带,rescue 空间文献背书(SIME 从 0.34→0.74);②杯位每轮随机=初始态异质,完全映射我们 can/square 的实验设计;③成功判定=杯在杯内,最简单;纸杯/金属杯便宜耐用,reset 秒级;④同班底可比性仅次于 SOE 本尊。
- **工程注意**:几乎无风险,是最稳的"第一战";注意 SIME 用了软指,我们若用普通平行夹爪需重调 demo。

### 🥉 3. USB / connector insertion(USB 插入)— SERL / HIL-SERL 真机
- **谁做过**:SERL(arXiv 2401.16013)、HIL-SERL(arXiv 2410.21845),Franka,SERL 系最出名任务。
- **为什么凸显我们**:①"DP 习惯性错位顶推"是 insertion 的默认失败模式,也是"引导策略做 DP 自己不会做的动作"最物理直观的展示;②单次成功率天然不高、pass@10 叙事最强(重试空间大且同初态可比);③与 RL fine-tuning 对比 = 我们的差异化:SERL 要在硬件上跑 25–50 分钟 RL 梯度,我们冻结 DP + 轻量 VIB + κ 信任域,零硬件梯度、动作有界——接触敏感场景下这是安全性与样本效率双重卖点;④USB 便宜、插入成功判定干净(深度/接触状态)。
- **工程注意**:需要 EE 位置控制精度够(位置控制模式);插错方向可能怼弯针脚→限力/限行程保护。

### 4. Push-T — Diffusion Policy 真机经典 / LPB 仿真
- **谁做过**:Diffusion Policy(RSS'23)真机;LPB、无数后续工作。
- **为什么凸显我们**:①T 块初始摆位连续异质→"失败集中在一部分初始态"与我们的 rescue 口径完美映射,评测协议(coverage 打分)现成;②便宜、安全、无抓取,是**闭环工程热身**最佳任务(先在 Push-T 上把"真机 explore→回灌→retrain→再 eval"整条链跑顺,再上接触任务);③人人都做过=审稿人零解释成本。
- **工程注意**:故事新意低,不能只靠它;成功判定用顶点覆盖(分割或 ArUco 贴纸)。

### 5. Toaster loading(吐司机放料)— SOE 真机原任务
- **谁做过**:SOE 真机三任务之二。
- **为什么凸显我们**:推入式 = insertion 家族,槽位窄、习惯性对不准;与 mug hanging 打包成"SOE 三任务组"一起同台,成本低(同一套平台/reset/判定基建)。
- **工程注意**:与 mug hanging 共享绝大多数工程;可作为任务组的第二点。

### 6. ToolHang(真机挂工具)— LPB 真机原任务【旗舰/stretch】
- **谁做过**:LPB(arXiv 2508.05941)真机;robomimic ToolHang 是五任务最难(SIME image π0=0.144)。
- **为什么凸显我们**:①我们整套代码就是 LPB 对齐架构(base DP/workspace/guidance 注入全是 LPB 范式),**工程延续性最强**;②"把仿真最难的 robomimic 任务搬上真机并自提升"含金量最高,且 ToolHang 有把手-挂钩几何=习惯性失败模式明确。
- **工程注意**:风险=base SR 可能低于 0.2(绝对数字难看)且几何工程量大(工具/挂钩夹具);建议作为后期旗舰任务,不是首批。

### 7. Clip hanging(夹子挂杆)— SERL 真机
- **谁做过**:SERL 三任务之一(有自动 reset 与 reward classifier 现成实现)。
- **为什么凸显我们**:挂钩家族简化版,同 mug hanging 的优势逻辑;SERL 开源实现可参考 reset 与判定;适合作为 hanging 家族的快糙热身或消融补充任务。

### 8. Thread Velcro(穿魔术贴扎带)— ACT/ALOHA 真机
- **谁做过**:ACT(arXiv 2304.13705),双臂 ALOHA,base **20%**(Insert 相位是瓶颈,阶段分解:Lift 92%→Grasp 40%→Insert 20%)。
- **为什么凸显我们**:①base 20%=全表最大 rescue 空间,"pass@10 从 0.2 抬到 0.6+"是冲击性数字;②小接触面=对 jerk/安全关注度高,凸显我们动作有界。
- **工程注意**:先例是双臂,我们若单臂需把任务改造(锚点固定 loop)→严格意义上不再是"别人做过的同一任务",可比性打折;难度高,建议中后期。

### 9. Drawer / cabinet opening(开抽屉/柜门)— HIL-SERL 家族 / REBOOT 系
- **谁做过**:HIL-SERL(laptop lid 等)、REBOOT 系自动 reset 先例。
- **为什么凸显我们**:①力接触任务,DP 习惯性失败=拉错方向/手柄打滑;②成功判定=抽屉位移,极简;reset=推回,便宜;单臂可行。
- **工程注意**:初始态异质性有限(抽屉状态空间小),rescue 故事比 1–3 弱;适合作为覆盖面补充。

### 10. Put-X-in-pot(X 放入锅/容器:pepper/mushroom/sushi)— V-GPS 真机 + robomimic Can 的真机近亲
- **谁做过**:V-GPS(CoRL'24)WidowX 6 任务,冻结 base + 测试期引导成功率 ×2~×3——**"测试期引导抬成功率"的直接先例**。
- **为什么凸显我们**:①与我们仿真 can(pick-place)直接呼应,"sim can → real can"叙事连续;②V-GPS 证明了"不重训、测试期引导"这个大方向有市场,我们换上更便宜的引导信号(VIB entropy cost,不需要训 value function)可以直接对标它;③成功判定=物在容器内,最简;耗材耐用。
- **工程注意**:任务本体平凡,单做不亮;要加初始位姿扰动/遮挡才有 rescue 空间;适合作为与 V-GPS 对比的窗口任务。

### 11. Nut threading / 方螺母拧装(robomimic Square 的真机版)— ACT 家族 / Furniture-Bench
- **谁做过**:拧装类在 ACT(Prep Tape 含 threading 64%)、Furniture-Bench(DPPO 仿真 Square 57%→97%)有先例。
- **为什么凸显我们**:与我们 square 仿真结果(+10/+13,窄锥探索的主场结论"定向锥>喷雾")直接呼应;螺纹接触=最典型的窄容差任务。
- **工程注意**:螺纹几何对硬件精度/力保护要求最高,reset 费时(要退回起点),单臂拧螺纹难;工程风险大,除非硬件条件好。

### 12. 动态分拣(brick sorting / backpack packing)— RISE 真机【备选】
- **谁做过**:RISE(OpenDriveLab,RSS'26,+35%/+45%)。
- **为什么凸显我们**:展示我们的方法在动态场景不炸(jerk 有界),并可蹭"vs world-model 自提升"的比较话题。
- **工程注意**:场景工程量大、判定复杂、我们的 chunk 式慢去噪在高速动态上天然劣势——**不推荐首批**,列此备查。

### 附:Cloth folding(DPPO 真机)— 不推荐
可形变物体:VIB next-latent 学习信号弱、成功判定/reset 贵、与 rescue 口径不匹配。DPPO 做过但那是 RL fine-tuning 的主场,不是我们的。

---

## 3. 建议组合与里程碑顺序

**推荐的三任务组合**(三类优势各占一个,兼顾可比性梯队):
1. **Cup stacking**(SOE 系同台,最稳,先跑通闭环)
2. **Mug hanging + Toaster loading**(SOE 三任务组打包,主战场,同协议同台)
3. **USB insertion**(接触精度,与 SERL/HIL-SERL 的 RL 路线正面对比,凸显"零硬件梯度、动作有界"卖点)

> **更新(2026-08-30)**:用户以插入段精度超配(±0.6° 旋转容差/毫米级横向,对 10Hz chunk 式 DP 管线不现实)为由否决 USB insertion;曾暂定 Toaster loading(SOE),后因用户"第三任务不选 SOE/SIME"约束改定 **Drawer opening 开抽屉**(AutoSERL 判据"位移 ≥5cm",HIL-SERL 铰链家族先例;容差厘米级)。**前两个任务 Cup stacking(SIME)与 Mug hanging(SOE)保持不变(用户 2026-08-30 确认定稿)**;规格见 `real_robot_task_details.md` v3。

**里程碑顺序**(按工程风险递增):
- **R1 热身**:Push-T 或 cup stacking——把"真机 rollout→rescue×10 记账→回灌 hdf5→retrain DP+dyn→再 eval"整条链跑顺,校准真机上的 base SR 到 0.3–0.7 甜点带。
- **R2 主战**:SOE 三任务(mug hanging 为首)——SOE 口径同台,目标=SR 与 pass@10 双赢 SOE(仿真已证明我们 pass@10 逐轮领先、终值 +3~+10)。
- **R3 旗舰**:USB insertion(或 ToolHang 真机)——最难接触任务,与 RL 系对比,论文差异化卖点。

**平台连续性提示**:SOE/SIME 用 Flexiv Rizon + Robotiq 2F-85 + 双 D435i(腕+固定侧),84×84 双摄与我们的管线完全对齐;若实验室为同款/同级平台,SOE 真机协议(场景播种、rescue×10、demo 数)可直接移植到我们的 `SOE_scripts_2` 框架。

---

### 主要来源
- SOE: [arXiv:2509.19292](https://arxiv.org/abs/2509.19292) / [项目页](https://ericjin2002.github.io/SOE/)
- SIME: [arXiv:2505.01396](https://arxiv.org/html/2505.01396v1)
- LPB: [arXiv:2508.05941](https://arxiv.org/abs/2508.05941) / [项目页](https://latentpolicybarrier.github.io/)
- SERL: [项目页](https://rail-berkeley.github.io/serl/) / [arXiv:2401.16013](https://arxiv.org/abs/2401.16013)
- HIL-SERL: [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)
- DPPO: [项目页](https://diffusion-ppo.github.io/) / [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)
- RISE: [arXiv:2602.11075](https://arxiv.org/html/2602.11075v1)
- V-GPS: [项目页](https://nakamotoo.github.io/V-GPS/) / [arXiv:2410.13816](https://arxiv.org/abs/2410.13816)
- ACT/ALOHA: [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- Diffusion Policy: [项目页](https://diffusion-policy.cs.columbia.edu/)
- REBOOT: [arXiv:2309.03322](https://arxiv.org/abs/2309.03322)
