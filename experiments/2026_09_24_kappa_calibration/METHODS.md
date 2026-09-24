# κ 标定方法与判定边界

本文记录探索过程中实际采用的方法。方法是在探索中依次提出的，未将这些
100 个评估场景视作完全未见过的统计检验集。所有结论限于这些 checkpoint、
场景和采样协议；额外训练 seed 的泛化尚未测试。

## 共同约束

- 初始评估为 seed 42–141 的100个场景，固定初始失败集。
- DP、SCOUT 各自做5次重试，4个 worker、每个25个环境；没有首次成功即停止。
- 代码中的总 pass@5 是初始成功数加失败集救回数后除以100。
- 主要判定看救回数：SCOUT > DP 为通过，SCOUT > 1.5 DP 为优秀。
  恰好1.5倍不算优秀。失败集不足1.5 DP时，优秀线不可达。
- 不把参数标定收敛当作任务成功；必须完成环境评估。
- 所有固定参考都来自 base，不随 round1 的测量继续漂移。
- core 标定采用现有校准器的128个观测、RNG seed0、同一动作分母。
  这些观测不是100个评估场景，标定时不读取环境成功结果。

## 1. P = ηκ 与 R 联合标定

若同一动作位置的成本满足 f_new=c f_base，则 κ_new=c κ_base、
η_new=η_base/c 恰好保持 η∇min(f,κ) 不变。因此定义 P=ηκ。
这只是对纯数值缩放的条件恒等式，不代表不同任务或重新训练后的成本形状相同。

给定P，通过 η=P/κ 在 core 上数值求解 R≈0.01。使用有界对数空间括区间和
二分法；没有环境成功率网格搜索。R或P允许10%相对误差，复用已有core点时
仅按P接近程度选择，不使用该点的rollout成功率。

共同 P≈6 的五任务base检验被tool_hang否定；coffee在base和round1通过，
round2未通过（9对DP11）。因此当前不能推荐P6作为稳定跨轮法。
Threading另外检验其优秀base点P=11.732053760076715，不能把它与P6混称。

## 2. 保留自然KL的base分位数

用不施加guidance的DP采样，仍以首个guided timestep的意图为KL锚点，
收集所有guided steps和core观测的KL，构成经验分布F。

q = F_base(κ_base)
κ_round = quantile(KL_natural_round, q)

之后在新κ处重新标定η，使R达标。只在0.05<q<0.95时使用，避免极端分位数。
本次can参考κ_base=2.5，q=.406328125，得到round1κ≈13.826、round2κ≈7.753。
Round1完成15对DP13，round2完成4对DP2；两轮均通过基本条件。

自然KL包含从锚点时刻到后续去噪时刻的DP漂移。Can/square的guidance start
为100，其他三任务为50；自然KL差异不能都解释为任务或dyn模型的单位差异。

## 3. 已有C标定与两种求解方式

原C为未截断KL的均值。原流程先调η再调κ，最终R可以改变；这不能单独视作错误。
另一个独立问题是，文档规定固定base κ=2.5，代码却让base参考κ随上一轮κ漂移。
本地已修复base参考问题，并增加可选的有界求解和失败拒绝。
生产服务器已有campaign脚本未被替换。

新增联合模式保留指定base优秀点的C，外层求κ，内层每次重新求η以满足R。
外层和内层都只读取core测量值。在给定边界和预算内找不到合格参数时，保存失败诊断并拒绝
使用该结果启动SCOUT评估。由于均值C受自然漂移和尾部影响，此法也需实测。

本次coffee固定优秀base点的C_target=9.847478866577148：
round1κ=5.452538663326289、η=.4923616104816283、C=10.2651、R=.00954219；
round2κ=3.8555270635198524、η=.6307592481161841、C=9.69385、R=.009784。
两轮环境评估分别为17对DP12、11对DP11；round2未超过DP。它们不等于旧脚本在固定η下的C标定。

运行入口是 scripts/calibration/kappa_pcalib.py 的 --c-reference，参数应指向
有C_mean和已验证R的固定base校准JSON。--potential-cap 模式仍保留。

顺序模式额外传 --sequential-c：先在上一轮κ处标定η，使R接近0.01，随后固定η
求κ以匹配base C。Round1从base κ和η开始；round2从round1仅用core求出的参数开始。
两轮C目标仍固定在base κ=2.5。此模式记录最终R，但不要求最终R仍为0.01；不能
将其与联合C/R模式混称。Coffee和threading先测round2，若通过再补round1环境评估。
该选择只决定实验顺序，不参与core参数求解。数值求根不是rollout grid search。
Coffee round2的顺序模式得到κ=1.146255、η=1.269378、C=4.014093（目标4.218468）、
最终R=.00810985；救回9/DP11，未通过。其round1因此不再补做环境评估。
Threading round2的顺序模式κ=2.973018、η=1.191946、C=9.435774（目标9.905537）、
最终R=.00986710；救回20/DP23，未通过，其round1也不再补做环境评估。

## 4. 独立DP样本的后验KL尺度

同一core观测运行8次独立无guidance的DP，保存每次最终预测动作及首个guided
时刻动作的dyn后验。主候选是不同样本之间KL(final_i || anchor_j), i≠j的
汇总中位数，用它直接给出κ，再标定η。没有根据rollout结果选分位数。

base中位数：can4.838144、square.921471、coffee.09548945、threading.010235、
tool_hang.00759784。已完成base结果为can18/DP8、square23/DP9、coffee26/DP16、
threading17/DP16、tool_hang33/DP35。Tool的base未通过，因此不能作为五任务通用规则。
Can的round1和round2分别为15/DP13、3/DP2，均通过基本条件；round2恰好1.5倍，
不算优秀。Coffee round2为9/DP11，也不能声称该规则普遍跨轮有效。
Threading的round1为18/DP17，round2为25/DP23，连同base的17/DP16均满足
基本条件，但没有达到优秀线，且只有1、1、2个场景的净增益。
core结果本身不能证明有效。
Seed0同一样本final-to-anchor的KL已与原natural探针对齐，核对通过。

### 只标定、不跑环境的入口

`scripts/calibration/kappa_diversity_eval.py --calibrate-only` 先计算8次DP采样的
KL中位数，再在该κ处标定η。每次需要加载模型前检查指定GPU的UUID及占用，
也检查缓存的DP、dyn、core身份。输出实际κ、η、R及JSON路径。

新checkpoint使用独立的 `experiments/<研究名>/<task>/checkpoints.json`，包含
`task, dp_ckpt, vib_ckpt, core_hdf5, eta_initial`；不要改写已有实验的身份文件。
从服务器repo根目录运行，GPU参数填入事先确认空闲的卡及其UUID：

```bash
/mnt/workspace/baojiachun/.venv_mg/bin/python scripts/calibration/kappa_diversity_eval.py \
  --source experiments/<研究名>/<task> \
  --gpu <空闲GPU编号> --gpu-uuid <已核实的GPU_UUID> --calibrate-only
```

该入口的缓存复用路径已在can round1验证，返回κ20.217892、η.707685、R.010293。
省略 `--calibrate-only` 时还会在已有冻结失败集上执行环境评估；仅core标定不要求
先跑100个评估场景。此入口没有扩大该方法在任务间的实证适用范围。

## 5. 动作敏感度比例（目前只有诊断）

在DP归一化动作坐标中施加固定RMS随机扰动，测 S=E[KL]/epsilon²。
主epsilon=.05，16个方向，另用.01和.1检查局部二次近似。
候选κ_round=κ_base*S_round/S_base尚未完成环境验证，不能算已成立或已失败。
不同状态上的敏感度比例差异明显。该测量使用各checkpoint的DP归一化动作坐标，
没有单独分离dyn变化与动作normalizer变化，不能直接归因为dyn自身的统一缩放。

## 6. 保持base κ的对照

为区分κ调整是否必要，补充coffee κ=2.5、tool_hang κ=1的固定候选，
只重新标定η。R目标分别为0.01和成功tool base点的.007156729252424086。
直接复用此前C求解过程中在这个确切κ处首次达到R目标的core测量，
不使用rollout结果挑选η，不引入新的κ搜索点。
Coffee先测round2，通过才补round1；tool两轮并行。
Coffee round2固定κ=2.5、η=1.106664、R=.0109254，救回9/DP11，未通过；
因此不再补其round1环境评估。
Tool_hang round2固定κ=1、η=8.298740、R=.00765968，救回14/DP37，未通过。
其round1固定κ=1、η=8.486101、R=.00733852，救回13/DP40，也未通过。
每份fixed_base_kappa.json保留原始core测量来源和SHA256。
Threading已有round2 κ=2.5、R≈.01053的joint_c结果19/DP23，因此相同
base κ/R条件已经有负对照，不另行重复。

## 7. κ=1的两阶段共同配方

初始base探针还有一项共同正结果：先在κ=2.5将η标定到R≈.01，再保持η、
改用κ=1。它在can/square/coffee/threading/tool的base分别救回10/19/19/19/41，
对应DP8/9/16/16/35，全部过基本线，但只有square优秀。
五个最终R分别约.003531/.003708/.007195/.007831/.007157，不是最终R=.01。
这项结果不能因R发生变化而被忽略，也不能混同于κ=1重新匹配R的另一组结果。

在复核总表时补充同一配方的跨轮检验：coffee round2固定κ=1，η使用之前在
κ=2.5达到R目标的1.1066641191457374。先测实际R，再用同一冻结失败集评估。
该配方的固定常数来自base探索；round2参数没有按rollout成功率选择。
Coffee round2的实际R=.007035926、C=2.791032，救回9/DP11，未通过。因此
该配方支持本次五任务base的基本条件，但不能推广为本批checkpoint的跨轮共同规则。


## Tool_hang较低R的固定任务参考

同一base κ=1，原固定η=2.6224248560499186对应R=.007156729252424086，
救回41对DP35；重新提高η到R≈.01后救回34对DP35。因此tool的固定任务P/R
验证采用前者的实际R，而非强行指定.01。其base P=2.6224248560499186。
两轮core标定均给出κ=.08838834764831845、η=29.66935038184085，R分别为
.0072407673572808424和.007404827016396008。环境结果为round1 30/DP40、
round2 20/DP37，均未过基本线。联合C/R的round1为21/DP40，round2为12/DP37，
也均未通过。


## 注入时间分布诊断

injection_temporal_diagnostics.json保留平均R之外的步骤分布信息。
unmasked_fraction表示KL没有超过κ的比例；锚点处梯度本身可以为零，所以
它不等于真正有非零梯度的比例。effective_step_fraction定义为
(sum_t u_t)^2 / (T * sum_t u_t^2)，其中u_t为该步骤在core batch上平均的
绝对注入。它是时间集中程度指标，不是实际非零步骤数量。

Coffee的κ=2.5和中位数法两点R均接近.01，但有效步骤比例约.907/.397，
峰值步骤注入/平均注入约1.58/7.35；对应救回23/26，DP16。
这些诊断说明R相近并不保证时间分布相同，不能据此单独断言因果或成功。

Rollout日志的`[guidance-telemetry] mean_inject`是对整批注入张量先求L2范数，
再按调用次数平均（`scout/guidance/policy.py`），不是core标定使用的逐元素平均
绝对值。批大小还可能随活跃环境数量变化。不能把这个日志值直接除以动作尺度
当成在线R，也不能仅凭它断言失败来自线上注入剂量放大；本轮未记录足以重建
同口径在线R的逐步统计。
