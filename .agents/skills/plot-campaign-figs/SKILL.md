---
name: plot-campaign-figs
description: 把 SCOUT campaign 的逐轮结果画成汇总图(SR / pass@K 轨迹、多 seed 均值±sd 色带)。只要用户说「画图 / 出图 / 画一下结果 / 画这几轮的结果 / 汇总图 / 对比图 / 终表图 / figs summary / plot the results」,或要求把某个 campaign 的 round 指标可视化,就用本 skill:先从服务器 rollout json 现拉权威读数(禁止抄记忆数字出图),本地 matplotlib 按已接受先例风格出图,judge 视觉验收后落 experiments/<date>_<campaign>_figs_summary/ 交付并落账。只画用户点名的指标,不加任何装饰元素。
---

# 画 campaign 汇总图(plot-campaign-figs)

目的:把「campaign 完赛 → 出汇总图」固化为可重复流程,风格与已交付并被用户接受的先例图一脉相承。

**风格先例(动笔前必读)**:
- `experiments/2026_9_11_th95_3seed_figs_summary/plot_metrics.py` — 多 seed 均值±sd 色带版(主模板)
- `experiments/2026_9_4_figs_summary/` — 单 seed 版

## Step 1 — 定位数据(先定位再动手)

1. campaign 名 → 持久 memory 的 campaign 文件拿 DATA_ROOT;没有就读服务器 `scout/data/<campaign>/PROVENANCE.md`。
2. 权威读数 = 服务器 `DATA_ROOT/<NAME-s<SEED>>/<task>/rollout/<ARM>-exp<N>/log/*.json`:
   - eval SR = eval json 的 `success_rate`(100 固定场景)
   - pass@K = explore json 的 `pass_at_5` 键——**键名历史沿用,实为 pass@{ETRIES}**,以该轮 config 的 explore_try_times 辨
   - 救回数 = explore json 的 rescued 计数字段
3. ⚠️ wandb 面板的 `explore/pass@10` 是回灌键名 legacy bug(round_tp.sh:418 写死),勿当真;一切以 json 为准。
4. **禁止抄记忆里记的数字出图**——必须现拉 json;记忆数字只做交叉校验,不一致时以服务器实况为准并查明原因再画。
5. 多容器 campaign(数据共用 CPFS):任一端口一把拉齐,不必分容器取。

## Step 2 — 拉数(ssh stdin 管道,免引号地狱)

用本 skill 自带 `scripts/dump_metrics.py`(DATA_ROOT 作 argv[1];标量键原样保留、列表缩成 `<key>#len`):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 -p 1022 root@106.14.2.243 \
  "/root/workspace/baojiachun/.venv/bin/python - <DATA_ROOT>" \
  < .agents/skills/plot-campaign-figs/scripts/dump_metrics.py \
  > <figs_dir>/raw_metrics.txt
```

- 服务器 ssh 间歇超时是常态:≤2 次重试;仍不通 → 停下报告,不猜数。
- `raw_metrics.txt` 与图同目录交付(可复算凭证)。
- glob 是 `**/rollout/*-exp*/log/*.json` 全拉,本地再按键辨 eval / explore(见坑速查),别只认文件名。

## Step 3 — 出图(本地 matplotlib)

- 本地 python 跑(cwd = repo 根,**一律相对路径**,中文路径坑);`plot_metrics.py` 与 PNG 同目录交付,DATA 以字面量嵌入(th95 先例式,可复跑)。
- 照抄 th95 3seed 模板结构:figsize(6.5,4.5)、dpi 200;每臂跨 seed **mean±sd 色带(alpha .18)+ 均值线 + marker**;x = round(从 1 起);左下 `band = ±1 s.d. across seeds (n=N; seeds ...)` 脚注;grid alpha .3;legend lower right;标题 `<Task> - <Metric>, mean across N seeds`(图内文案用英文,先例如此)。
- **端点数值标注 = 对角错位配方**(tp15 定稿):按终值升序排 i=0,1,2,`xytext=(10+12i, 12i−6)` offset points、ha=left、va=center,xlim 右缘留 ~0.9 轮宽。**别用垂直堆叠/碰撞检测下移**(三端点聚簇在 ~16pt 数据窗内必局促、与下方臂的 marker 相撞),**也别用纯水平错位**(同值端点如 .91/.91 在 9-12pt 横距下文字必撞——文字宽 ~22pt);对角(横+纵同时错开)对聚簇与同值都免疫。
- 配色跨图固定:**ORBIT / SCOUT-orbit = #9467bd(紫)、ATY / SCOUT = #d62728(红)、DP = #1f77b4(蓝)**;出现新臂先按此语义就近映射,并在汇报里说明。
- SR 与 pass@K **分图**(fig1 SR / fig2 pass@K)。
- 末轮 eval-only 无 pass@K 的臂:断线 + NA 标注,不插值、不代填(补测是 launch-experiment 的事)。
- 单 seed campaign:无色带,细线 + marker(参照 2026_9_4 版)。

## Step 4 — 视觉验收(judge 门)

render 完 dispatch judge agent(只读 PNG):给 PNG 绝对路径 + 应有内容清单(臂 / 轮数 / 端点数值表 / 风格要求)→ 逐页 pass/fail;fail 项修复 → 重 render → 重 judge,直到全过。

⚠️ judge/主线 Read PNG 通道可能故障(只回「已上传 CDN」URL 不给图,tp13/tp15 会话均遇):fallback = 把 CDN URL 喂 `analyze_image` MCP 工具做同一验收(带同样的应有内容清单);判定标准不变。

⚠️ judge 全过 ≠ 用户接受:只画数据本身;用户没点名的元素(箭头 / 注释框 / 参考线 / 装饰)一律不加;示意图(非 metrics 图)另有「用户元素清单」规矩,不适用本 skill。

## Step 5 — 交付与落账

- 目录 `experiments/<date>_<campaign>_figs_summary/`:fig*.png + plot_metrics.py + raw_metrics.txt(+ 新写的辅助脚本)。
- campaign memory 追加交付行(图路径 + 一句话读数),MEMORY.md 索引同步;向用户汇报图路径 + 端点数值小结。
- git commit 听用户令(改动先过审红线照旧)。

## 坑速查

- 本地 bash 中文路径:一律相对路径(cwd = repo 根);新建/改文件用 Write/Edit 工具绝对路径。
- bash 变量赋值不做 glob 展开(`j=dir/*x; [ -f "$j" ]` 恒假)——glob 直接写进 for。
- tqdm \r 日志:`grep -o pattern file | tail -1`,别先 tail 整行再 grep。
- json 命名陷阱:guided 臂的 eval json 叫 `transport_{SCOUT|DP}_rollout_expN.json`(guide 名,不是臂名);探针/单进程 rescue json 无 `_explore_` 后缀。**explore json 也带 `success_rate` 键(= rescue 场景成功率)——按「含 success_rate ⇒ eval」辨会把 explore 全吞进 eval 分支**(tp15 实翻车);最稳判据 = 文件名含 `_explore_`,键只做交叉校验(pass_at_5 只在 explore)。
