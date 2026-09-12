# 2026-09-12 服务器副本收编(server consolidation)

用户令:服务器上 scout 为主 repo,其余 worktree 副本的分身。本目录记录两件事的账:
①各分身 `data/` 的训练产物已并入主 repo `scout/data/`(见下表,同文件系统 `mv`,瞬时、零拷贝);
②散落在各分身 `soe_scripts/`、`scout-gae/soe_scripts`(脚本 hub)、`scout-gae/scripts/probe/`、`baojiachun/soe_scripts/` 与 baojiachun 根目录的脚本已按「服务器为准、云端格式」收编进 `scripts/`。

## 脚本收编规则与产物

- **canonical 规则**:同一目标路径的全部服务器候选(12 个副本的 soe_scripts + scout-gae/scripts 结构化树 + baojiachun/soe_scripts + 根目录散件)按 mtime 最新者胜;分歧旧版存档于 `script_variants/`(12 个),逐文件决策表见 `script_audit.csv`。
- **路径映射**:已在云端 main 的脚本落回原路径;新脚本按 GAElike-dev 确立的分类(`probe/`、`rand/`)与既有 9 类归位。首版 collector 有 21 个文件被规则引擎放错位置,本地修复时逐一核对:21 个平铺版与分支版**逐字节相同**(0 内容损失),错位副本已删。
- **NAMES.md 特例**:术语表以 drift-dev 版为准(含 WalkCloud/soft-min 核/τ̃ adapt 最新词条,是 GAElike 版的严格超集);服务器三个旧版(gaelike 检出/scout-old/scout-exploit)全部存档于 `script_variants/`。
- **wc 探针脚本来源**:walkcloud 四探针 + wc_chain.sh 的服务器真身(/tmp/scout-drift)已被 /tmp 清理抹掉,唯一幸存副本在 drift-dev 分支,本次从 git 取回入 main。
- **worktree_state/**:各 git 副本搬迁前的未提交改动(`*.tracked.diff`)与未跟踪清单(`*.status.txt`),保全了如 scout-orbit 的 rollout_vec.py(deb826a 世代)等只在工作区存在的代码修改。
- 各分身中未收编的未跟踪杂项(configs/eval_can_e2*.yaml、configs/_tmp_*.yaml、relay2.log、*.bak-*)留在原处未动;分身 worktree 本体未删除(用户未令)。

## 数据搬迁账(scout/data,2026-09-12 执行,日志 consolidate_20260912/data_moves.log)

| 目录 | 来源分身 | 大小 | 实验身份 | 日期 |
|---|---|---|---|---|
| 2026_8_21_entropy | scout-entropy | 40G | CAN entropy 链(s233/s2333/s23333,e2 世代) | 08-24~25 |
| 2026_8_26_entropy | scout-entropy | 41G | SQUARE entropy 三 seed + core_rebuild(含 GAElike 探针复用的 round0 资产) | 08-26~31 |
| square_calib | scout-entropy | 3.3G | square entropy 剂量标定(s3/s5/s8 系) | 08-26 |
| sq_render_probe | scout-entropy | 10K | square 渲染探针 | 08-26 |
| vis_first_chunk | scout-entropy | 626M | 首 chunk 可视化 + run logs | 08-27 |
| gae_probe_can824 | scout-gae | 81G | GAElike CAN 探针(r0..r202/sweep0-90/ledger,基座 CAN-8-24-entropy) | 09-08~09 |
| gae_probe_sq91 | scout-gae | 18G | GAElike SQUARE 探针(r1/r2/evalonly+ledger) | 09-09 |
| fbclamp_test / hparam_test / sched_test | scout-hparam | 15G/4.7G/11G | orbit-hparam 标定(fb clamp/dimless/schedule,产出终版参数组 η̃0.33/σ0.16×0.5^(r−1)/fb soft/anneal2) | 09-02 |
| 2026_9_1_orbchain | scout-orbit | 1.1T | SQUARE orbit 链群:v3 终版(s233,完赛 09-04 SR .72/p@10 .89)+ s2333/s23333 b256 种子复刻(完赛 09-12,.63/.51) | 09-01~12 |
| 2026_9_2_orbchain | scout-orbit | 578G | CAN orbit 链群:v3 终版(s233,完赛 09-03 .84/.90)+ canpk 复刻 s2333/s23333(完赛 09-09/12,.91/.87) | 09-01~12 |
| 2026_9_2_orbchain_v0_can_s233 | scout-rand | 128G | CAN-9-2 s233 **v1 原版实体**(被 v3 重启替代的废链;撞名故加后缀) | 09-02 |
| 2026_9_1_toolhang | scout-orbit | 96G | TOOLHANG 早期尝试(被 09-04 重跑替代) | 09-03 |
| 2026_9_3_atyprobe | scout-orbit | 7.1G | aty s3/s30 剂量标定 | 09-03 |
| 2026_9_4_toolhang | scout-orbit | 632G | TOOLHANG-9-4 round0+双臂链(r3 完停,SCOUT .60>DP .42) | 09-04~07 |
| 2026_9_5_toolhang | scout-th95 | 3.3T | TOOLHANG-9-5 三 seed 三臂链(DP/ATY/ORBIT × s233/s2333/s23333,完赛 09-11/12,含 p5meas 补测、vis_r5_grid、ATY exp4 accum 备份) | 09-05~12 |
| _smoke_canpk / _smoke_sqb256 | scout-orbit | 5.0G/5.2G | canpk/sqb256 链启动冒烟 | 09-08 |
| _smoke_th6_1508 | scout-orbit | 96K | th6 冒烟 | 09-03 |
| perf_probe | scout-orbit | 2.5G | rollout 性能探针(H/M/L/S) | 09-03 |
| shardab2 / shardab4 | scout-orbit | 64K/57K | shard A/B 等价性测试 | 09-01 |
| toolhang_calib | scout-orbit | 5.3G | toolhang 剂量标定(SCALE_CHOSEN_12.0) | 09-01 |
| rand | scout-rand | 30G | entropy-random 探针群(dose/es_rho/fa_retry…+RESULTS.md) | 08-27~28 |
| particle | scout-rand | 52G | particle 族探针(G1/G2/G3 ps0/50/90、base_eval、cal_att) | 08-30 |
| can_orb_cal | scout-rand | 5.8G | CAN orbit η̃ 标定(s005-s050) | 09-02 |
| aty_test_s233_r4trio | scout-rand | 4.4G | aty r4 三件套测试 | 09-02 |
| hb_smoke_th94(-shard0of1) | scout-th94 | 3.5G | th94 心跳冒烟 | 09-04 |

搬入合计 ≈6.2T;`scout/data` 总计 ≈6.4T(原有 2026_8_14/2026_8_21/robomimic/transport/vis_final/logs 未动)。

## 其他记账

- **/tmp/scout-drift 已丢**:walkcloud/drift 探针的服务器数据与脚本随 /tmp 清理消失;账本与 probe log 幸存于 repo drift-dev(1439900/b8ef439),脚本本次已入 main。
- **scout-th94 软链**:原指向 scout-orbit/data/2026_9_4_toolhang 的软链已改指 `scout/data/2026_9_4_toolhang`。
- **git 分支核对**:服务器各分支(exploit-dev/GAElike-dev/entropy-random-dev)均为云端对应分支祖先,无未推送提交;服务器 main 停在初始提交 cfdfa13,本次直接快进到云端 main。
- **数据侧主清单**:`scout/data/CONSOLIDATION_20260912.md`(服务器,gitignore 内)与本文件同源。
