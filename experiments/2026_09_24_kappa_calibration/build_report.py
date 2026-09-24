"""Build tables from the downloaded, validated metadata snapshot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.calibration.kappa_report import task_rows

TASKS = ["can", "square", "coffee", "threading", "tool_hang"]
SNAPSHOT = HERE / "snapshot"
LABELS = {"excellent": "优秀", "minimum": "通过", "not_above_DP": "未过", "control": "DP"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    rows, index = [], {}
    for phase in ["base", "round1", "round2"]:
        current = []
        for task in TASKS:
            for row in task_rows(SNAPSHOT / phase / task):
                row = dict(row, phase=phase)
                current.append(row)
                index[(phase, task, row["arm"])] = row
        (HERE / f"results_{phase}.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        rows.extend(current)

    if args.final:
        audit = read(SNAPSHOT / "FINAL_AUDIT.json")
        if (audit["status"] != "all_launched_available_pair_evaluations_complete"
                or audit["paired_retry_records"] != len(rows)
                or audit.get("own_tmux_sessions_remaining") != 0):
            raise ValueError("Final report requires a matching completed server audit")

    def cell(phase, task, arm):
        row = index.get((phase, task, arm))
        if row:
            return f"{row['rescued']}/{row['dp_rescued']}（{LABELS[row['verdict']]}）"
        folder = SNAPSHOT / phase / task / arm
        if folder.exists():
            return "运行中"
        calibration_file = SNAPSHOT / phase / task / f"{arm}.json"
        if calibration_file.exists():
            c = read(calibration_file)
            if c.get("C_converged") is False or c.get("calibration_converged") is False:
                return "core 标定未达标，未 rollout"
        if phase != "base" and not (SNAPSHOT / phase / task / "checkpoints.json").exists():
            return "缺少 checkpoint"
        return "未评估"

    def calibration(row):
        source = SNAPSHOT / row["phase"] / row["task"]
        manifest = source / row["arm"] / "manifest.json"
        if manifest.exists():
            return read(manifest)["calibration"]
        if row["arm"] in ["k1", "k2.5", "k5"]:
            return read(source / "diagnostics" / f"{row['arm']}.json")
        return {}

    candidates = []
    for task, method, base_arm, round_arm in [
            ("can", "preserve_natural_KL_quantile", "k2.5", "natural_q"),
            ("can", "independent_DP_KL_median", "kl_median", "kl_median"),
            ("threading", "independent_DP_KL_median", "kl_median", "kl_median")]:
        selected = [index.get((phase, task, base_arm if phase == "base" else round_arm))
                    for phase in ["base", "round1", "round2"]]
        if all(row and row["rescued"] > row["dp_rescued"] for row in selected):
            rule_parameters = dict(target_R=.01, relative_R_tolerance=.1, core_observations=128)
            if method == "preserve_natural_KL_quantile":
                transfer = read(SNAPSHOT / "round1" / task / "natural_transfer.json")
                rule_parameters.update(reference_quantile=transfer["reference_quantile"],
                                       base_kappa=transfer["base_kappa"],
                                       formula="kappa_round = quantile(natural_KL_round, reference_quantile)")
            else:
                rule_parameters.update(independent_DP_draws=8,
                                       formula="kappa = pooled median KL(final_i || anchor_j), i != j")
            candidates.append(dict(task=task, method=method,
                                   rule_parameters=rule_parameters,
                                   validated_pairs=[dict(row, calibration=calibration(row))
                                                    for row in selected]))
    (HERE / "validated_candidates.json").write_text(json.dumps({
        "scope": "Exploratory results on the listed checkpoint pairs, seeds42-141, five retries; not a guarantee for future checkpoints.",
        "candidates": candidates}, indent=2) + "\n", encoding="utf-8")

    lines = ["# κ 标定实验结果", "",
             f"报告生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
             "状态：" + ("已收账；缺失 checkpoint 的项目单独注明。" if args.final else "实验仍在进行，以下仅统计已经完成并通过配对核验的结果。"), "",
             "表中 `SCOUT/DP` 均指初始失败场景的救回数，不能与总 pass@5 的比值混淆。",
             "严格超过 DP 为通过，严格超过 1.5 倍 DP 为优秀；恰好 1.5 倍仍只算通过。", "",
             "## 统一规则的 base 检验", "",
             "P≈6 的允许区间为±10%，R≈0.01 的允许区间也为±10%。中位数法统一使用8次独立DP采样，",
             "令 κ 等于 pooled median KL(final_i || anchor_j), i≠j，再标定 R≈0.01。", "",
             "κ=1两阶段配方先在κ=2.5标定η，再保持η、改用κ=1；它不保持最终R。", "",
             "| 任务 | κ=1两阶段配方 | κ=1且重标定R | P≈6 | 直接 KL 中位数 |", "|---|---:|---:|---:|---:|"]
    p6_arms = dict(can="k2.5", square="k2.5", coffee="p6", threading="rmatched_k1", tool_hang="k2.5")
    for task in TASKS:
        lines.append(f"| {task} | {cell('base', task, 'k1')} | {cell('base', task, 'rmatched_k1')} | {cell('base', task, p6_arms[task])} | {cell('base', task, 'kl_median')} |")
    lines += ["", "κ=1两阶段配方在五任务base均过基本线，但最终R并不相同；coffee round2为9/DP11，跨轮检验失败。",
              "保持最终R的三项统一候选均在tool_hang base未过基本线；这些结果不构成所有可能跨任务规则不存在的证明。",
              "目前可复用的固定任务候选是can的自然KL分位数法，以及can/threading的DP KL中位数法。",
              "它们在本次base、round1、round2上均过基本线，但threading只有1、1、2个场景的净增益。"]
    lines += ["", "## 固定任务跨轮验证", "",
              "所有参考值固定在 base。tool_hang 的 P/R 和 C/R 参考使用实际成功点 R=0.007156729252424086，",
              "其余这里的 R 目标为0.01；没有把 tool 的较低剂量结果混称为 R=0.01。", "",
              "| 任务 / 方法 | Base | Round1 | Round2 |", "|---|---:|---:|---:|"]
    methods = [("can", "自然 KL 分位数", "k2.5", "natural_q"),
               ("coffee", "P=6", "p6", "p6"),
               ("coffee", "联合 C/R，base κ≈7.07", "p6", "joint_c"),
               ("coffee", "联合 C/R，原始 base κ=2.5", "k2.5", "joint_c_base25"),
               ("coffee", "先 R 后 C，固定 base 参考", "k2.5", "sequential_c"),
               ("coffee", "保持 base κ=2.5，仅标定η", "k2.5", "fixed_base_kappa"),
               ("coffee", "κ=1两阶段配方", "k1", "fixed_k1_after_r25"),
               ("threading", "固定 base P", "k2.5", "pbase"),
               ("threading", "联合 C/R", "k2.5", "joint_c"),
               ("threading", "先 R 后 C，固定 base 参考", "k2.5", "sequential_c"),
               ("tool_hang", "固定 base P/R", "k1", "pbase"),
               ("tool_hang", "联合 C/R", "k1", "joint_c")]
    methods.append(("tool_hang", "保持 base κ=1，仅标定η", "k1", "fixed_base_kappa"))
    for task, method, base_arm, round_arm in methods:
        lines.append(f"| {task} / {method} | {cell('base', task, base_arm)} | {cell('round1', task, round_arm)} | {cell('round2', task, round_arm)} |")
    for task in TASKS:
        lines.append(f"| {task} / KL 中位数 | {cell('base', task, 'kl_median')} | {cell('round1', task, 'kl_median')} | {cell('round2', task, 'kl_median')} |")
    lines += ["", "下列对照的优秀线超过初始失败场景总数，因此不可达；基本条件仍要求严格超过DP：", ""]
    for row in rows:
        if row["arm"] == "dp" and not row["excellent_attainable"]:
            lines.append(f"- {row['task']} {row['phase']}：失败场景{row['n_failed']}，DP救回{row['rescued']}，"
                         f"优秀线{row['excellent_threshold']}，基本通过线{row['minimum_threshold']}。")
    lines += ["", "## 每项已完成 SCOUT 评估", "",
              "κ=1/2.5/5 的原始探针固定η；它们的 R 并不相同。rmatched 与新校准方法均报告实际测得的R。", "",
              "sequential_c 先在上一轮κ处将R标定到0.01，再固定η匹配base C；最终R允许变化。", "",
              "| 阶段 | 任务 | 方法 | κ | η | R | 救回 SCOUT/DP | 总 pass@5 | 新救回/丢失 DP 场景 | 判定 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["arm"] == "dp":
            continue
        c = calibration(row)
        params = [f"{c[key]:.6g}" if key in c else "—" for key in ["kappa", "eta", "R_mean"]]
        lines.append(f"| {row['phase']} | {row['task']} | {row['arm']} | {' | '.join(params)} | "
                     f"{row['rescued']}/{row['dp_rescued']} | {row['pass_at_5']:.2f} | "
                     f"{row['gain_vs_dp']}/{row['loss_vs_dp']} | {LABELS[row['verdict']]} |")
    lines += ["", "## 口径、证据与限制", "",
              "- 初始评估100个场景，seed42–141；冻结初始失败集，DP和SCOUT各5次重试，4个worker，每个25环境。",
              "- 总pass@5按代码口径包含初始成功场景。只在失败集上增加重试。",
              "- 实际R的分母是core原始abs_actions的平均绝对值，并非noisy action。",
              "- κ截断梯度作用范围，不把最终动作投影回KL球；最终未截断KL可以超过κ。",
              "- 所有完成记录核对DP/dyn身份、失败场景集合、逐场景救回数、5次重试和4个分片。",
              "- 这些是在固定场景集上的探索结果，训练seed为233；未进行额外训练seed或新场景集的独立确认。",
              "- 有限候选失败不能证明所有跨任务规律都不存在；core标定收敛也不能替代环境评估。",
              "- Square round1/2尚缺可用checkpoint，已找到旧删除记录；没有用exp6替代它们。", "",
              f"详细方法见 [METHODS.md]({(HERE / 'METHODS.md').as_posix()})。",
              f"原始元数据快照在 [snapshot]({SNAPSHOT.as_posix()})；该目录是下载时点的副本。",
              f"通过三轮基本条件的候选及完整参数在 [validated_candidates.json]({(HERE / 'validated_candidates.json').as_posix()})。",
              "服务器权威目录：`/mnt/workspace/baojiachun/scout/experiments/2026_09_24_kappa_calibration`。", ""]
    if args.final:
        usage = read(SNAPSHOT / "resource_usage.json")
        lines += [f"最终审计：{audit['initial_evaluations']}次初始评估、{audit['DP_controls']}条DP对照、"
                  f"{audit['SCOUT_arms']}条SCOUT评估；本次实验的控制进程均已退出。",
                  f"原始rollout HDF5保留在服务器实验目录；该目录约{usage['experiment_logical_GiB']:.1f} GiB。"
                  "本地快照仅包含元数据与诊断数组。", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Validated {len(rows)} completed control/SCOUT rows and wrote RESULTS.md")


if __name__ == "__main__":
    main()
