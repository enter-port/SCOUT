#!/bin/bash
# experiment2 setup (user 2026-08-17): mirror experiment1's layout under
# data/experiment2 with hardlinked cores + frozen DP-base (zero-copy), then
# write the *_e2 configs and round2.sh driver. ONLY difference from
# experiment1: VIB beta 1e-4 -> 1e-3 (diag 2026-08-17 showed KL holds ~0.46
# nats with free_bits, so the 10x sweep is worth a full chain).
set -eu
cd /root/workspace/baojiachun/scout

E1=data/experiment1
E2=data/experiment2

mkdir -p "$E2/logs" \
         "$E2/can/rollout" "$E2/can/train/DP" "$E2/can/train/dyn" \
         "$E2/square/rollout" "$E2/square/train/DP" "$E2/square/train/dyn"

# cores: hardlink from experiment1 (same inode as the data/robomimic source)
ln -f "$E1/can/rollout/can_core.hdf5"       "$E2/can/rollout/can_core.hdf5"
ln -f "$E1/square/rollout/square_core.hdf5" "$E2/square/rollout/square_core.hdf5"

# frozen DP-base: hardlink tree (~31G each, zero extra disk)
cp -al "$E1/can/train/DP/DP-base"    "$E2/can/train/DP/DP-base"
cp -al "$E1/square/train/DP/DP-base" "$E2/square/train/DP/DP-base"

# VIB configs: experiment2 paths + beta 1e-3 + wandb 2-*
for t in can square; do
  sed -e 's#data/experiment1#data/experiment2#g' \
      -e 's/^beta: 1\.0e-4.*/beta: 1.0e-3                # experiment2: 10x beta (user 2026-08-17); diag: KL holds ~0.46 with free_bits floor/' \
      -e 's/^  project: 1-\(.*\)-dyn$/  project: 2-\1-dyn/' \
      "configs/vib_${t}_image.yaml" > "configs/vib_${t}_image_e2.yaml"
  sed 's#data/experiment1#data/experiment2#g' \
      "configs/eval_${t}.yaml" > "configs/eval_${t}_e2.yaml"
done

# round2.sh: same driver, experiment2 data root + 2-* wandb projects + _e2 configs
sed -e 's#data/experiment1#data/experiment2#g' \
    -e 's/vib_${TASK}_image\.yaml/vib_${TASK}_image_e2.yaml/g' \
    -e 's/vib_{task}_image\.yaml/vib_{task}_image_e2.yaml/g' \
    -e 's/eval_${TASK}\.yaml/eval_${TASK}_e2.yaml/g' \
    -e 's/1-${TASK}-eval/2-${TASK}-eval/' \
    -e 's/1-${TASK}-DP/2-${TASK}-DP/g' \
    -e 's/1-{task}-dyn/2-{task}-dyn/' \
    soe_scripts/round.sh > soe_scripts/round2.sh
chmod +x soe_scripts/round2.sh

cat > "$E2/experiment.md" <<'MDEOF'
# experiment2 — SCOUT 链 β=1e-3(can / square)

启动:2026-08-17(用户指令)。与 experiment1 的**唯一差异:VIB β = 1e-3**(experiment1 为 1e-4)。

依据(2026-08-17 diag,`diag_vib/out_can_beta1e-3`,can core):β=1e-3 下 KL 稳定 ~0.46 nats 不消失(free_bits=0.05 地板,全程 0.43–0.49 无塌缩趋势);latent_mse 0.0080 与 1e-4 持平;guidance |dNLL/da| = 0.096(1e-4 为 0.181,减半但非零)。

## 设置(其余全部同 experiment1,见 data/experiment1/experiment.md)
- 数据:core 硬链自 data/robomimic;DP retrain 用 success_accum、dyn retrain 用 all_accum(均跨轮累积)
- DP-base:硬链自 experiment1(can/square 各 checkpoints/580.ckpt,31G 零拷贝,inode 相同)
- 驱动:`soe_scripts/round2.sh`(round.sh 的 experiment2 版:DATA=data/experiment2、wandb 项目 2-*、config 用 *_e2)
- 配置:`configs/vib_{can,square}_image_e2.yaml`(β=1e-3)、`configs/eval_{can,square}_e2.yaml`
- seed 42 固定(SOE 协议,同 experiment1)
- GPU:can-SCOUT=4、square-SCOUT=5;会话 `round2_<task>_SCOUT_<exp>`;console log 在 `data/experiment2/logs/`
- wandb 项目:`2-can-dyn` / `2-square-dyn` / `2-can-eval` / `2-square-eval`(SCOUT 链无 DP-retrain 项目名冲突,run 名 task 在前)

## 观察点
- KL 是否维持(对照 diag ~0.46);
- guidance 梯度减半后探索是否更保守:jerk / rescued 与 experiment1 的 β=1e-4 链对比;
- 六轮 success_rate / pass@5 曲线 vs experiment1。
MDEOF

echo "=== verify tree ==="
find "$E2" -maxdepth 3 | sort
echo "=== verify beta/wandb ==="
grep -H '^beta:' configs/vib_can_image_e2.yaml configs/vib_square_image_e2.yaml
grep -H 'project:' configs/vib_can_image_e2.yaml configs/vib_square_image_e2.yaml
grep -H 'save_dir:' configs/vib_can_image_e2.yaml configs/vib_square_image_e2.yaml
echo "=== verify round2.sh diffs ==="
diff soe_scripts/round.sh soe_scripts/round2.sh | head -40
