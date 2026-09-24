# scout 模块

这里包含 SCOUT 的模型、guidance、标定、rollout 和训练实现。当前标准训练由根目录
`train.py` 读取 campaign 配置并调用 `scripts/atom/` 中的原子操作。

核心目录：

- `model/`：状态编码器、VIB encoder 和动力学 decoder；
- `guidance/`：KL cost、atypical 和 ORBIT guidance；
- `calib/`：P6、KL median、R/C 等剂量标定；
- `eval/`：固定场景 eval、rescue explore、shard merge 和 HDF5 数据选择；
- `train_vib.py`：dyn/VIB 训练入口。

标准 threading 流程见 [campaign 配置](../configs/threading/campaign.json) 和
[atom 说明](../scripts/README.md)。
