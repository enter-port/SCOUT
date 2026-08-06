# SCOUT 训练策略:VIB 动力学模型。
#   编码器  p̄_theta(z | S_t, A_t)  -> (mu, logvar)   用 SOE 的 EncoderMLP
#   解码器  q_phi(S_{t+1} | S_t, z) -> Ŝ_{t+1}        确定性回归(第一版)
#   loss   = next-state 重建 MSE + beta * KL[p̄(z|S,A) || N(0,I)]
#
# 与 SOE 的 DPExt (policy/dp_ext.py) 的关系:
#   - 风格对齐:loss dict + 显式 backward;
#   - 本质区别:
#       (1) 解码器预测 next state(DPExt 通过共用 D 预测 action);
#       (2) base DP 在 VIB 训练时不在场(冻结、单独训练),
#           因此前向是一条直链,无需梯度隔离 —— backward() 直接 loss.backward()。
#
# 训练目标的解读(见 idea/idea_notes.md §3):
#   max I(Z; S_{t+1}|S_t) - beta I(Z; A_t|S_t)
#   => VIB 上界 = E[-log q_phi(S_{t+1}|S_t,z)] + beta*KL
#   确定性回归下,-log q 用 ||S_{t+1} - Ŝ_{t+1}||^2 近似(MSE)。

import os as _os
import sys as _sys

_S = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "SOE", "src"))
if _S not in _sys.path:
    _sys.path.insert(0, _S)

import torch
import torch.nn as nn

from policy.vqvae_modules.vqvae import EncoderMLP  # noqa: E402


class ScoutDynamicsDecoder(nn.Module):
    """q_phi(S_{t+1} | S_t, z):确定性回归预测下一状态(第一版)。"""

    def __init__(self, state_dim, style_dim, hidden_dim=256, layer_num=2):
        super().__init__()
        self.net = EncoderMLP(
            input_dim=state_dim + style_dim,
            output_dim=state_dim,
            hidden_dim=hidden_dim,
            layer_num=layer_num,
        )

    def forward(self, state_t, z):
        return self.net(torch.cat([state_t, z], dim=-1))


class SCOUT(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        style_dim=16,
        hidden_dim=256,
        kl_weight=1e-3,
        encoder_layer_num=2,
        decoder_layer_num=2,
    ):
        """
        Args:
            state_dim:  状态向量维度(各 low_dim obs key 拼接后的总维)。
            action_dim: 单步动作维度。
            style_dim:  skill 潜空间维度 d(默认 16,与 SOE 一致)。
            kl_weight:  beta —— VIB 的 KL 权重,本方法的 make-or-break 旋钮。
        """
        super().__init__()
        # p̄_theta(z | S_t, A_t) -> (mu, logvar)
        self.encoder = EncoderMLP(
            input_dim=state_dim + action_dim,
            output_dim=style_dim * 2,
            hidden_dim=hidden_dim,
            layer_num=encoder_layer_num,
        )
        # q_phi(S_{t+1} | S_t, z)
        self.decoder = ScoutDynamicsDecoder(state_dim, style_dim, hidden_dim, decoder_layer_num)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.style_dim = style_dim
        self.kl_weight = kl_weight

    def encode(self, state_t, action_t):
        """返回 (mu, logvar, std)。"""
        out = self.encoder(torch.cat([state_t, action_t], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std

    def forward(self, state_t, action_t, state_tp1, **kwargs):
        """
        训练前向。batch 由 train_scout.pre_process_data 传入三个 (B, dim) 张量。
        推理/探索不在此处(见 guided_sampler.py)。
        """
        mu, logvar, std = self.encode(state_t, action_t)
        # 重参数化 z = mu + std * eps(训练用标准重参数化,无 noise_scale)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # ① next-state 重建:mean over batch, sum over dims(与 KL 同口径,便于 beta 权衡)
        state_tp1_hat = self.decoder(state_t, z)
        recon_loss = torch.mean(torch.sum((state_tp1 - state_tp1_hat) ** 2, dim=-1))

        # ② KL[p̄(z|S,A) || N(0,I)],解析对角高斯 KL,与 DPExt 一致
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

        loss = recon_loss + self.kl_weight * kl_loss
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def backward(self, loss):
        # base DP 不在场,直链,单次 backward 即可,无需梯度隔离。
        loss["loss"].backward()

    # ---- 推理期编码器接口(供 guided_sampler 使用)----
    @torch.no_grad()
    def get_mu(self, state_t, action_t):
        """返回编码器均值 mu(s, a)(测试期 classifier guidance 用)。"""
        mu, _, _ = self.encode(state_t, action_t)
        return mu
