#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Export trained PPO checkpoint to TorchScript .pt for Go2 Edu real-robot deployment.

Deployment model (DeployActor):
    Input : obs (batch, 48)  -> [lin_vel(3), ang_vel(3), gravity(3), commands(3),
                                 dof_pos(12), dof_vel(12), actions(12)]
    Output: action (batch, 12)

Internally maintains trajectory_history (1, history_length, 42) and runs:
    1. update history with obs[:, 3:9] + obs[:, 12:]  (42 dims)
    2. latent = history_encoder(history.flatten(1))
    3. action = actor(concat(latent, obs[:, 3:]))

Only the actor path is exported (critic is simulator-only, not needed on robot).

导出训练好的 PPO checkpoint 为 TorchScript .pt，用于 Go2 Edu 真机部署。
部署模型只保留 actor 路径（critic 是仿真专用，真机不需要）。
"""

import torch
import torch.nn as nn

from agent_ppo.model.model4onnx import TeacherActorCritic
from agent_ppo.conf.conf import Config


class DeployActor(nn.Module):
    """
    Deployment-only actor wrapper.
    Takes raw 48-dim obs, maintains trajectory_history internally, outputs 12-dim action.
    部署专用 actor 包装器：吃 48 维原始 obs，内部维护 trajectory_history，输出 12 维 action。
    """

    def __init__(self, history_encoder, actor, history_length, single_history_dim):
        super().__init__()
        self.history_encoder = history_encoder
        self.actor = actor
        self.history_length = history_length
        self.single_history_dim = single_history_dim
        # Trajectory history buffer (persistent across forward calls)
        # 轨迹历史缓冲（跨 forward 调用持久化）
        self.register_buffer(
            "trajectory_history",
            torch.zeros(1, history_length, single_history_dim),
        )

    def reset(self):
        """Clear trajectory history (call at episode start on real robot).
        清空轨迹历史（真机每 episode 开始时调用）。"""
        self.trajectory_history.zero_()

    def forward(self, obs):
        """
        Args:
            obs: (batch, 48) tensor
        Returns:
            action: (batch, 12) tensor
        """
        # 1. Extract history obs: obs[:, 3:9] + obs[:, 12:] = 42 dims
        #    (drop lin_vel(3) + commands(3))
        # 1. 提取历史观测：去掉 lin_vel 和 commands，剩 42 维
        obs_without_command = torch.cat(
            (obs[:, 3:9], obs[:, 12:]),
            dim=1,
        )

        # 2. Roll history window: shift left, append new at end
        # 2. 滚动历史窗口：左移一位，末尾追加新观测
        new_history = torch.cat(
            (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
            dim=1,
        )
        # In-place update so next forward sees updated history
        # 就地更新缓冲，下一次 forward 能看到更新后的历史
        self.trajectory_history.copy_(new_history)

        # 3. Encode history to latent
        # 3. 将历史编码为 latent 向量
        latent_vector = self.history_encoder(new_history.flatten(1))

        # 4. Actor: concat(latent, obs[:, 3:]) -> action
        # 4. Actor 策略网络：concat(latent, obs[:, 3:]) -> action
        obs_for_actor = obs[:, 3:]
        actor_input = torch.cat((latent_vector, obs_for_actor), dim=-1)
        action = self.actor(actor_input)
        return action


def load_model(ckpt_path, history_length=5, device="cpu"):
    """
    Build a TeacherActorCritic matching the checkpoint and load weights.
    构建与 checkpoint 结构匹配的 TeacherActorCritic 并加载权重。

    IMPORTANT: history_length must match the training-time value (default 5 per Config).
    重要：history_length 必须与训练时一致（Config 默认 5）。
    """
    # Training-time architecture (hard-coded to match agent_ppo defaults & ckpt shapes)
    # 训练时网络结构（与 agent_ppo 默认值和 checkpoint shape 对齐）
    num_actions = 12
    num_critic_obs = Config.NUM_PRIVILEGED_OBS          # 259
    single_history_dim = Config.HISTORY_OBS_DIM         # 42
    history_dim = history_length * single_history_dim   # 5 * 42 = 210

    model = TeacherActorCritic(
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        num_actor_obs=Config.NUM_OBSERVATIONS,          # 48
        actor_obs_dim_in=Config.ACTOR_INPUT_OBS_DIM,    # 45
        history_dim=history_dim,                        # 210
        single_history_dim=single_history_dim,          # 42
        num_envs=1,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, history_length, single_history_dim


if __name__ == "__main__":
    CKPT_PATH = "ckpt/model.ckpt-21471.pkl"
    OUT_PATH = "21471.pt"
    DEVICE = "cpu"
    HISTORY_LENGTH = 5  # MUST match training config (5 * 42 = 210 history_dim)

    # 1. Load full model (actor+critic+history_encoder) from checkpoint
    # 1. 从 checkpoint 加载完整模型
    full_model, history_length, single_history_dim = load_model(
        CKPT_PATH, history_length=HISTORY_LENGTH, device=DEVICE
    )
    print(
        f"[OK] Loaded checkpoint: history_length={history_length}, "
        f"history_dim={history_length * single_history_dim}, "
        f"actor_input={full_model.actor[0].in_features}, "
        f"num_actions={full_model.actor[-1].out_features}"
    )

    # 2. Wrap into deployment-only actor (critic is dropped)
    # 2. 包装成部署专用 actor（丢弃 critic）
    deploy_model = DeployActor(
        history_encoder=full_model.history_encoder,
        actor=full_model.actor,
        history_length=history_length,
        single_history_dim=single_history_dim,
    ).to(DEVICE)
    deploy_model.eval()

    # 3. Trace with dummy 48-dim obs
    # 3. 用 48 维 dummy obs 做 trace
    dummy_obs = torch.zeros(1, Config.NUM_OBSERVATIONS, device=DEVICE)  # (1, 48)
    with torch.no_grad():
        traced = torch.jit.trace(deploy_model, dummy_obs, check_trace=False)

    # Sanity check: run traced model a couple of times
    # 简单验证：跑几次 traced 模型
    with torch.no_grad():
        for i in range(3):
            out = traced(dummy_obs)
        print(f"[OK] Traced model output shape: {tuple(out.shape)}, "
              f"sample values: {out[0, :4].tolist()}")

    traced.save(OUT_PATH)
    print(f"[OK] Exported TorchScript model to: {OUT_PATH}")
    print("     Usage on real robot:")
    print("       model = torch.jit.load('21471.pt')")
    print("       model.reset()  # call at each episode start")
    print("       action = model(obs_48d)  # obs shape: (1, 48)")
