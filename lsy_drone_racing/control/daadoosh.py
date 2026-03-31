# This is the custom controller implementation. Another other customer controller implementations should be placed in separate files.
# Only requirement is to inherit from the base class Controller and implement its methods. Any type of logic can be used here including reinforcement learning.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as e:
    raise ImportError("This controller requires PyTorch. Install torch first.") from e


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value


class MyController(Controller):
    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)

        self.config = config
        self.device = torch.device("cpu")

        self.obs_dim = 10
        self.act_dim = 3

        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_speed = 1.00
        self.deadzone_max_speed = 0.45
        self.target_z_blend = 0.20

        # Gate geometry offsets
        self.gate_center_up_offset = 0.0
        self.gate_side_offset = 0.00
        self.gate_approach_offset = 0.30

        # Gate transit behavior
        self.pre_gate_deadzone = 0.55
        self.post_gate_clearance = 0.45
        self.deadzone_speed = 0.32
        self.base_speed_far = 0.90
        self.base_speed_mid = 0.55
        self.base_speed_near = 0.30
        self.deadzone_smoothing = 0.985
        self.normal_smoothing = 0.90
        self.normal_residual_scale = 0.30
        self.deadzone_residual_scale = 0.10
        self.low_altitude_boost = 0.45

        # Reward shaping
        self.w_gate = 10.0
        self.w_progress = 4.0
        self.w_time = 0.01
        self.w_crash = 20.0
        self.w_ground = 10.0
        self.w_unstable = 2.50
        self.w_backward = 8.0
        self.ground_z_thresh = 0.12

        self.model_path = str(Path(__file__).with_name("my_controller_actor_critic.pt"))
        if os.path.exists(self.model_path):
            self.policy.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self._initial_obs = self._clone_obs(obs)
        self.reset()

    def _clone_obs(self, obs: dict) -> dict:
        cloned = {}
        for key, value in obs.items():
            cloned[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        return cloned

    def _quat_to_rot(self, quat: NDArray[np.floating]) -> R:
        return R.from_quat(np.asarray(quat, dtype=np.float64))

    def _normalize(self, vec: NDArray[np.floating]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / norm).astype(np.float32)

    def _scalar_int(self, value) -> int:
        return int(np.asarray(value).reshape(-1)[0])

    def _get_gate_axes(self, obs: dict[str, NDArray[np.floating]], gate_idx: int):
        gate_quat = np.asarray(obs["gates_quat"][gate_idx], dtype=np.float32)
        rot = self._quat_to_rot(gate_quat)

        gate_forward = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64)).astype(np.float32)
        gate_right = rot.apply(np.array([0.0, 1.0, 0.0], dtype=np.float64)).astype(np.float32)
        gate_up = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64)).astype(np.float32)

        gate_forward = self._normalize(gate_forward)
        gate_right = self._normalize(gate_right)
        gate_up = self._normalize(gate_up)
        return gate_forward, gate_right, gate_up

    def _get_gate_center(self, obs: dict[str, NDArray[np.floating]], gate_idx: int) -> np.ndarray:
        gate_origin = np.asarray(obs["gates_pos"][gate_idx][:3], dtype=np.float32)
        _, gate_right, gate_up = self._get_gate_axes(obs, gate_idx)
        gate_center = (
            gate_origin
            + self.gate_center_up_offset * gate_up
            + self.gate_side_offset * gate_right
        )
        return gate_center.astype(np.float32)

    def _env_target_gate(self, obs: dict[str, NDArray[np.floating]]) -> int:
        return self._scalar_int(obs["target_gate"])

    def _build_gate_lock(
        self, obs: dict[str, NDArray[np.floating]], gate_idx: int, drone_pos: np.ndarray
    ) -> None:
        gate_center = self._get_gate_center(obs, gate_idx)
        gate_forward, _, _ = self._get_gate_axes(obs, gate_idx)
        signed_dist = float(np.dot(drone_pos - gate_center, gate_forward))

        self.locked_gate_idx = gate_idx
        self.locked_gate_approach_sign = 1.0 if signed_dist >= 0.0 else -1.0
        self.transit_direction = self._normalize(
            (-self.locked_gate_approach_sign) * gate_forward
        )
        self.deadzone_target = (
            gate_center + self.post_gate_clearance * self.transit_direction
        ).astype(np.float32)
        self.deadzone_active = False
        self.locked_gate_passed = False

    def _clear_gate_lock(self) -> None:
        self.locked_gate_idx = None
        self.deadzone_active = False
        self.locked_gate_passed = False
        self.transit_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.locked_gate_approach_sign = 1.0
        self.deadzone_target = None

    def _gate_passed_for_lock(self, obs: dict[str, NDArray[np.floating]], gate_idx: int) -> bool:
        env_target = self._env_target_gate(obs)
        if env_target == -1:
            return True
        return env_target != gate_idx and env_target > gate_idx

    def _enter_deadzone(self) -> None:
        # Lock the current gate and stop re-aiming at a new goal while transiting it.
        self.deadzone_active = True

    def _gate_clearance_distance(
        self, obs: dict[str, NDArray[np.floating]], gate_idx: int, drone_pos: np.ndarray
    ) -> float:
        gate_center = self._get_gate_center(obs, gate_idx)
        return float(np.dot(drone_pos - gate_center, self.transit_direction))

    def _update_gate_lock(self, obs: dict[str, NDArray[np.floating]]) -> None:
        drone_pos = np.asarray(obs["pos"], dtype=np.float32)
        env_target = self._env_target_gate(obs)

        if self.locked_gate_idx is None:
            if env_target != -1:
                self._build_gate_lock(obs, env_target, drone_pos)
            return

        if (
            not self.deadzone_active
            and not self.locked_gate_passed
            and env_target not in (-1, self.locked_gate_idx)
        ):
            self._build_gate_lock(obs, env_target, drone_pos)

        gate_center = self._get_gate_center(obs, self.locked_gate_idx)
        dist_to_center = float(np.linalg.norm(drone_pos - gate_center))

        if not self.deadzone_active and dist_to_center <= self.pre_gate_deadzone:
            # Commit to the locked gate once the drone is close enough to transit it.
            self._enter_deadzone()

        if not self.locked_gate_passed and self._gate_passed_for_lock(obs, self.locked_gate_idx):
            # Keep the old gate locked after pass until the gate plane is physically cleared.
            self.locked_gate_passed = True
            self._enter_deadzone()

        if self.locked_gate_passed:
            clearance = self._gate_clearance_distance(obs, self.locked_gate_idx, drone_pos)
            if clearance >= self.post_gate_clearance:
                self._clear_gate_lock()
                if env_target != -1:
                    # Only relock once the previous gate has been fully cleared.
                    self._build_gate_lock(obs, env_target, drone_pos)

    def _get_gate_target(
        self,
        obs: dict[str, NDArray[np.floating]],
        gate_idx: int,
        drone_pos: np.ndarray,
        approach_sign: float,
    ) -> np.ndarray:
        gate_center = self._get_gate_center(obs, gate_idx)
        gate_forward, _, _ = self._get_gate_axes(obs, gate_idx)
        approach_target = gate_center + approach_sign * self.gate_approach_offset * gate_forward

        if np.linalg.norm(gate_center - drone_pos) < 0.8:
            return gate_center.astype(np.float32)
        return approach_target.astype(np.float32)

    def _get_navigation_target(
        self, obs: dict[str, NDArray[np.floating]], drone_pos: np.ndarray
    ) -> tuple[int, np.ndarray, np.ndarray, float]:
        self._update_gate_lock(obs)

        if self.locked_gate_idx is None:
            zero = np.zeros(3, dtype=np.float32)
            return -1, zero, zero, 0.0

        if self.deadzone_active and self.deadzone_target is not None:
            target = self.deadzone_target.copy()
        else:
            target = self._get_gate_target(
                obs, self.locked_gate_idx, drone_pos, self.locked_gate_approach_sign
            )

        rel_target = target - drone_pos
        dist = float(np.linalg.norm(rel_target))
        return int(self.locked_gate_idx), target.astype(np.float32), rel_target.astype(np.float32), dist

    def _build_policy_obs(
        self, rel_target: np.ndarray, dist: float, obs: dict[str, NDArray[np.floating]]
    ) -> np.ndarray:
        drone_vel = np.asarray(obs["vel"], dtype=np.float32)
        ang_vel = np.asarray(obs["ang_vel"], dtype=np.float32)
        return np.concatenate(
            [rel_target, drone_vel, ang_vel, np.array([dist], dtype=np.float32)],
            dtype=np.float32,
        )

    def _sample_action(self, policy_obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.policy(obs_t)
            dist = Normal(mean, std)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            action = torch.tanh(raw_action) * self.max_speed

        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def _speed_profile(self, dist: float) -> float:
        if dist < 0.3:
            return self.base_speed_near
        if dist < 0.8:
            return self.base_speed_mid
        return self.base_speed_far

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        drone_pos = np.asarray(obs["pos"], dtype=np.float32)
        gate_idx, target, rel_target, dist = self._get_navigation_target(obs, drone_pos)

        if gate_idx == -1:
            return np.zeros(13, dtype=np.float32)

        if self.deadzone_active:
            base_des_vel = self.deadzone_speed * self.transit_direction
            smoothing = self.deadzone_smoothing
            residual_scale = self.deadzone_residual_scale
            speed_limit = self.deadzone_max_speed
        else:
            gate_dir = self._normalize(rel_target)
            base_des_vel = self._speed_profile(dist) * gate_dir
            smoothing = self.normal_smoothing
            residual_scale = self.normal_residual_scale
            speed_limit = self.max_speed

        policy_obs = self._build_policy_obs(rel_target, dist, obs)
        residual_vel, log_prob, value = self._sample_action(policy_obs)
        raw_des_vel = base_des_vel + residual_scale * residual_vel

        if self.prev_cmd_vel is None:
            des_vel = raw_des_vel
        else:
            des_vel = smoothing * self.prev_cmd_vel + (1.0 - smoothing) * raw_des_vel

        if drone_pos[2] < 0.25:
            des_vel = des_vel.copy()
            des_vel[2] += self.low_altitude_boost

        speed = float(np.linalg.norm(des_vel))
        if speed > speed_limit and speed > 1e-6:
            des_vel = des_vel / speed * speed_limit

        desired_z = max(float(target[2]), 0.35)
        target = target.copy()
        target[2] = (1.0 - self.target_z_blend) * drone_pos[2] + self.target_z_blend * desired_z

        yaw = float(np.arctan2(des_vel[1], des_vel[0] + 1e-6))

        action = np.array(
            [
                target[0],
                target[1],
                target[2],
                des_vel[0],
                des_vel[1],
                des_vel[2],
                0.0,
                0.0,
                0.0,
                yaw,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.prev_cmd_vel = des_vel.copy()
        self.pending_transition = {
            "obs": policy_obs,
            "action": residual_vel.copy(),
            "log_prob": log_prob,
            "value": value,
            "gate_idx_before": gate_idx,
            "locked_passed_before": self.locked_gate_passed,
            "dist_before": dist,
            "reward_target": target.copy(),
        }
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        if self.pending_transition is None:
            self._update_gate_lock(obs)
            return False

        prev_gate_idx = self.pending_transition["gate_idx_before"]
        prev_dist = self.pending_transition["dist_before"]
        reward_target = self.pending_transition["reward_target"]

        drone_pos = np.asarray(obs["pos"], dtype=np.float32)
        curr_dist = float(np.linalg.norm(reward_target - drone_pos)) if prev_gate_idx != -1 else 0.0

        gate_passed = 0.0
        if prev_gate_idx != -1 and not self.pending_transition["locked_passed_before"]:
            gate_passed = float(self._gate_passed_for_lock(obs, prev_gate_idx))

        progress = 0.0
        backward_penalty = 0.0
        if prev_gate_idx != -1:
            progress = float(prev_dist - curr_dist)
            if progress < 0.0:
                backward_penalty = -progress

        on_ground = float(drone_pos[2] < self.ground_z_thresh)
        ang_vel_norm = float(np.linalg.norm(np.asarray(obs["ang_vel"], dtype=np.float32)))

        vel_change_penalty = 0.0
        speed_change_penalty = 0.0
        action = np.asarray(action, dtype=np.float32)
        if self.prev_action is not None:
            vel_change_penalty = float(np.linalg.norm(action[3:6] - self.prev_action[3:6]))
            speed_change_penalty = float(
                abs(np.linalg.norm(action[3:6]) - np.linalg.norm(self.prev_action[3:6]))
            )

        unstable_motion = ang_vel_norm + vel_change_penalty + speed_change_penalty
        crashed = float(terminated or truncated)

        shaped_reward = (
            self.w_gate * gate_passed
            + self.w_progress * progress
            - self.w_backward * backward_penalty
            - self.w_ground * on_ground
            - self.w_unstable * unstable_motion
            - self.w_time
            - self.w_crash * crashed
        )

        self.episode_buffer.append(
            {
                "obs": self.pending_transition["obs"],
                "action": self.pending_transition["action"],
                "log_prob": self.pending_transition["log_prob"],
                "value": self.pending_transition["value"],
                "reward": shaped_reward,
                "done": bool(terminated or truncated),
            }
        )

        self.prev_action = action.copy()
        self.pending_transition = None
        self._update_gate_lock(obs)
        return False

    def episode_callback(self):
        if len(self.episode_buffer) == 0:
            return

        returns = []
        running_return = 0.0
        for transition in reversed(self.episode_buffer):
            if transition["done"]:
                running_return = 0.0
            running_return = transition["reward"] + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        obs_batch = torch.tensor(
            np.array([t["obs"] for t in self.episode_buffer], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        act_batch = torch.tensor(
            np.array([t["action"] for t in self.episode_buffer], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        values_old = torch.tensor(
            np.array([t["value"] for t in self.episode_buffer], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        returns_t = torch.tensor(
            np.array(returns, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        advantages = returns_t - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(5):
            mean, std, values = self.policy(obs_batch)
            dist = Normal(mean, std)

            scaled = torch.clamp(act_batch / self.max_speed, -0.999, 0.999)
            unsquashed = 0.5 * torch.log((1 + scaled) / (1 - scaled))
            log_probs = dist.log_prob(unsquashed).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = nn.functional.mse_loss(values, returns_t)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        torch.save(self.policy.state_dict(), self.model_path)

    def reset(self):
        self.episode_buffer: list[dict] = []
        self.pending_transition: dict | None = None
        self.prev_action = None
        self.prev_cmd_vel = None

        self.locked_gate_idx: int | None = None
        self.deadzone_active = False
        self.locked_gate_passed = False
        self.transit_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.locked_gate_approach_sign = 1.0
        self.deadzone_target: np.ndarray | None = None

        self._update_gate_lock(self._initial_obs)

    def episode_reset(self):
        self.reset()
