# This is the custom controller implementation. Another other customer controller implementations should be placed in separate files.
# Only requirement is to inherit from the base class Controller and implement its methods. Any type of logic can be used here including reinforcement learning.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.control.controller import Controller


class MyController(Controller):
    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)

        self.config = config

        # Speed profile
        self.max_speed = 1.2
        self.mid_speed = 1.0
        self.min_speed = 0.45

        # Distance thresholds
        self.d_far = 1.5
        self.d_near = 0.8
        self.d_gate = 0.35

        # Velocity smoothing
        self.normal_smoothing = 0.80
        self.gate_smoothing = 0.90
        self.turn_smoothing = 0.90

        # Turn cooldown after clearing a gate
        self.turn_cooldown_steps = 8
        self.turn_cooldown = 0

        # Gate-clearance logic
        self.exit_distance = 0.55
        self.max_exit_steps = 8
        self.exit_steps_left = 0
        self.in_exit_phase = False

        self.prev_gate_idx = int(obs["target_gate"])
        self.prev_gate_pos = None
        self.prev_passage_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.prev_des_vel = np.zeros(3, dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        drone_pos = np.array(obs["pos"], dtype=np.float32)
        gate_idx = int(obs["target_gate"])

        if gate_idx == -1:
            return np.zeros(13, dtype=np.float32)

        gate_pos = np.array(obs["gates_pos"][gate_idx][:3], dtype=np.float32)

        # Direction to current gate
        dir_vec = gate_pos - drone_pos
        dist = np.linalg.norm(dir_vec) + 1e-6
        dir_unit = dir_vec / dist

        # Detect gate switch. First clear the old gate before turning.
        if gate_idx != self.prev_gate_idx:
            old_gate_idx = self.prev_gate_idx
            if old_gate_idx != -1:
                self.prev_gate_pos = np.array(obs["gates_pos"][old_gate_idx][:3], dtype=np.float32)
            else:
                self.prev_gate_pos = gate_pos.copy()

            prev_speed = np.linalg.norm(self.prev_des_vel)
            if prev_speed > 0.05:
                self.prev_passage_dir = self.prev_des_vel / prev_speed
            else:
                self.prev_passage_dir = dir_unit.copy()

            self.in_exit_phase = True
            self.exit_steps_left = self.max_exit_steps
            self.turn_cooldown = 0
            self.prev_gate_idx = gate_idx

        raw_des_vel = None
        target = gate_pos.copy()

        # Phase A: clear previous gate using a fixed target beyond that gate
        if self.in_exit_phase and self.prev_gate_pos is not None:
            cleared = np.dot(drone_pos - self.prev_gate_pos, self.prev_passage_dir)

            if cleared < self.exit_distance and self.exit_steps_left > 0:
                self.exit_steps_left -= 1
                speed = self.min_speed
                smoothing = self.turn_smoothing
                raw_des_vel = speed * self.prev_passage_dir
                target = self.prev_gate_pos + self.exit_distance * self.prev_passage_dir
            else:
                self.in_exit_phase = False
                self.turn_cooldown = self.turn_cooldown_steps

        # Phase B/C: normal gate-seeking after gate is cleared
        if raw_des_vel is None:
            if dist > self.d_far:
                speed = self.max_speed
                smoothing = self.normal_smoothing
            elif dist > self.d_near:
                speed = self.mid_speed
                smoothing = self.gate_smoothing
            else:
                speed = self.min_speed
                smoothing = self.gate_smoothing

            if self.turn_cooldown > 0:
                speed = self.min_speed
                smoothing = self.turn_smoothing
                self.turn_cooldown -= 1

            raw_des_vel = speed * dir_unit

        # Smooth vertical target to reduce sudden drops/climbs
        target[2] = 0.8 * drone_pos[2] + 0.2 * target[2]

        des_vel = smoothing * self.prev_des_vel + (1.0 - smoothing) * raw_des_vel
        self.prev_des_vel = des_vel.astype(np.float32)

        yaw = np.arctan2(des_vel[1], des_vel[0])

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
        return False

    def episode_callback(self):
        pass

    def reset(self):
        self.turn_cooldown = 0
        self.exit_steps_left = 0
        self.in_exit_phase = False
        self.prev_gate_pos = None
        self.prev_passage_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.prev_des_vel = np.zeros(3, dtype=np.float32)

    def episode_reset(self):
        self.reset()