# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    camera_offset = torch.tensor([-3.0, -2.0, 1.4], dtype=torch.float32, device=env.device)
    lookat_offset = torch.tensor([0.0, 0.0, 0.35], dtype=torch.float32, device=env.device)
    camera_pos = robot_pos + camera_offset
    lookat_pos = robot_pos + lookat_offset
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=lookat_pos.cpu().numpy()
    )
