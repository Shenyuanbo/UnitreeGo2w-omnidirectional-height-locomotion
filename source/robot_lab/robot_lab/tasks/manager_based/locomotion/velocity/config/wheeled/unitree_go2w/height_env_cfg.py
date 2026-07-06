# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp


from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    CommandsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_GO2W_CFG  # isort: skip


DEFAULT_BASE_HEIGHT = 0.40
FINAL_HEIGHT_RANGE = (0.32, 0.43)
INITIAL_HEIGHT_RANGE = (0.38, 0.43)
STAGE_A_HEIGHT_RANGE = (DEFAULT_BASE_HEIGHT, DEFAULT_BASE_HEIGHT)

# ------------------------重写Command模块，加入高度command----------------------------------

class UnitreeGo2WBaseHeightCommand(CommandTerm):
    """Continuous base-height command used only by the Go2W height task."""

    cfg: UnitreeGo2WBaseHeightCommandCfg

    def __init__(self, cfg: UnitreeGo2WBaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.height_range[0] > cfg.height_range[1]:
            raise ValueError(f"Invalid height_range: {cfg.height_range}")
        self.height_command = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

    def __str__(self) -> str:
        return (
            "UnitreeGo2WBaseHeightCommand:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tHeight range: {self.cfg.height_range}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """Return the current base-height command in meters. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        low, high = self.cfg.height_range
        self.height_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def _update_command(self):
        pass


@configclass
class UnitreeGo2WBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for the continuous base-height command in meters."""

    class_type: type = UnitreeGo2WBaseHeightCommand

    height_range: tuple[float, float] = INITIAL_HEIGHT_RANGE
    """Sampled target base-height range in meters."""

@configclass
class UnitreeGo2WCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""

    base_height = UnitreeGo2WBaseHeightCommandCfg(
        resampling_time_range=(3.0, 5.0),
        height_range=INITIAL_HEIGHT_RANGE,
    )

# -----------------------------------------------------------------------------------------


def normalized_base_height_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_range: tuple[float, float] = FINAL_HEIGHT_RANGE,
) -> torch.Tensor:
    """Return base-height command normalized to roughly [-1, 1] using the final command range."""

    command = env.command_manager.get_command(command_name).float()
    center = 0.5 * (height_range[0] + height_range[1])
    half_range = max(0.5 * (height_range[1] - height_range[0]), 1e-6)
    return (command - center) / half_range


def track_base_height_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.025,
) -> torch.Tensor:
    """Reward tracking the commanded base height with an exponential kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name).float().squeeze(-1)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = target_height
        else:
            adjusted_target_height = torch.mean(ray_hits, dim=1) + target_height
    else:
        adjusted_target_height = target_height

    height_error = asset.data.root_pos_w[:, 2] - adjusted_target_height
    return torch.exp(-torch.square(height_error) / (std * std))


def stand_still_without_velocity_or_height_cmd(
    env: ManagerBasedRLEnv,
    velocity_command_name: str,
    height_command_name: str,
    velocity_command_threshold: float,
    height_command_threshold: float,
    default_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize default-pose deviation only when both velocity and height commands are neutral."""

    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)

    velocity_is_zero = (
        torch.linalg.norm(env.command_manager.get_command(velocity_command_name), dim=1) < velocity_command_threshold
    )
    height_is_neutral = (
        torch.abs(env.command_manager.get_command(height_command_name).squeeze(-1) - default_height)
        < height_command_threshold
    )
    reward *= velocity_is_zero & height_is_neutral
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def height_command_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    reward_term_name: str,
    initial_range: tuple[float, float] = INITIAL_HEIGHT_RANGE,
    final_range: tuple[float, float] = FINAL_HEIGHT_RANGE,
    range_step: float = 0.005,
    success_threshold: float = 0.75,
) -> torch.Tensor:
    """Expand the sampled height-command range when height tracking is good."""

    height_command_cfg = env.command_manager.get_term(command_name).cfg

    if env.common_step_counter == 0:
        height_command_cfg.height_range = initial_range

    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        reward_rate = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s

        if reward_rate > success_threshold * reward_term_cfg.weight:
            low, high = height_command_cfg.height_range
            new_low = max(final_range[0], low - range_step)
            new_high = min(final_range[1], high + range_step)
            height_command_cfg.height_range = (new_low, new_high)

    low, high = height_command_cfg.height_range
    return torch.tensor(high - low, device=env.device)


def velocity_command_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    lin_reward_term_name: str,
    yaw_reward_term_name: str,
    range_multiplier: tuple[float, float] = (0.2, 1.0),
    lin_range_step: float = 0.1,
    yaw_range_step: float = 0.1,
    success_threshold: float = 0.75,
) -> torch.Tensor:
    """Expand x/y/yaw velocity-command ranges when locomotion tracking is good."""

    command_cfg = env.command_manager.get_term(command_name).cfg
    ranges = command_cfg.ranges

    if env.common_step_counter == 0 or not hasattr(env, "_height_task_final_velocity_ranges"):
        env._height_task_final_velocity_ranges = {
            "lin_vel_x": tuple(ranges.lin_vel_x),
            "lin_vel_y": tuple(ranges.lin_vel_y),
            "ang_vel_z": tuple(ranges.ang_vel_z),
        }

        def scaled_range(range_value: tuple[float, float], multiplier: float) -> tuple[float, float]:
            return (range_value[0] * multiplier, range_value[1] * multiplier)

        ranges.lin_vel_x = scaled_range(env._height_task_final_velocity_ranges["lin_vel_x"], range_multiplier[0])
        ranges.lin_vel_y = scaled_range(env._height_task_final_velocity_ranges["lin_vel_y"], range_multiplier[0])
        ranges.ang_vel_z = scaled_range(env._height_task_final_velocity_ranges["ang_vel_z"], range_multiplier[0])

    if env.common_step_counter % env.max_episode_length == 0:
        lin_reward_cfg = env.reward_manager.get_term_cfg(lin_reward_term_name)
        yaw_reward_cfg = env.reward_manager.get_term_cfg(yaw_reward_term_name)
        lin_reward_rate = torch.mean(env.reward_manager._episode_sums[lin_reward_term_name][env_ids])
        yaw_reward_rate = torch.mean(env.reward_manager._episode_sums[yaw_reward_term_name][env_ids])
        lin_reward_rate /= env.max_episode_length_s
        yaw_reward_rate /= env.max_episode_length_s

        def expand_range(
            current_range: tuple[float, float],
            final_range: tuple[float, float],
            range_step: float,
        ) -> tuple[float, float]:
            low = max(final_range[0] * range_multiplier[1], current_range[0] - range_step)
            high = min(final_range[1] * range_multiplier[1], current_range[1] + range_step)
            return (low, high)

        if lin_reward_rate > success_threshold * lin_reward_cfg.weight:
            ranges.lin_vel_x = expand_range(
                tuple(ranges.lin_vel_x), env._height_task_final_velocity_ranges["lin_vel_x"], lin_range_step
            )
            ranges.lin_vel_y = expand_range(
                tuple(ranges.lin_vel_y), env._height_task_final_velocity_ranges["lin_vel_y"], lin_range_step
            )

        if yaw_reward_rate > success_threshold * yaw_reward_cfg.weight:
            ranges.ang_vel_z = expand_range(
                tuple(ranges.ang_vel_z), env._height_task_final_velocity_ranges["ang_vel_z"], yaw_range_step
            )

    command_range_width = (
        ranges.lin_vel_x[1] - ranges.lin_vel_x[0]
        + ranges.lin_vel_y[1] - ranges.lin_vel_y[0]
        + ranges.ang_vel_z[1] - ranges.ang_vel_z[0]
    ) / 3.0
    return torch.tensor(command_range_width, device=env.device)


@configclass
class UnitreeGo2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class UnitreeGo2WRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    track_base_height_command_exp = RewTerm(
        func=track_base_height_command_exp,
        weight=0.0,
        params={
            "command_name": "base_height",
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "std": 0.025,
        },
    )


@configclass
class UnitreeGo2WHeightEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()
    rewards: UnitreeGo2WRewardsCfg = UnitreeGo2WRewardsCfg()
    commands: UnitreeGo2WCommandsCfg = UnitreeGo2WCommandsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.policy.height_commands = ObsTerm(
            func=normalized_base_height_command,
            params={"command_name": "base_height", "height_range": FINAL_HEIGHT_RANGE},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.height_commands = ObsTerm(
            func=normalized_base_height_command,
            params={"command_name": "base_height", "height_range": FINAL_HEIGHT_RANGE},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # Weights: match UnitreeGo2WFlatEnvCfg/UnitreeGo2WRoughEnvCfg locomotion setup; height tracking is off for Stage A.
        self.rewards.is_terminated.weight = 0

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.body_lin_acc_l2.weight = 0

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_without_cmd.weight = -2
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.joint_mirror.weight = -0.05

        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.5e-4

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_base_height_command_exp.weight = 1.6

        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_gait.weight = 0
        self.rewards.upward.weight = 1

        # Root penalty params
        self.rewards.base_height_l2.params["target_height"] = DEFAULT_BASE_HEIGHT
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalty params
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names

        # Stand-still and symmetry params
        self.rewards.stand_still_without_cmd.func = stand_still_without_velocity_or_height_cmd
        self.rewards.stand_still_without_cmd.params = {
            "velocity_command_name": "base_velocity",
            "height_command_name": "base_height",
            "velocity_command_threshold": 0.1,
            "height_command_threshold": 0.015,
            "default_height": DEFAULT_BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
        }
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Contact params
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Command-tracking params
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.track_base_height_command_exp.params["command_name"] = "base_height"
        self.rewards.track_base_height_command_exp.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.track_base_height_command_exp.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.track_base_height_command_exp.params["std"] = 0.025

        # Foot/contact gait params
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WHeightEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        # self.curriculum.command_levels = CurrTerm(
        #     func=velocity_command_levels,
        #     params={
        #         "command_name": "base_velocity",
        #         "lin_reward_term_name": "track_lin_vel_xy_exp",
        #         "yaw_reward_term_name": "track_ang_vel_z_exp",
        #         "range_multiplier": (0.2, 1.0),
        #         "lin_range_step": 0.1,
        #         "yaw_range_step": 0.1,
        #         "success_threshold": 0.75,
        #     },
        # )

        self.curriculum.command_levels = None

        self.curriculum.height_command_levels = CurrTerm(
            func=height_command_levels,
            params={
                "command_name": "base_height",
                "reward_term_name": "track_base_height_command_exp",
                "initial_range": INITIAL_HEIGHT_RANGE,
                "final_range": FINAL_HEIGHT_RANGE,
                "range_step": 0.005,
                "success_threshold": 0.75,
            },
        )

        # ------------------------------Commands------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.base_height.height_range = INITIAL_HEIGHT_RANGE
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1, 1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.rel_standing_envs = 0.10
